#!/usr/bin/env python
"""LoRA fine-tuning script for NVIDIA Canary models using NeMo and Lhotse.

The script loads a `.nemo` checkpoint, injects LoRA adapters into linear
layers, prepares training and validation data from JSONL manifests and runs a
basic training loop.  Upon completion (or Ctrl+C), the LoRA weights are merged
back into the base model and exported as a new `.nemo` file.
"""

import argparse
import json
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from nemo.collections.asr.models import EncDecMultiTaskModel
from lhotse import CutSet, Recording, RecordingSet, SupervisionSegment, SupervisionSet


class LoRALinear(nn.Module):
    """Simple LoRA wrapper for `nn.Linear` layers."""

    def __init__(self, base_linear: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        if not isinstance(base_linear, nn.Linear):
            raise TypeError("base_linear must be nn.Linear")
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.weight = base_linear.weight
        self.bias = base_linear.bias
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.lora_A = nn.Linear(self.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, self.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)
        self.dropout = nn.Dropout(p=dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        base = F.linear(x, self.weight, self.bias)
        lora = self.lora_B(self.lora_A(self.dropout(x)))
        return base + self.scaling * lora


def inject_lora_modules(
    model: nn.Module,
    patterns,
    r=8,
    alpha=16,
    dropout=0.0,
    fallback_prefixes=("encoder", "transf_decoder"),
    debug_print=0,
):
    """Replace `nn.Linear` layers matching patterns with `LoRALinear` wrappers."""

    compiled = [re.compile(p) for p in patterns]
    replaced = 0
    linear_names = []

    def _match(name):
        return any(p.search(name) for p in compiled)

    for module_name, module in model.named_modules():
        for child_name, child in module.named_children():
            full_name = f"{module_name}.{child_name}" if module_name else child_name
            if isinstance(child, nn.Linear):
                linear_names.append(full_name)

    if debug_print > 0:
        print("[DEBUG] First linear modules:")
        for n in linear_names[:debug_print]:
            print("  -", n)

    for module_name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full_name = f"{module_name}.{child_name}" if module_name else child_name
            if isinstance(child, nn.Linear) and _match(full_name):
                setattr(module, child_name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
                replaced += 1

    if replaced > 0:
        return replaced, False

    for module_name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full_name = f"{module_name}.{child_name}" if module_name else child_name
            if isinstance(child, nn.Linear) and any(pref in full_name for pref in fallback_prefixes):
                setattr(module, child_name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
                replaced += 1

    return replaced, True


def merge_lora_into_linear(model: nn.Module):
    """Merge LoRA weights into their base `nn.Linear` modules."""

    count = 0
    for module_name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                with torch.no_grad():
                    delta = child.lora_B.weight @ child.lora_A.weight
                    child.weight += child.scaling * delta
                new_lin = nn.Linear(child.in_features, child.out_features, bias=(child.bias is not None))
                with torch.no_grad():
                    new_lin.weight.copy_(child.weight)
                    if child.bias is not None:
                        new_lin.bias.copy_(child.bias)
                setattr(module, child_name, new_lin)
                count += 1
    print(f"[LoRA] merged into base Linear: {count}")
    return count


def build_cuts_from_jsonl(jsonl_path: str):
    """Convert a JSONL manifest to a Lhotse `CutSet`."""

    recs = []
    sups = []
    n = 0
    base_dir = Path(jsonl_path).resolve().parent
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            o = json.loads(line)
            wav_raw = o["audio_filepath"]
            wav_path = Path(str(wav_raw))
            if not wav_path.is_absolute():
                wav_path = (base_dir / wav_path).resolve()
            wav = str(wav_path)
            txt = o.get("text", "")
            src = o.get("source_lang", "ru")
            tgt = o.get("target_lang", "ru")
            pnc = o.get("pnc", "yes")
            rid = f"r{i:08d}"
            rec = Recording.from_file(wav, recording_id=rid)
            sup = SupervisionSegment(
                id=f"{rid}-0",
                recording_id=rid,
                start=0.0,
                duration=rec.duration,
                channel=0,
                text=txt,
                language=tgt,
                custom={"source_lang": src, "target_lang": tgt, "pnc": pnc},
            )
            recs.append(rec)
            sups.append(sup)
            n += 1

    cuts = CutSet.from_manifests(
        RecordingSet.from_recordings(recs),
        SupervisionSet.from_segments(sups),
    ).trim_to_supervisions()

    def add_cut_lang(cut):
        sup = cut.supervisions[0]
        src = getattr(sup, "custom", {}).get("source_lang", "ru")
        tgt = getattr(sup, "custom", {}).get("target_lang", "ru")
        pnc = getattr(sup, "custom", {}).get("pnc", "yes")
        try:
            return cut.with_custom({**(getattr(cut, "custom", {}) or {}), "source_lang": src, "target_lang": tgt, "pnc": pnc})
        except Exception:
            try:
                cur = getattr(cut, "custom", {}) or {}
                cur.update({"source_lang": src, "target_lang": tgt, "pnc": pnc})
                cut.custom = cur
            except Exception:
                pass
            return cut

    cuts = cuts.map(add_cut_lang)
    return cuts, n


class CudaMemReport(Callback):
    """Periodic CUDA memory reporting callback."""

    def __init__(self, every_n_steps=50):
        super().__init__()
        self.every = int(max(1, every_n_steps))

    def _report(self, tag):
        if not torch.cuda.is_available():
            return
        dev = torch.cuda.current_device()
        total = torch.cuda.get_device_properties(dev).total_memory
        try:
            free, total_rt = torch.cuda.mem_get_info()
        except Exception:
            free, total_rt = 0, 0
        alloc = torch.cuda.memory_allocated(dev)
        reserv = torch.cuda.memory_reserved(dev)
        peak = torch.cuda.max_memory_allocated(dev)
        gb = 1024 ** 3
        print(
            f"[MEM:{tag}] alloc={alloc/gb:.2f}G  reserved={reserv/gb:.2f}G  free={free/gb:.2f}G  total={total/gb:.2f}G  peak={peak/gb:.2f}G"
        )

    def on_fit_start(self, trainer, pl_module):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._report("fit_start")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (trainer.global_step % self.every) == 0:
            self._report(f"step{trainer.global_step}")

    def on_validation_end(self, trainer, pl_module):
        self._report("val_end")

    def on_fit_end(self, trainer, pl_module):
        self._report("fit_end")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nemo", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--precision", default="bf16")
    ap.add_argument("--max_steps", type=int, default=3000)
    ap.add_argument("--val_every_n_steps", type=int, default=200)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.0)
    ap.add_argument("--debug_linear", type=int, default=0)
    ap.add_argument("--exp_dir", default="exp/canary_ru_lora_win")
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--export_nemo", default="models/nemo/canary-ru-lora-merged.nemo")
    ap.add_argument("--mem_report_steps", type=int, default=50)
    ap.add_argument(
        "--log",
        choices=["csv", "tb", "none"],
        default="csv",
        help="csv=CSVLogger, tb=TensorBoard, none=no logger",
    )
    args = ap.parse_args()

    Path(args.exp_dir).mkdir(parents=True, exist_ok=True)
    Path("models/nemo").mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading .nemo from {args.nemo}")
    model = EncDecMultiTaskModel.restore_from(
        restore_path=args.nemo,
        map_location="cuda" if torch.cuda.is_available() else "cpu",
    )

    for p in model.parameters():
        p.requires_grad = False
    patterns = [
        r"(self_)?attn\.(q_proj|k_proj|v_proj|out_proj)",
        r"linear_qkv",
        r"linear_proj",
        r"linear_fc1",
        r"linear_fc2",
        r"\bfc1\b",
        r"\bfc2\b",
        r"linear1",
        r"linear2",
        r"\bproj\b",
        r"to_q\b",
        r"to_k\b",
        r"to_v\b",
        r"to_out\b",
    ]
    replaced, used_fallback = inject_lora_modules(
        model,
        patterns,
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        fallback_prefixes=("encoder", "transf_decoder"),
        debug_print=args.debug_linear,
    )
    print(f"[LoRA] injected modules: {replaced} (fallback={used_fallback})")
    assert replaced > 0, "LoRA did not attach to any linear layer; adjust patterns"

    print("[INFO] Building & saving Lhotse CutSets to disk...")
    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True, parents=True)
    cuts_tr, n_tr = build_cuts_from_jsonl(args.train)
    cuts_va, n_va = build_cuts_from_jsonl(args.val)
    train_cuts_path = out_dir / "train_cuts.jsonl.gz"
    val_cuts_path = out_dir / "val_cuts.jsonl.gz"
    cuts_tr.to_file(train_cuts_path)
    cuts_va.to_file(val_cuts_path)
    print(f"[INFO] Cuts saved: train≈{n_tr} -> {train_cuts_path}, val≈{n_va} -> {val_cuts_path}")

    train_cfg = {
        "use_lhotse": True,
        "cuts_path": str(train_cuts_path),
        "num_workers": 0,
        "bucketing_sampler": False,
        "shuffle": True,
        "batch_size": args.bs,
        "pretokenize": False,
    }
    val_cfg = {
        "use_lhotse": True,
        "cuts_path": str(val_cuts_path),
        "num_workers": 0,
        "bucketing_sampler": False,
        "shuffle": False,
        "batch_size": args.bs,
        "pretokenize": False,
    }
    model.setup_training_data(train_cfg)
    model.setup_validation_data(val_cfg)

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    logger = None
    if args.log == "csv":
        logger = CSVLogger(save_dir=args.exp_dir, name="pl_logs")
    elif args.log == "tb":
        logger = TensorBoardLogger(save_dir=args.exp_dir, name="tb_logs")

    ckpt_cb = ModelCheckpoint(
        dirpath=args.exp_dir,
        filename="step{step:06d}",
        save_last=True,
        save_top_k=-1,
        every_n_train_steps=args.save_every,
        monitor=None,
    )
    mem_cb = CudaMemReport(every_n_steps=args.mem_report_steps)
    callbacks = [ckpt_cb, mem_cb]
    if logger is not None:
        callbacks.append(LearningRateMonitor(logging_interval="step"))

    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        max_steps=args.max_steps,
        accumulate_grad_batches=args.accum,
        precision=args.precision,
        logger=logger,
        enable_checkpointing=True,
        callbacks=callbacks,
        val_check_interval=args.val_every_n_steps,
        log_every_n_steps=10,
    )

    try:
        trainer.fit(model)
    except KeyboardInterrupt:
        int_ckpt = Path(args.exp_dir) / "interrupt.ckpt"
        print(f"\n[CTRL-C] Saving interrupt checkpoint -> {int_ckpt}")
        trainer.save_checkpoint(str(int_ckpt))
        try:
            print("[CTRL-C] Merging LoRA and exporting .nemo...")
            merge_lora_into_linear(model)
            out_nemo = Path(args.export_nemo).with_name("canary-ru-lora-merged-interrupt.nemo")
            out_nemo.parent.mkdir(parents=True, exist_ok=True)
            model.save_to(str(out_nemo))
            print(f"[CTRL-C] Exported -> {out_nemo}")
        except Exception as e:  # pragma: no cover - best effort
            print("[CTRL-C] Nemo export failed:", e)
        raise

    print("[INFO] Merging LoRA into base Linear and exporting NEMO...")
    merge_lora_into_linear(model)
    out_nemo = Path(args.export_nemo)
    out_nemo.parent.mkdir(parents=True, exist_ok=True)
    model.save_to(str(out_nemo))
    print(f"[OK] Exported merged .nemo -> {out_nemo}")


if __name__ == "__main__":
    main()
