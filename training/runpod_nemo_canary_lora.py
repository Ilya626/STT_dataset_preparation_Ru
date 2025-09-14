#!/usr/bin/env python
"""
Runpod/Linux launcher for native NeMo LoRA fine-tuning of Canary.

Goals (per guide):
- Use NeMo native adapter (LoRA) instead of custom layer wrapping.
- Attach adapters only to attention/MLP projections of the top encoder/decoder layers
  (linear_qkv, linear_proj, linear_fc1, linear_fc2).
- Keep changes minimal and explicit; rely on NeMo APIs.

Requirements on the machine (e.g., Runpod A6000 48GB):
- Python 3.11/3.12 recommended
- torch / torchaudio CUDA builds (e.g., torch 2.6 + cu12x)
- nemo_toolkit, lhotse, lightning

Example:
  python transcribe/training/runpod_nemo_canary_lora.py \
    --nemo /workspace/models/canary-1b-v2.nemo \
    --train /workspace/data/train.jsonl \
    --val /workspace/data/val.jsonl \
    --outdir /workspace/exp/canary_ru_lora_a6000 \
    --export /workspace/models/canary-ru-lora-a6000.nemo \
      --bs 8 --accum 1 --precision bf16 --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
      --enc_lora_layers 6 --dec_lora_layers 2

Notes:
- If your NeMo build lacks native adapter support, this script will raise with setup tips.
"""

import argparse
from pathlib import Path
import sys
import os
import shutil
import re

import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

def _import_nemo_encdec_model():
    try:
        from nemo.collections.asr.models import EncDecMultiTaskModel as _M
        return _M
    except Exception as e:
        msg = str(e)
        if "No module named 'hydra'" in msg or 'hydra' in msg.lower():
            try:
                import subprocess, sys as _sys
                subprocess.check_call([_sys.executable, '-m', 'pip', 'install', '-U', 'hydra-core', 'omegaconf'])
                from nemo.collections.asr.models import EncDecMultiTaskModel as _M
                return _M
            except Exception as e2:
                raise SystemExit(f"[ERR] NeMo import failed after installing hydra-core/omegaconf: {e2}")
        # 2) Missing pyannote
        if 'pyannote' in msg.lower():
            try:
                import subprocess, sys as _sys
                subprocess.check_call([_sys.executable, '-m', 'pip', 'install', '-U', 'pyannote.audio'])
                from nemo.collections.asr.models import EncDecMultiTaskModel as _M
                return _M
            except Exception:
                import sys as _sys, types as _types
                for name in ['pyannote', 'pyannote.audio', 'pyannote.core', 'pyannote.metrics']:
                    if name not in _sys.modules:
                        _sys.modules[name] = _types.ModuleType(name)
                try:
                    from nemo.collections.asr.models import EncDecMultiTaskModel as _M
                    return _M
                except Exception as e3:
                    raise SystemExit(f"[ERR] NeMo import failed (pyannote). Consider installing pyannote.audio or using Python 3.10 image. Underlying: {e3}")
        raise SystemExit(f"[ERR] Cannot import NeMo ASR models. Install nemo_toolkit. Underlying error: {e}")

EncDecMultiTaskModel = _import_nemo_encdec_model()

# Fallback custom LoRA injection (uses Windows script utilities)
# Ensure repo root is on sys.path so that `import transcribe.*` works even when
# running this script from inside the `transcribe` directory.
# Ensure repository root is on sys.path
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

try:
    from training.finetune_canary import (
        inject_lora_modules,
        merge_lora_into_linear,
    )
except Exception:
    inject_lora_modules = None  # type: ignore
    merge_lora_into_linear = None  # type: ignore


class CudaMemReport(Callback):
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
            f"[MEM:{tag}] alloc={alloc/gb:.2f}G reserved={reserv/gb:.2f}G free={free/gb:.2f}G total={total/gb:.2f}G peak={peak/gb:.2f}G"
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


class TerminalReport(Callback):
    """Print compact train/val metrics to terminal every N steps."""

    def __init__(self, every_n_steps: int = 50):
        super().__init__()
        self.every = int(max(1, every_n_steps))

    @staticmethod
    def _to_float(x):
        try:
            import torch

            if isinstance(x, torch.Tensor):
                x = x.detach().float().item()
        except Exception:
            pass
        try:
            return float(x)
        except Exception:
            return x

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step % self.every != 0:
            return
        # Collect a few common metrics if present
        metrics = dict(trainer.callback_metrics)
        keys = [k for k in ["loss", "train_loss", "val_loss", "wer", "cer"] if k in metrics]
        kv = []
        for k in keys:
            v = self._to_float(metrics[k])
            if isinstance(v, float):
                kv.append(f"{k}={v:.4f}")
            else:
                kv.append(f"{k}={v}")
        # Try to get current LR
        try:
            lr = trainer.optimizers[0].param_groups[0].get("lr", None)
            if lr is not None:
                kv.append(f"lr={float(lr):.6f}")
        except Exception:
            pass
        print("[REPORT] step=", step, " ", " ".join(kv))

    def on_validation_end(self, trainer, pl_module):
        metrics = dict(trainer.callback_metrics)
        msg = ["[VAL]"]
        for k in ["val_loss", "wer", "cer"]:
            if k in metrics:
                v = self._to_float(metrics[k])
                if isinstance(v, float):
                    msg.append(f"{k}={v:.4f}")
                else:
                    msg.append(f"{k}={v}")
        print(" ".join(msg))


def nemo_adapter_api_available(model) -> bool:
    return hasattr(model, "add_adapter")


def _flag_present(name: str) -> bool:
    return any(a == name or a.startswith(name + "=") for a in sys.argv[1:])


def apply_preset(args):
    if not getattr(args, "preset", None):
        return args
    preset = args.preset
    # Presets tuned for A6000 48GB
    if preset == "a6000-fast":
        preset_vals = dict(bs=12, accum=1, num_workers=12, lora_r=32, lora_alpha=64, lora_dropout=0.05)
    elif preset == "a6000-max":
        preset_vals = dict(bs=16, accum=1, num_workers=12, lora_r=32, lora_alpha=64, lora_dropout=0.05)
    else:
        return args
    # Only set values the user did not override explicitly on CLI
    for k, v in preset_vals.items():
        flag = "--" + k.replace("_", "-")
        if not _flag_present(flag):
            setattr(args, k, v)
    return args


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nemo", required=False, default="/workspace/models/canary-1b-v2.nemo", help="Path to base .nemo checkpoint (e.g., canary-1b-v2.nemo)")
    ap.add_argument("--auto_download", action="store_true", help="Auto-download .nemo if missing via Hugging Face")
    ap.add_argument("--model_id", default="nvidia/canary-1b-v2", help="HF repo id for Canary .nemo")
    ap.add_argument("--nemo_name", default="canary-1b-v2.nemo", help="Filename of .nemo artifact in the repo")
    ap.add_argument("--train", required=True, help="Train JSONL manifest (audio_filepath/text/source_lang/target_lang/pnc)")
    ap.add_argument("--val", required=True, help="Val JSONL manifest")
    ap.add_argument("--outdir", default="/workspace/exp/canary_ru_lora_a6000")
    ap.add_argument("--export", default="/workspace/models/canary-ru-lora-a6000.nemo")
    ap.add_argument("--adapter_name", default="ru_lora")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--enc_lora_layers", type=int, default=6, help="Number of top encoder layers to adapt")
    ap.add_argument("--dec_lora_layers", type=int, default=2, help="Number of top decoder layers to adapt")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--precision", default="bf16")
    ap.add_argument("--val_every_n_steps", type=int, default=200)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--max_steps", type=int, default=3000)
    ap.add_argument("--mem_report_steps", type=int, default=50)
    # Adaptive bucketing (Lhotse)
    ap.add_argument("--bucketing", action="store_true", help="Enable duration-based adaptive bucketing")
    ap.add_argument("--bucket_bins", default="2.5,5,10,20,40", help="Comma-separated duration bin edges in seconds")
    ap.add_argument("--bucket_bs", default="", help="Comma-separated batch sizes per bin; empty=auto")
    ap.add_argument("--prefetch_factor", type=int, default=2, help="DataLoader prefetch per worker")
    ap.add_argument("--val_bs", type=int, default=0, help="Override validation batch size (0=use train bs)")
    ap.add_argument("--term_report_steps", type=int, default=50, help="Print metrics to terminal every N steps")
    ap.add_argument("--log", choices=["csv", "tb", "none"], default="csv")
    ap.add_argument("--resume", default="", help="Optional path to Lightning checkpoint to resume from (ckpt_path)")
    ap.add_argument("--num_workers", type=int, default=8, help="DataLoader workers (Linux: 8; Windows: 0)")
    ap.add_argument("--preset", choices=["a6000-fast", "a6000-max"], default=None, help="Pre-tuned config profiles")
    # Scheduler / optimization best-practice
    ap.add_argument("--sched", choices=["cosine", "inverse_sqrt", "none"], default="inverse_sqrt")
    ap.add_argument("--warmup_steps", type=int, default=None)
    ap.add_argument("--warmup_ratio", type=float, default=None)
    ap.add_argument("--min_lr", type=float, default=1e-6)
    ap.add_argument("--fused_optim", action="store_true", help="Try fused AdamW if supported")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_duration", type=float, default=40.0, help="Max utterance duration (s) for Lhotse sampler")
    # Generalization & control
    ap.add_argument("--monitor", default="val_loss", help="Metric name to monitor for best/early stop")
    ap.add_argument("--early_stop", action="store_true", help="Enable EarlyStopping on --monitor")
    ap.add_argument("--es_patience", type=int, default=3)
    ap.add_argument("--es_min_delta", type=float, default=0.003, help="Relative min improvement (~0.3%)")
    ap.add_argument("--gradient_clip_val", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=None, help="Override base learning rate if supported")
    ap.add_argument("--weight_decay", type=float, default=None, help="Override weight decay if supported")
    args = ap.parse_args()

    # Apply preset-derived values (unless overridden explicitly on CLI)
    args = apply_preset(args)

    # Safer CUDA allocator by default
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # Seed
    try:
        seed_everything(int(args.seed), workers=True)
    except Exception:
        pass

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    Path(Path(args.export).parent).mkdir(parents=True, exist_ok=True)

    # Optionally download .nemo from HF if not present
    nemo_path = Path(args.nemo)
    if args.auto_download or not nemo_path.exists():
        try:
            from huggingface_hub import hf_hub_download
        except Exception as e:
            raise SystemExit("[ERR] huggingface_hub is required for auto-download. Install it or provide --nemo path.\n" + str(e))

        # Configure local caches if not set
        repo_root = Path(__file__).resolve().parents[2]
        os.environ.setdefault("HF_HOME", str(repo_root / ".hf"))
        os.environ.setdefault("TRANSFORMERS_CACHE", os.environ["HF_HOME"])  # compat
        os.environ.setdefault("HF_HUB_CACHE", str(Path(os.environ["HF_HOME"]) / "hub"))
        os.environ.setdefault("TORCH_HOME", str(repo_root / ".torch"))
        Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
        Path(os.environ["HF_HUB_CACHE"]).mkdir(parents=True, exist_ok=True)
        Path(os.environ["TORCH_HOME"]).mkdir(parents=True, exist_ok=True)

        print(f"[DL] Downloading {args.model_id}:{args.nemo_name} from Hugging Face...")
        token = os.environ.get("HF_TOKEN")
        try:
            downloaded = hf_hub_download(repo_id=args.model_id, filename=args.nemo_name, token=token)
        except Exception as e:
            print(f"[WARN] hf_hub_download failed: {e}. Trying direct HTTP...")
            url = f"https://huggingface.co/{args.model_id}/resolve/main/{args.nemo_name}"
            dest = Path(os.environ["HF_HOME"]) / args.nemo_name
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                import requests

                headers = {"Authorization": f"Bearer {token}"} if token else {}
                with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                    r.raise_for_status()
                    with open(dest, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                downloaded = str(dest)
            except Exception as e2:
                raise SystemExit(f"[ERR] Failed to download {url}: {e2}") from e2

        # If a destination path was requested, copy there; else use cache path
        if nemo_path.suffix.lower() == ".nemo":
            nemo_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copyfile(downloaded, nemo_path)
                print(f"[DL] Copied .nemo -> {nemo_path}")
            except Exception as e:
                print(f"[WARN] Could not copy to {nemo_path}: {e}. Using cache path instead: {downloaded}")
                nemo_path = Path(downloaded)
        else:
            nemo_path = Path(downloaded)

    print(f"[INFO] Restoring base model: {nemo_path}")
    model = EncDecMultiTaskModel.restore_from(
        restore_path=str(nemo_path),
        map_location="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Best-effort override of optimizer hyperparams if user requested
    try:
        cfg = getattr(model, "cfg", None) or getattr(model, "_cfg", None)
        if cfg is not None and hasattr(cfg, "optim"):
            if args.lr is not None and hasattr(cfg.optim, "lr"):
                old = cfg.optim.lr
                cfg.optim.lr = args.lr
                print(f"[OPT] Overriding lr: {old} -> {cfg.optim.lr}")
            if args.weight_decay is not None and hasattr(cfg.optim, "weight_decay"):
                old = cfg.optim.weight_decay
                cfg.optim.weight_decay = args.weight_decay
                print(f"[OPT] Overriding weight_decay: {old} -> {cfg.optim.weight_decay}")
            # Fused AdamW when available
            if args.fused_optim and hasattr(cfg.optim, "fused"):
                try:
                    cfg.optim.fused = True
                    print("[OPT] Using fused AdamW")
                except Exception:
                    pass
            # Scheduler override
            try:
                sched = None
                if args.sched == "cosine":
                    sched = {
                        "name": "CosineAnnealing",
                        "min_lr": float(args.min_lr),
                        "warmup_steps": int(args.warmup_steps) if args.warmup_steps else None,
                        "warmup_ratio": float(args.warmup_ratio) if args.warmup_ratio else None,
                    }
                elif args.sched == "inverse_sqrt":
                    sched = {
                        "name": "InverseSquareRootAnnealing",
                        "min_lr": float(args.min_lr),
                        "warmup_steps": int(args.warmup_steps) if args.warmup_steps else None,
                        "warmup_ratio": float(args.warmup_ratio) if args.warmup_ratio else None,
                    }
                elif args.sched == "none":
                    sched = {"name": "Null"}
                if sched is not None:
                    cfg.optim.sched = sched
                    print(f"[OPT] Scheduler -> {sched['name']}")
            except Exception as e:
                print("[OPT][WARN] Cannot set scheduler:", e)
    except Exception as e:
        print(f"[OPT][WARN] Could not override optimizer cfg: {e}")

    # Determine target module names for LoRA based on top-N layers
    mods = ("linear_qkv", "linear_proj", "linear_fc1", "linear_fc2")
    enc_ids = set()
    dec_ids = set()
    for n, _ in model.named_modules():
        m = re.search(r"encoder.*?(\d+)", n)
        if m:
            enc_ids.add(int(m.group(1)))
        m = re.search(r"transf_decoder.*?(\d+)", n)
        if m:
            dec_ids.add(int(m.group(1)))
    num_enc = max(enc_ids) + 1 if enc_ids else 0
    num_dec = max(dec_ids) + 1 if dec_ids else 0
    enc_start = max(0, num_enc - int(args.enc_lora_layers))
    dec_start = max(0, num_dec - int(args.dec_lora_layers))
    enabled_modules = []
    for n, _ in model.named_modules():
        if n.endswith(mods):
            m = re.search(r"encoder.*?(\d+)", n)
            if m and int(m.group(1)) >= enc_start:
                enabled_modules.append(n)
                continue
            m = re.search(r"transf_decoder.*?(\d+)", n)
            if m and int(m.group(1)) >= dec_start:
                enabled_modules.append(n)
    print(f"[ADAPTER] LoRA target modules: {len(enabled_modules)} (enc_last={args.enc_lora_layers} dec_last={args.dec_lora_layers})")

    # Prepare LoRA config (try native class first, fallback to Hydra-style dict)
    lora_cfg = None
    try:
        # Newer NeMo
        from nemo.core.adapters.lora import LoRAConfig as _LoRAConfig  # type: ignore
        lora_cfg = _LoRAConfig(
            r=int(args.lora_r), alpha=int(args.lora_alpha), dropout=float(args.lora_dropout),
            enabled_modules=enabled_modules,
        )
    except Exception:
        try:
            # Older NeMo
            from nemo.core.adapters import LoRAConfig as _LoRAConfig  # type: ignore
            lora_cfg = _LoRAConfig(
                r=int(args.lora_r), alpha=int(args.lora_alpha), dropout=float(args.lora_dropout),
                enabled_modules=enabled_modules,
            )
        except Exception:
            # Last resort: Hydra dict (works if add_adapter hydrates cfg)
            lora_cfg = {
                "_target_": "nemo.core.adapters.lora.LoRAConfig",
                "r": int(args.lora_r),
                "alpha": int(args.lora_alpha),
                "dropout": float(args.lora_dropout),
                "enabled_modules": enabled_modules,
            }

    used_adapter = False
    if nemo_adapter_api_available(model):
        print(
            f"[ADAPTER] Adding LoRA adapter '{args.adapter_name}' with cfg: r={args.lora_r} alpha={args.lora_alpha} dropout={args.lora_dropout}"
        )
        try:
            model.add_adapter(name=args.adapter_name, cfg=lora_cfg)
            # Enable the adapter; method name can differ by version.
            if hasattr(model, "set_enabled_adapters"):
                model.set_enabled_adapters([args.adapter_name], enabled=True)
            elif hasattr(model, "enable_adapters"):
                model.enable_adapters(adapters=[args.adapter_name])
            else:
                print(
                    "[WARN] Could not find an explicit enable_adapters API; proceeding (adapter may be active by default)"
                )
            # Freeze base weights, train only adapter params if API exposed
            try:
                for p in model.parameters():
                    p.requires_grad = False
                if hasattr(model, "adapter_parameters"):
                    for p in model.adapter_parameters():
                        p.requires_grad = True
            except Exception:
                pass
            used_adapter = True
        except Exception as e:
            print(
                "[WARN] Failed to add/enable LoRA via NeMo adapter API; falling back to custom injection:\n",
                e,
            )

    # Fallback: custom LoRA injection into Linear layers
    used_fallback_lora = False
    if not used_adapter:
        if inject_lora_modules is None:
            raise SystemExit(
                "[ERR] NeMo adapter API not available and fallback injector not importable."
            )
        print("[FALLBACK] Injecting custom LoRA into Linear layers (no NeMo adapters)")
        # Freeze base weights; LoRA layers hold trainable params
        for p in model.parameters():
            p.requires_grad = False
        patterns = [re.escape(n) + "$" for n in enabled_modules]
        replaced, used_fb = inject_lora_modules(
            model,
            patterns,
            r=int(args.lora_r),
            alpha=int(args.lora_alpha),
            dropout=float(args.lora_dropout),
            fallback_prefixes=(),
            debug_print=0,
        )
        print(f"[FALLBACK] LoRA injected modules: {replaced} (prefix_fallback={used_fb})")
        if replaced <= 0:
            raise SystemExit("[ERR] Fallback LoRA did not attach; adjust patterns")
        used_fallback_lora = True

    # Data config via Lhotse cuts produced on-the-fly by model's setup methods
    train_cfg = {
        "use_lhotse": True,
        "cuts_path": str(Path("data") / "train_cuts.jsonl.gz"),  # will be written if not exists
        "num_workers": int(args.num_workers),  # Linux; Windows can use 0
        "shuffle": True,
        "batch_size": int(args.bs),
        "pretokenize": False,
        "pin_memory": True,
        "max_duration": float(args.max_duration),
        "persistent_workers": bool(int(args.num_workers) > 0),
        "prefetch_factor": int(args.prefetch_factor),
        # keys below are for non-Lhotse manifest paths and ignored by Lhotse; avoid warnings
    }
    val_cfg = {
        "use_lhotse": True,
        "cuts_path": str(Path("data") / "val_cuts.jsonl.gz"),
        "num_workers": int(args.num_workers),
        "shuffle": False,
        "batch_size": int(args.val_bs) if int(args.val_bs) > 0 else int(args.bs),
        "pretokenize": False,
        "pin_memory": True,
        "max_duration": float(args.max_duration),
        "persistent_workers": bool(int(args.num_workers) > 0),
        "prefetch_factor": int(args.prefetch_factor),
    }

    if args.bucketing:
        def _parse_bins(s: str):
            return [float(x.strip()) for x in s.split(',') if x.strip()]
        def _parse_bs(s: str):
            items = [x.strip() for x in s.split(',') if x.strip()]
            return [int(x) for x in items] if items else []
        bins = _parse_bins(args.bucket_bins)
        bs_list = _parse_bs(args.bucket_bs)
        if not bs_list:
            base = int(args.bs)
            factors = [1.3, 1.0, 0.8, 0.6, 0.5]
            if len(bins) < len(factors):
                factors = factors[:len(bins)]
            elif len(bins) > len(factors):
                factors += [factors[-1]] * (len(bins) - len(factors))
            bs_list = [max(1, int(base * f)) for f in factors]
        if len(bs_list) != len(bins):
            raise SystemExit("--bucket_bs must have the same number of items as --bucket_bins")
        for cfg in (train_cfg, val_cfg):
            cfg.update({
                "use_bucketing": True,
                "bucketing_sampler": True,
                "bucket_duration_bins": bins,
                "bucket_batch_size": bs_list,
                "bucket_buffer_size": 20000,
            })

    # The model's setup_* functions in our Windows script generate Lhotse cuts from JSONL.
    # Here we mimic the same behavior by calling them with extended config that includes input_manifest.
    def _ensure_cuts(path_jsonl, out_path):
        try:
            from training.finetune_canary import build_cuts_from_jsonl  # reuse helper
            cuts, _ = build_cuts_from_jsonl(path_jsonl)
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            cuts.to_file(out_path)
        except Exception as e:
            print(f"[WARN] Could not prebuild cuts via helper: {e}. Proceeding; model may build internally if supported.")

    _ensure_cuts(args.train, train_cfg["cuts_path"])
    _ensure_cuts(args.val, val_cfg["cuts_path"])

    model.setup_training_data(train_cfg)
    model.setup_validation_data(val_cfg)

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    logger = None
    if args.log == "csv":
        logger = CSVLogger(save_dir=args.outdir, name="pl_logs")
    elif args.log == "tb":
        logger = TensorBoardLogger(save_dir=args.outdir, name="tb_logs")

    # Step checkpoints
    ckpt_cb = ModelCheckpoint(
        dirpath=args.outdir,
        filename="step{step:06d}",
        save_last=True,
        save_top_k=-1,
        every_n_train_steps=int(args.save_every),
    )
    # Best checkpoint by monitored metric
    best_cb = ModelCheckpoint(
        dirpath=args.outdir,
        filename="best",
        save_top_k=1,
        monitor=args.monitor,
        mode="min",
    )
    mem_cb = CudaMemReport(every_n_steps=int(args.mem_report_steps))
    term_cb = TerminalReport(every_n_steps=int(args.term_report_steps))
    callbacks = [ckpt_cb, best_cb, mem_cb, term_cb]
    if args.early_stop:
        es = EarlyStopping(monitor=args.monitor, mode="min", patience=int(args.es_patience), min_delta=float(args.es_min_delta))
        callbacks.append(es)
    if logger is not None:
        callbacks.append(LearningRateMonitor(logging_interval="step"))

    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        max_steps=int(args.max_steps),
        accumulate_grad_batches=int(args.accum),
        precision=args.precision,
        gradient_clip_val=float(args.gradient_clip_val),
        logger=logger,
        enable_checkpointing=True,
        callbacks=callbacks,
        val_check_interval=int(args.val_every_n_steps),
        log_every_n_steps=10,
        num_sanity_val_steps=0,
    )

    print(
        "[TRAIN] Starting fit() with LoRA (mode="
        + ("adapter" if used_adapter else "fallback")
        + f") | bs={args.bs} accum={args.accum} workers={args.num_workers} precision={args.precision}"
    )
    ckpt_path = args.resume if args.resume else None
    trainer.fit(model, ckpt_path=ckpt_path)

    print("[EXPORT] Saving model to .nemo")
    try:
        if not used_adapter and merge_lora_into_linear is not None:
            print("[EXPORT] Merging fallback LoRA weights into base Linear layers ...")
            merge_lora_into_linear(model)
        model.save_to(str(args.export))
    except Exception as e:
        print("[ERR] Export failed:", e)
        sys.exit(1)
    print(f"[OK] Exported: {args.export}")


if __name__ == "__main__":
    main()
