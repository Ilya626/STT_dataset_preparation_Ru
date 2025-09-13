#!/usr/bin/env python
"""
Unified Runpod launcher for Canary fine-tuning.

Features:
- Downloads dataset ZIP from a URL (e.g., Hugging Face) and unpacks it
- Normalizes/creates JSONL manifests with absolute audio paths
- Splits into train/val (unless explicit manifests provided)
- Runs either LoRA or Partial fine-tuning using existing scripts

Examples (Runpod A6000):

  # LoRA with dataset ZIP
  python training/runpod_canary_launcher.py \
    --dataset_url https://huggingface.co/USERNAME/REPO/resolve/main/fine-tuning-dataset.zip \
    --method lora \
    --nemo /workspace/models/canary-1b-v2.nemo \
    --outdir /workspace/exp/canary_ru_lora \
    --export /workspace/models/canary-ru-lora.nemo \
    --preset a6000-fast

  # Partial unfreeze using an already unpacked folder
  python training/runpod_canary_launcher.py \
    --dataset_dir /workspace/data/fine-tuning-dataset-20250913 \
    --method partial \
    --nemo /workspace/models/canary-1b-v2.nemo \
    --outdir /workspace/exp/canary_partial \
    --export /workspace/models/canary-partial.nemo \
    --unfreeze_encoder_last 4 --unfreeze_decoder_last 2 --unfreeze_head \
    --preset a6000-fast

Notes:
- If the dataset directory contains a single manifest named "manifest.json" or
  "manifest.jsonl", it will be used. Otherwise, provide --manifest_file.
- If your manifest uses keys like {"audio": "...", "text": "..."} or
  {"audio": "...", "ref": "..."}, this script converts it into the expected
  format with keys {"audio_filepath": "...", "text": "..."} and absolute paths.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Iterable, Tuple


def _log(msg: str):
    print(f"[UNIFY] {msg}")


def download_zip(url: str, dest_dir: Path, token: str | None = None) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = url.split("/")[-1] or "dataset.zip"
    dst = dest_dir / filename
    # Reuse existing file if present and non-empty
    if dst.exists() and dst.stat().st_size > 0:
        _log(f"ZIP уже скачан: {dst} (пропускаем загрузку)")
        return dst
    _log(f"Downloading ZIP from: {url}\n -> {dst}")
    try:
        import requests

        headers = {"Authorization": f"Bearer {token}"} if token else {}
        with requests.get(url, headers=headers, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(dst, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
    except Exception as e:
        raise SystemExit(f"Failed to download {url}: {e}")
    return dst


def unzip(zip_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    # If already extracted, don't do it again
    pre_entries = [p for p in out_dir.iterdir() if not p.name.startswith('.')]
    if pre_entries:
        _log(f"Распаковка уже есть: {out_dir} (пропускаем)")
    else:
        _log(f"Unzipping {zip_path} -> {out_dir}")
        try:
            import zipfile

            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(out_dir)
        except Exception as e:
            raise SystemExit(f"Failed to unzip {zip_path}: {e}")

    # If a single top-level folder present, return it; else return out_dir
    entries = [p for p in out_dir.iterdir() if not p.name.startswith('.')]
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return out_dir


def _iter_manifest_lines(path: Path) -> Iterable[dict]:
    with path.open('r', encoding='utf-8') as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except json.JSONDecodeError as e:
                raise SystemExit(f"Bad JSON on line {ln} of {path}: {e}")


def normalize_manifest(src_manifest: Path, base_dir: Path, out_manifest: Path) -> int:
    """
    Convert various manifest flavors to JSONL with keys:
      - audio_filepath: absolute path
      - text: transcript string
    """
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_manifest.open('w', encoding='utf-8') as fo:
        for obj in _iter_manifest_lines(src_manifest):
            # Pick text field: prefer 'text', else 'ref', else 'transcript'
            text = obj.get('text')
            if text is None:
                text = obj.get('ref')
            if text is None:
                text = obj.get('transcript')
            if text is None:
                text = ""

            # Pick audio path: prefer 'audio_filepath', else 'audio', else 'wav'
            ap = obj.get('audio_filepath') or obj.get('audio') or obj.get('wav')
            if ap is None:
                raise SystemExit(f"Manifest missing audio path field in {src_manifest}")
            ap = str(ap).replace('\\', '/')
            p = Path(ap)
            if not p.is_absolute():
                p = (base_dir / p).resolve()
            if not p.exists():
                alt1 = (base_dir / Path(ap)).resolve()
                if alt1.exists():
                    p = alt1

            out = {"audio_filepath": str(p), "text": text}
            fo.write(json.dumps(out, ensure_ascii=False) + "\n")
            n += 1
    return n


def find_manifest(dataset_dir: Path, manifest_file: str | None) -> Path:
    if manifest_file:
        cand = dataset_dir / manifest_file
        if not cand.exists():
            raise SystemExit(f"Manifest file not found: {cand}")
        return cand
    for name in ("manifest.jsonl", "manifest.json"):
        cand = dataset_dir / name
        if cand.exists():
            return cand
    for ext in ("*.jsonl", "*.json"):
        matches = list(dataset_dir.glob(ext))
        if matches:
            return matches[0]
    raise SystemExit(f"Could not locate a manifest file in {dataset_dir}")


def split_manifest(src_jsonl: Path, out_train: Path, out_val: Path, val_ratio: float, seed: int = 42) -> Tuple[int, int]:
    lines = src_jsonl.read_text(encoding='utf-8').splitlines()
    rng = random.Random(seed)
    rng.shuffle(lines)
    n = len(lines)
    n_val = max(1, int(n * val_ratio)) if n > 0 else 0
    val = lines[:n_val]
    train = lines[n_val:]
    out_train.parent.mkdir(parents=True, exist_ok=True)
    out_val.parent.mkdir(parents=True, exist_ok=True)
    out_train.write_text("\n".join(train) + ("\n" if train else ""), encoding='utf-8')
    out_val.write_text("\n".join(val) + ("\n" if val else ""), encoding='utf-8')
    return len(train), len(val)


def run_subprocess(argv: list[str]) -> int:
    import subprocess

    _log("Executing: " + " ".join(argv))
    return subprocess.call(argv)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    # Dataset acquisition
    ap.add_argument("--dataset_url", default=None, help="HTTP(S) URL to a ZIP containing WAVs and a manifest")
    ap.add_argument("--dataset_dir", default=None, help="Use an already-unpacked dataset directory instead of --dataset_url")
    ap.add_argument("--manifest_file", default=None, help="Manifest filename inside the dataset directory (default: auto-detect)")
    ap.add_argument("--data_root", default="/workspace/data", help="Where to place the downloaded/unzipped dataset")
    ap.add_argument("--work_dir", default="/workspace/data/work", help="Where to place normalized/split manifests")
    ap.add_argument("--val_ratio", type=float, default=0.02, help="Fraction for validation split if a single manifest is provided")
    ap.add_argument("--hf_token", default=None, help="Hugging Face token for private ZIP (or set HF_TOKEN env)")
    ap.add_argument("--seed", type=int, default=42)

    # Training choice
    ap.add_argument("--method", choices=["lora", "partial"], required=True)

    # Common training args
    ap.add_argument("--nemo", required=True, help="Path to base .nemo checkpoint")
    ap.add_argument("--outdir", required=True, help="Directory for checkpoints/logs")
    ap.add_argument("--export", required=True, help="Path to write the exported .nemo")
    ap.add_argument("--preset", choices=["a6000-fast", "a6000-max"], default=None)
    ap.add_argument("--precision", default="bf16-mixed")
    ap.add_argument("--bs", type=int, default=None)
    ap.add_argument("--accum", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=None)

    # LoRA-specific overrides (optional)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    # Partial-specific overrides (optional)
    ap.add_argument("--unfreeze_encoder_last", type=int, default=4)
    ap.add_argument("--unfreeze_decoder_last", type=int, default=2)
    ap.add_argument("--unfreeze_head", action="store_true")
    ap.add_argument("--train_norms", action="store_true")
    ap.add_argument("--train_bias", action="store_true")
    ap.add_argument("--grad_ckpt", action="store_true")

    args, extra = ap.parse_known_args()

    data_root = Path(args.data_root).resolve()
    work_dir = Path(args.work_dir).resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Resolve dataset directory
    if args.dataset_url and args.dataset_dir:
        raise SystemExit("Provide either --dataset_url or --dataset_dir, not both")
    if not args.dataset_url and not args.dataset_dir:
        raise SystemExit("One of --dataset_url or --dataset_dir is required")

    if args.dataset_url:
        import os as _os
        token = args.hf_token or _os.environ.get("HF_TOKEN")
        zip_path = download_zip(args.dataset_url, data_root, token=token)
        dataset_dir = unzip(zip_path, data_root / (zip_path.stem + "_unzipped"))
    else:
        dataset_dir = Path(args.dataset_dir).resolve()
        if not dataset_dir.exists():
            raise SystemExit(f"Dataset dir does not exist: {dataset_dir}")

    _log(f"Dataset directory: {dataset_dir}")

    # Pick or create manifests
    src_manifest = find_manifest(dataset_dir, args.manifest_file)
    norm_manifest = work_dir / "manifest.normalized.jsonl"
    n_norm = normalize_manifest(src_manifest, dataset_dir, norm_manifest)
    _log(f"Normalized manifest -> {norm_manifest} ({n_norm} rows)")

    train_jsonl = work_dir / "train_portable.jsonl"
    val_jsonl = work_dir / "val_portable.jsonl"
    n_tr, n_va = split_manifest(norm_manifest, train_jsonl, val_jsonl, args.val_ratio, args.seed)
    _log(f"Split: train={n_tr}, val={n_va} (val_ratio={args.val_ratio})")

    # Dispatch to selected training method by calling existing scripts
    repo_root = Path(__file__).resolve().parents[1]
    py = sys.executable

    if args.method == "lora":
        cmd = [
            py, str(repo_root / "training" / "runpod_nemo_canary_lora.py"),
            "--nemo", args.nemo,
            "--train", str(train_jsonl),
            "--val", str(val_jsonl),
            "--outdir", args.outdir,
            "--export", args.export,
        ]
        if args.preset:
            cmd += ["--preset", args.preset]
        if args.precision:
            cmd += ["--precision", args.precision]
        if args.bs is not None:
            cmd += ["--bs", str(args.bs)]
        if args.accum is not None:
            cmd += ["--accum", str(args.accum)]
        if args.num_workers is not None:
            cmd += ["--num_workers", str(args.num_workers)]
        # LoRA hyperparams
        cmd += [
            "--lora_r", str(args.lora_r),
            "--lora_alpha", str(args.lora_alpha),
            "--lora_dropout", str(args.lora_dropout),
        ]
    else:  # partial
        cmd = [
            py, str(repo_root / "training" / "runpod_nemo_canary_partial.py"),
            "--nemo", args.nemo,
            "--train", str(train_jsonl),
            "--val", str(val_jsonl),
            "--outdir", args.outdir,
            "--export", args.export,
        ]
        if args.preset:
            cmd += ["--preset", args.preset]
        if args.precision:
            cmd += ["--precision", args.precision]
        if args.bs is not None:
            cmd += ["--bs", str(args.bs)]
        if args.accum is not None:
            cmd += ["--accum", str(args.accum)]
        if args.num_workers is not None:
            cmd += ["--num_workers", str(args.num_workers)]
        # Partial unfreeze knobs
        cmd += [
            "--unfreeze_encoder_last", str(args.unfreeze_encoder_last),
            "--unfreeze_decoder_last", str(args.unfreeze_decoder_last),
        ]
        if args.unfreeze_head:
            cmd.append("--unfreeze_head")
        if args.train_norms:
            cmd.append("--train_norms")
        if args.train_bias:
            cmd.append("--train_bias")
        if args.grad_ckpt:
            cmd.append("--grad_ckpt")

    code = run_subprocess(cmd + extra)
    if code != 0:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
