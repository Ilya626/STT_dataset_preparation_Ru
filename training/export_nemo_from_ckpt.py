#!/usr/bin/env python
"""
Export NeMo .nemo from a Lightning checkpoint (.ckpt).

Usage examples (Runpod):

  # Explicit paths
  python training/export_nemo_from_ckpt.py \
    --base /workspace/models/canary-1b-v2.nemo \
    --ckpt /workspace/exp/canary_partial_bs90_new/best.ckpt \
    --out  /workspace/models/canary-partial-bs90-new.nemo

  # Auto-pick best.ckpt from an experiment dir (fallback: best-*.ckpt → last.ckpt)
  python training/export_nemo_from_ckpt.py \
    --base /workspace/models/canary-1b-v2.nemo \
    --ckpt-dir /workspace/exp/canary_partial_bs90_new \
    --out  /workspace/models/canary-partial-bs90-new.nemo

Notes
- Runs on CPU by default; GPU is not required.
- Loads checkpoint's 'state_dict' into the base model with strict=False to ignore
  any auxiliary keys. Prints missing/unexpected keys counts for visibility.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    from nemo.collections.asr.models import EncDecMultiTaskModel
except Exception as e:  # pragma: no cover
    print("[ERR] Install nemo_toolkit to use this script:", e, file=sys.stderr)
    sys.exit(1)


def pick_ckpt_from_dir(d: Path) -> Path:
    # Priority: best.ckpt → newest best-*.ckpt → last.ckpt
    cand = d / "best.ckpt"
    if cand.exists():
        return cand
    bests = sorted(d.glob("best-*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if bests:
        return bests[0]
    last = d / "last.ckpt"
    if last.exists():
        return last
    raise SystemExit(f"No best.ckpt / best-*.ckpt / last.ckpt found in {d}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--base", required=True, help="Path to base .nemo (e.g., canary-1b-v2.nemo)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--ckpt", help="Path to Lightning .ckpt to load")
    g.add_argument("--ckpt-dir", help="Experiment dir to auto-pick best.ckpt from")
    ap.add_argument("--out", required=True, help="Output .nemo path")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Map model/ckpt to this device while loading")
    ap.add_argument("--strict", action="store_true", help="Use strict=True when loading state_dict")
    args = ap.parse_args()

    base = Path(args.base).expanduser().resolve()
    if not base.exists():
        raise SystemExit(f"Base .nemo not found: {base}")

    if args.ckpt:
        ckpt = Path(args.ckpt).expanduser().resolve()
    else:
        ckpt = pick_ckpt_from_dir(Path(args.ckpt_dir).expanduser().resolve())
    if not ckpt.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt}")

    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[LOAD] base .nemo: {base}")
    model = EncDecMultiTaskModel.restore_from(restore_path=str(base), map_location=args.device)

    print(f"[CKPT] loading: {ckpt}")
    obj = torch.load(str(ckpt), map_location=args.device)
    sd = obj.get("state_dict", obj)
    missing, unexpected = model.load_state_dict(sd, strict=args.strict)
    print(f"[CKPT] loaded (missing={len(missing)}, unexpected={len(unexpected)})")
    if missing:
        print("  missing sample:", missing[:5])
    if unexpected:
        print("  unexpected sample:", unexpected[:5])

    print(f"[SAVE] -> {out}")
    model.save_to(str(out))
    print(f"[OK] saved .nemo: {out}")


if __name__ == "__main__":
    main()

