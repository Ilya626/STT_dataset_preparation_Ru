#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import soundfile as sf


# Read as float32 everywhere (consistent with project scripts)
_sf_read = sf.read


def _sf_read_float32(*args, **kwargs):
    kwargs.setdefault("dtype", "float32")
    return _sf_read(*args, **kwargs)


sf.read = _sf_read_float32


def ensure_wav_mono16k(data: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
    """Return mono float32 audio resampled to 16 kHz."""
    if getattr(data, "ndim", 1) > 1:
        data = data.mean(axis=1)
    data = np.asarray(data, dtype=np.float32)
    if sr != 16000:
        try:
            import librosa  # type: ignore

            data = librosa.resample(y=data, orig_sr=sr, target_sr=16000)
            sr = 16000
        except Exception:
            # Fallback to scipy if available
            try:
                from scipy.signal import resample_poly  # type: ignore

                # Use polyphase resampling for speed/quality
                # Compute rational approximation for sr->16000
                from math import gcd

                g = gcd(sr, 16000)
                up = 16000 // g
                down = sr // g
                data = resample_poly(data, up, down).astype(np.float32)
                sr = 16000
            except Exception as exc:
                raise RuntimeError(
                    "Need resample to 16k but no resampler available (librosa/scipy)"
                ) from exc
    return data, 16000


def save_wav(path: Path, data: np.ndarray, sr: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, sr, subtype="PCM_16", format="WAV")


def load_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


@dataclass
class Item:
    dataset: str
    audio_rel: str  # relative to in_dir (e.g., dataset_name\\<hash>.wav)
    text: str
    base: str  # basename without extension
    abs_audio: Path
    duration: Optional[float] = None


def build_duration_index(src_manifest_path: Path) -> Dict[str, float]:
    """Map basename (without extension) -> duration from source manifest.jsonl."""
    idx: Dict[str, float] = {}
    for row in load_jsonl(src_manifest_path):
        ap = str(row.get("audio_filepath", ""))
        dur = row.get("duration")
        if not ap or dur is None:
            continue
        base = Path(ap).stem
        idx[base] = float(dur)
    return idx


def read_duration_from_wav(path: Path) -> float:
    data, sr = sf.read(str(path))
    if sr <= 0:
        raise ValueError(f"Invalid sample rate in {path}")
    return float(len(data) / sr)


def pick_sep(seps: List[Tuple[str, Path]], rnd: random.Random) -> Tuple[str, np.ndarray]:
    """Return (text, audio_array_16k)."""
    word, p = rnd.choice(seps)
    arr, sr = sf.read(str(p))
    arr, _ = ensure_wav_mono16k(arr, sr)
    return word, arr


def normalize_space(s: str) -> str:
    return " ".join(str(s).split())


def merge_batches(
    items: List[Item],
    out_dataset_dir: Path,
    sep_bank: List[Tuple[str, Path]],
    rnd: random.Random,
    max_dur: float,
    sep_planning_sec: float,
) -> List[dict]:
    """Merge items into batches <= max_dur, inserting sep words.

    Returns list of manifest rows for merged items.
    """
    out_rows: List[dict] = []

    batch: List[Item] = []
    batch_planned_dur = 0.0

    def flush_batch():
        nonlocal batch, batch_planned_dur
        if not batch:
            return

        audio_parts: List[np.ndarray] = []
        texts: List[str] = []
        for i, it in enumerate(batch):
            # Append segment audio
            a, sr = sf.read(str(it.abs_audio))
            a, _ = ensure_wav_mono16k(a, sr)
            audio_parts.append(a)
            texts.append(normalize_space(it.text))

            # Insert separator between segments
            if i < len(batch) - 1:
                sep_word, sep_arr = pick_sep(sep_bank, rnd)
                audio_parts.append(sep_arr)
                texts.append(f"{sep_word}.")

        merged = np.concatenate(audio_parts) if audio_parts else np.zeros(0, dtype=np.float32)
        # Name output deterministically by joined bases
        joined_key = "+".join([it.base for it in batch])
        out_name = f"{abs(hash(joined_key)) & 0xFFFFFFFFFFFFFFFF:016x}.wav"
        out_path = out_dataset_dir / out_name
        save_wav(out_path, merged, 16000)

        out_text = normalize_space(" ".join(texts))
        out_rows.append(
            {
                "audio": f"{out_dataset_dir.name}\\{out_name}",
                "text": out_text,
                "dataset": out_dataset_dir.name,
            }
        )

        batch = []
        batch_planned_dur = 0.0

    for it in items:
        d = it.duration or 0.0
        # If empty batch, just add it if d <= max_dur; otherwise flush as-is (single long item)
        if not batch:
            if d <= max_dur:
                batch.append(it)
                batch_planned_dur = d
            else:
                # Save as-is without separators
                out_dataset_dir.mkdir(parents=True, exist_ok=True)
                a, sr = sf.read(str(it.abs_audio))
                a, _ = ensure_wav_mono16k(a, sr)
                out_name = f"{it.base}.wav"
                out_path = out_dataset_dir / out_name
                save_wav(out_path, a, 16000)
                out_rows.append(
                    {
                        "audio": f"{out_dataset_dir.name}\\{out_name}",
                        "text": normalize_space(it.text),
                        "dataset": out_dataset_dir.name,
                    }
                )
            continue

        # Adding a new item adds its duration plus one separator planning duration
        prospective = batch_planned_dur + sep_planning_sec + d
        if prospective <= max_dur:
            batch.append(it)
            batch_planned_dur = prospective
        else:
            # Close and start new batch
            flush_batch()
            if d <= max_dur:
                batch.append(it)
                batch_planned_dur = d
            else:
                # Single long item again
                out_dataset_dir.mkdir(parents=True, exist_ok=True)
                a, sr = sf.read(str(it.abs_audio))
                a, _ = ensure_wav_mono16k(a, sr)
                out_name = f"{it.base}.wav"
                out_path = out_dataset_dir / out_name
                save_wav(out_path, a, 16000)
                out_rows.append(
                    {
                        "audio": f"{out_dataset_dir.name}\\{out_name}",
                        "text": normalize_space(it.text),
                        "dataset": out_dataset_dir.name,
                    }
                )

    flush_batch()
    return out_rows


def main():
    p = argparse.ArgumentParser(
        description=(
            "Merge short segments up to 35s with random separator words inserted between them. "
            "Durations for planning are read from source manifests in data_wav (or computed from WAV)."
        )
    )
    p.add_argument("--in-dir", type=Path, default=Path("fine-tuning-dataset-20250913"), help="Input dataset directory containing manifest.json and audio subfolders")
    p.add_argument("--out-dir", type=Path, default=None, help="Output directory for merged dataset (default: <in-dir>-merged35)")
    p.add_argument("--source-manifests", type=Path, default=Path("data_wav"), help="Directory with source dataset folders containing manifest.jsonl (for durations)")
    p.add_argument("--sep-dir", type=Path, default=Path("sep"), help="Directory with separator WAV files; filename stem is the word")
    p.add_argument("--max-dur", type=float, default=35.0, help="Maximum merged duration in seconds")
    p.add_argument("--sep-plan-sec", type=float, default=1.5, help="Planned duration per separator for packing (audio uses real file)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    p.add_argument("--skip-datasets", type=str, nargs="*", default=["Ilya626"], help="Dataset names containing any of these substrings will be copied without merging")
    p.add_argument("--limit-per-dataset", type=int, default=None, help="Optional limit of items per dataset for a quick test run")
    args = p.parse_args()

    in_dir: Path = args.in_dir
    out_dir: Path = args.out_dir or Path(str(in_dir) + "-merged35")
    src_root: Path = args.source_manifests
    sep_dir: Path = args.sep_dir
    max_dur: float = float(args.max_dur)
    sep_plan_sec: float = float(args.sep_plan_sec)
    rnd = random.Random(args.seed)
    skip_substrings = [s.lower() for s in (args.skip_datasets or [])]

    manifest_path = in_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Input manifest not found: {manifest_path}")

    # Load sep bank (word, path)
    sep_bank: List[Tuple[str, Path]] = []
    for wav in sep_dir.glob("*.wav"):
        word = wav.stem
        if not word:
            continue
        sep_bank.append((word, wav))
    if not sep_bank:
        raise FileNotFoundError(f"No .wav files found in sep dir: {sep_dir}")

    # Load final manifest and group by dataset
    by_dataset: Dict[str, List[Item]] = defaultdict(list)
    for row in load_jsonl(manifest_path):
        ds = str(row.get("dataset", "")).strip()
        audio_rel = str(row.get("audio", "")).strip()
        text = str(row.get("text", "")).strip()
        if not ds or not audio_rel:
            continue
        base = Path(audio_rel).stem
        # Resolve audio path; handle special case where files live under dataset_portable/
        abs_audio = in_dir / audio_rel
        if not abs_audio.exists():
            alt = in_dir / "dataset_portable" / audio_rel
            if alt.exists():
                abs_audio = alt
        by_dataset[ds].append(Item(dataset=ds, audio_rel=audio_rel, text=text, base=base, abs_audio=abs_audio))

    out_dir.mkdir(parents=True, exist_ok=True)

    out_manifest_rows: List[dict] = []

    for ds, items in by_dataset.items():
        print(f"Processing dataset: {ds} (items: {len(items)})")
        ds_src_manifest = src_root / ds / "manifest.jsonl"
        dur_index: Dict[str, float] = {}
        if ds_src_manifest.exists():
            dur_index = build_duration_index(ds_src_manifest)
        else:
            print(f"  WARNING: source manifest not found for {ds}: {ds_src_manifest}")

        # Filter out items with missing audio files
        missing = [it for it in items if not it.abs_audio.exists()]
        if missing:
            print(f"  WARNING: {len(missing)} items missing audio files; they will be skipped")
        items = [it for it in items if it.abs_audio.exists()]

        # Fill durations
        for it in items:
            it.duration = dur_index.get(it.base)
            if it.duration is None:
                try:
                    it.duration = read_duration_from_wav(it.abs_audio)
                except Exception:
                    it.duration = None

        # Optionally limit for a quick dry run (applies to both paths)
        if args.limit_per_dataset is not None and args.limit_per_dataset > 0:
            items = items[: args.limit_per_dataset]
            print(f"  Limiting to first {len(items)} items for testing")

        # Decide whether to skip merging
        ds_lower = ds.lower()
        should_skip = any(key in ds_lower for key in skip_substrings)

        out_dataset_dir = out_dir / ds
        out_dataset_dir.mkdir(parents=True, exist_ok=True)

        if should_skip:
            print(f"  Skipping merge (copy-as-is): {ds}")
            for it in items:
                # Copy or re-encode to ensure 16k mono PCM16
                if not it.abs_audio.exists():
                    continue
                a, sr = sf.read(str(it.abs_audio))
                a, _ = ensure_wav_mono16k(a, sr)
                out_path = out_dataset_dir / f"{it.base}.wav"
                save_wav(out_path, a, 16000)
                out_manifest_rows.append(
                    {
                        "audio": f"{ds}\\{out_path.name}",
                        "text": it.text,
                        "dataset": ds,
                    }
                )
            continue

        # Sort items in the same order as manifest (already loaded order)
        # Pack into batches and merge with separators
        out_rows = merge_batches(items, out_dataset_dir, sep_bank, rnd, max_dur, sep_plan_sec)
        out_manifest_rows.extend(out_rows)

    # Write output manifest
    out_manifest_path = out_dir / "manifest.json"
    with out_manifest_path.open("w", encoding="utf-8") as f:
        for row in out_manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Done. Wrote {len(out_manifest_rows)} rows to {out_manifest_path}")


if __name__ == "__main__":
    main()
