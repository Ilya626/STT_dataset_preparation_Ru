from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ANALYSIS_DIR = Path("analysis_output")
DEFAULT_OUT_PATH = Path("final_mix.jsonl")
DEFAULT_DATASET_DIR = Path(
    f"fine-tuning-dataset-{datetime.now().strftime('%Y%m%d')}"
)
EASY_FRACTION = 0.2
DIFFICULTY_SCALE = 5.0
NUM_BINS = 10


def resolve_path(p: Path) -> Path:
    """Resolve ``p`` relative to the script directory if not absolute."""
    return p if p.is_absolute() else (BASE_DIR / p).resolve()


def _clean_ref(text: str) -> str:
    """Remove special artifacts from reference text for manifest writing.

    - Remove backslashes '\\'
    - Replace double hyphens "--" with a single space
    - Collapse multiple spaces and strip
    """
    if not isinstance(text, str):
        return ""
    s = text.replace("\\", " ")
    s = s.replace("--", " ")
    # Collapse whitespace
    s = " ".join(s.split())
    return s


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter datasets and build a mixed training split",
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=DEFAULT_ANALYSIS_DIR,
        help="Directory with analysis outputs",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_PATH,
        help="Path to write the resulting mix (JSONL)",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Directory to write manifest.json and copied audio",
    )
    parser.add_argument(
        "--target-min",
        type=int,
        default=20000,
        help="Minimum desired number of samples in the mix",
    )
    parser.add_argument(
        "--target-max",
        type=int,
        default=25000,
        help="Maximum desired number of samples in the mix",
    )
    parser.add_argument(
        "--difficulty-scale",
        type=float,
        default=DIFFICULTY_SCALE,
        help="Multiplier applied to difficulty before normalisation",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Seed for reproducible sampling",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_dataset(path: Path) -> pd.DataFrame:
    easy = read_jsonl(path / "easy.jsonl") if (path / "easy.jsonl").exists() else []
    difficult = (
        read_jsonl(path / "difficult.jsonl")
        if (path / "difficult.jsonl").exists()
        else []
    )
    for r in easy:
        r["category"] = "easy"
    for r in difficult:
        r["category"] = "difficult"
    return pd.DataFrame(easy + difficult)


def stratified_sample(df: pd.DataFrame, n: int, random_state: int) -> pd.DataFrame:
    if n >= len(df) or len(df) == 0:
        return df
    bins = np.linspace(df["pc1"].min(), df["pc1"].max(), NUM_BINS + 1)
    df = df.copy()
    df["bin"] = pd.cut(df["pc1"], bins, include_lowest=True)
    per_bin = max(1, n // NUM_BINS)
    parts = []
    for _, group in df.groupby("bin"):
        k = min(len(group), per_bin)
        if k > 0:
            parts.append(group.sample(k, random_state=random_state))
    sample = pd.concat(parts)
    if len(sample) < n:
        remaining = n - len(sample)
        others = df.drop(sample.index)
        if not others.empty:
            sample = pd.concat(
                [sample, others.sample(min(remaining, len(others)), random_state=random_state)]
            )
    return sample


def save_dataset(df: pd.DataFrame, out_dir: Path) -> None:
    """Copy audio files and write a manifest.json for the final dataset."""

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.json"
    total_duration = 0.0
    with manifest.open("w", encoding="utf-8") as mf:
        for row in df.itertuples(index=False):
            src = Path(row.audio)
            if not src.is_absolute():
                src = resolve_path(src)
            dest_dir = out_dir / row.dataset
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / src.name
            try:
                shutil.copy2(src, dest)
            except FileNotFoundError:
                print(f"Warning: missing audio file {src}")
                continue
            # Clean reference text for manifest
            text = _clean_ref(getattr(row, "ref", ""))
            entry = {
                "audio": str(dest.relative_to(out_dir)),
                "text": text,
                "dataset": row.dataset,
            }
            # Accumulate duration if available; otherwise try to infer from file
            dur = getattr(row, "duration", None)
            try:
                if isinstance(dur, (int, float)) and float(dur) > 0:
                    total_duration += float(dur)
                else:
                    try:
                        import soundfile as sf  # type: ignore
                        info = sf.info(str(dest))
                        if info.frames and info.samplerate:
                            total_duration += float(info.frames) / float(info.samplerate)
                    except Exception:
                        try:
                            import wave, contextlib  # type: ignore
                            with contextlib.closing(wave.open(str(dest), "rb")) as wf:
                                fr = wf.getframerate(); nf = wf.getnframes()
                                if fr and nf:
                                    total_duration += float(nf) / float(fr)
                        except Exception:
                            pass
            except Exception:
                pass
            mf.write(json.dumps(entry, ensure_ascii=False) + "\n")
    # Report total duration
    hours = total_duration / 3600.0
    print(f"Saved manifest and audio files to {out_dir}")
    print(f"Total duration: {total_duration:.1f} sec ({hours:.2f} h)")


def main() -> None:
    args = parse_args()
    analysis_dir = resolve_path(args.analysis_dir)
    out_path = resolve_path(args.out)
    dataset_dir = resolve_path(args.dataset_dir)

    if not analysis_dir.exists():
        print(f"No analysis directory found at {analysis_dir}")
        return

    datasets: Dict[str, pd.DataFrame] = {}
    total_easy = 0
    total_difficult = 0
    for ds_dir in sorted(p for p in analysis_dir.iterdir() if p.is_dir()):
        df = load_dataset(ds_dir)
        if df.empty:
            continue
        datasets[ds_dir.name] = df
        total_easy += int((df["category"] == "easy").sum())
        total_difficult += int((df["category"] == "difficult").sum())

    print(f"Total easy: {total_easy}, total difficult: {total_difficult}")

    small, large = {}, {}
    for name, df in datasets.items():
        if len(df) < 1000:
            small[name] = df
        else:
            large[name] = df

    selected: List[pd.DataFrame] = []
    for name, df in small.items():
        df = df.copy()
        df["dataset"] = name
        selected.append(df)

    remaining_min = args.target_min - sum(len(df) for df in selected)
    remaining_max = args.target_max - sum(len(df) for df in selected)

    if large and remaining_max > 0:
        combined = pd.concat(large.values(), ignore_index=True)
        combined["semantic_diff"] = 1.0 - combined["semantic"]
        combined["difficulty_scaled"] = combined["difficulty"] * args.difficulty_scale
        feats = combined[["wer", "semantic_diff", "difficulty_scaled"]]
        means = feats.mean()
        stds = feats.std().replace(0, 1)

        total_available = sum(len(df) for df in large.values())
        remaining = min(remaining_max, total_available)
        quotas = {
            name: int(remaining * len(df) / total_available) for name, df in large.items()
        }

        for name, df in large.items():
            df = df.copy()
            df["semantic_diff"] = 1.0 - df["semantic"]
            df["difficulty_scaled"] = df["difficulty"] * args.difficulty_scale
            df["wer_norm"] = (df["wer"] - means["wer"]) / stds["wer"]
            df["semantic_norm"] = (df["semantic_diff"] - means["semantic_diff"]) / stds[
                "semantic_diff"
            ]
            df["difficulty_norm"] = (
                df["difficulty_scaled"] - means["difficulty_scaled"]
            ) / stds["difficulty_scaled"]
            pca = PCA(n_components=2, random_state=args.random_state)
            comps = pca.fit_transform(
                df[["wer_norm", "semantic_norm", "difficulty_norm"]]
            )
            df["pc1"], df["pc2"] = comps[:, 0], comps[:, 1]
            quota = quotas.get(name, 0)
            need_diff = int(quota * (1 - EASY_FRACTION))
            need_easy = quota - need_diff
            diff_df = df[df["category"] == "difficult"]
            easy_df = df[df["category"] == "easy"]
            parts = []
            if need_diff > 0 and not diff_df.empty:
                parts.append(stratified_sample(diff_df, need_diff, args.random_state))
            if need_easy > 0 and not easy_df.empty:
                parts.append(stratified_sample(easy_df, need_easy, args.random_state))
            if parts:
                out_df = pd.concat(parts)
                out_df["dataset"] = name
                selected.append(out_df)

    final = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()
    drop_cols = [
        c
        for c in final.columns
        if c.startswith("pc")
        or c.endswith("_norm")
        or c in {"semantic_diff", "difficulty_scaled", "bin"}
    ]
    final.drop(columns=drop_cols, inplace=True, errors="ignore")

    save_dataset(final, dataset_dir)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for _, row in final.iterrows():
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")

    print(f"Saved {len(final)} records to {out_path}")
    for name in datasets:
        count = int((final["dataset"] == name).sum())
        print(f"{name}: {count}")


if __name__ == "__main__":
    main()
