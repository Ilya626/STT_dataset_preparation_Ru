from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

# configuration defaults
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PREDS_DIR = Path("predictions")
DEFAULT_OUT_DIR = Path("analysis_output")
PRED_FILE_NAME = "predictions.jsonl"
DEFAULT_TAIL_FRACTION = 0.05


def resolve_path(p: Path) -> Path:
    """Resolve ``p`` relative to the script directory if not absolute."""

    return p if p.is_absolute() else (BASE_DIR / p).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse prediction metrics")
    parser.add_argument(
        "--preds-dir",
        type=Path,
        default=DEFAULT_PREDS_DIR,
        help="Directory containing prediction subfolders",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory to store analysis results",
    )
    parser.add_argument(
        "--tail-fraction",
        type=float,
        default=DEFAULT_TAIL_FRACTION,
        help="Fraction of WER tail to drop at each end",
    )
    args = parser.parse_args()
    args.preds_dir = resolve_path(args.preds_dir)
    args.out_dir = resolve_path(args.out_dir)
    return args


def compute_semantic_similarity(refs: List[str], hyps: List[str]) -> "np.ndarray":
    """Return cosine similarity between reference and hypothesis texts."""

    import numpy as np
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    emb_ref = model.encode(refs)
    emb_hyp = model.encode(hyps)
    # cosine similarity
    sims = (
        (emb_ref * emb_hyp).sum(axis=1)
        / (np.linalg.norm(emb_ref, axis=1) * np.linalg.norm(emb_hyp, axis=1))
    )
    return sims


def analyse_dataset(pred_file: Path, out_dir: Path, tail_fraction: float) -> None:
    import numpy as np
    from jiwer import wer
    import matplotlib.pyplot as plt

    rows: List[dict] = []
    refs: List[str] = []
    hyps: List[str] = []
    with pred_file.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            ref = row.get("ref") or row.get("text") or ""
            hyp = row.get("hyp") or row.get("pred_text") or ""
            rows.append({"audio": row.get("audio"), "ref": ref, "hyp": hyp})
            refs.append(ref)
            hyps.append(hyp)

    w = wer(refs, hyps) if refs else 0.0
    ser_flags = [ref.strip() != hyp.strip() for ref, hyp in zip(refs, hyps)]
    ser = float(np.mean(ser_flags)) if ser_flags else 0.0

    sample_wers = [wer([r], [h]) for r, h in zip(refs, hyps)]
    semantic_sims = compute_semantic_similarity(refs, hyps)

    for row, swer, sim, sf in zip(rows, sample_wers, semantic_sims, ser_flags):
        row.update({"wer": swer, "semantic": float(sim), "ser": int(sf)})

    out_dir.mkdir(parents=True, exist_ok=True)

    wers = np.array(sample_wers)
    q_low = np.quantile(wers, tail_fraction)
    q_high = np.quantile(wers, 1 - tail_fraction)
    filtered = [r for r in rows if q_low <= r["wer"] <= q_high]
    filtered.sort(key=lambda r: r["wer"])
    cut = int(len(filtered) * 0.3)
    easy = filtered[:cut]
    difficult = filtered[cut:]

    def write_jsonl(path: Path, data: List[dict]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for row in data:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_jsonl(out_dir / "easy.jsonl", easy)
    write_jsonl(out_dir / "difficult.jsonl", difficult)

    plt.figure()
    plt.hist(wers, bins=50)
    plt.axvline(q_low, color="red", linestyle="dashed")
    plt.axvline(q_high, color="red", linestyle="dashed")
    plt.title("WER distribution")
    plt.xlabel("WER")
    plt.ylabel("Count")
    plt.savefig(out_dir / "wer_distribution.png")

    plt.figure()
    plt.hist(semantic_sims, bins=50)
    plt.title("Semantic similarity distribution")
    plt.xlabel("Cosine similarity")
    plt.ylabel("Count")
    plt.savefig(out_dir / "semantic_distribution.png")

    plt.figure()
    plt.scatter(wers, semantic_sims, alpha=0.5)
    plt.xlabel("WER")
    plt.ylabel("Semantic similarity")
    plt.title("WER vs Semantic similarity")
    plt.savefig(out_dir / "wer_vs_semantic.png")

    print(f"WER: {w:.4f}")
    print(f"SER: {ser:.4f}")
    print(f"Saved analysis to {out_dir}")


def main() -> None:
    args = parse_args()

    if not args.preds_dir.exists():
        print(f"No predictions directory found at {args.preds_dir}")
        return

    for pred_dir in sorted(p for p in args.preds_dir.iterdir() if p.is_dir()):
        pred_file = pred_dir / PRED_FILE_NAME
        if not pred_file.exists():
            print(f"Skipping {pred_dir.name}, missing {PRED_FILE_NAME}")
            continue
        out_dir = args.out_dir / pred_dir.name
        print(f"Analyzing {pred_dir.name}...")
        analyse_dataset(pred_file, out_dir, args.tail_fraction)


if __name__ == "__main__":
    main()
