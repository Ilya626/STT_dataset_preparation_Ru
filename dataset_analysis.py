from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List

# configuration defaults
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PREDS_DIR = Path("predictions")
DEFAULT_OUT_DIR = Path("analysis_output")
PRED_FILE_NAME = "predictions.jsonl"
DEFAULT_TAIL_FRACTION = 0.05
# Semantics carry more weight than raw WER in the difficulty score
DIFFICULTY_WER_WEIGHT = 0.3
DIFFICULTY_SEM_WEIGHT = 1.0 - DIFFICULTY_WER_WEIGHT


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
        help="Fraction of difficulty tail to drop at each end",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=None,
        help="If set, skip predictions with confidence below this value",
    )
    parser.add_argument(
        "--semantic-weight",
        type=float,
        default=2.0,
        help="Weight for semantic similarity contribution to difficulty",
    )
    parser.add_argument(
        "--wer-weight",
        type=float,
        default=1.0,
        help="Weight for WER contribution to difficulty",
    )
    args = parser.parse_args()
    args.preds_dir = resolve_path(args.preds_dir)
    args.out_dir = resolve_path(args.out_dir)
    return args


def _norm(text: str) -> str:
    """Normalize text by replacing ``ё``/``Ё`` with ``е``/``Е``."""

    return text.replace("ё", "е").replace("Ё", "Е")


def compute_semantic_similarity(refs: List[str], hyps: List[str]) -> "np.ndarray":
    """Return cosine similarity between reference and hypothesis texts."""

    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except Exception:  # pragma: no cover - optional dependency
        # Fallback if sentence transformers isn't available.
        return np.zeros(len(refs)) if refs else []

    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    emb_ref = model.encode(refs)
    emb_hyp = model.encode(hyps)
    # cosine similarity
    sims = (
        (emb_ref * emb_hyp).sum(axis=1)
        / (np.linalg.norm(emb_ref, axis=1) * np.linalg.norm(emb_hyp, axis=1))
    )
    return sims


def analyse_dataset(
    pred_file: Path,
    out_dir: Path,
    tail_fraction: float,
    confidence_threshold: float | None = None,
) -> None:
    import numpy as np
    from jiwer import wer
    try:
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - optional dependency
        plt = None

    rows: List[dict] = []
    refs: List[str] = []
    hyps: List[str] = []
    with pred_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            audio = None
            conf = None
            # Some prediction dumps contain raw ``Hypothesis(...)`` lines. Extract
            # the ``text='…'`` segment if encountered.
            if line.startswith("Hypothesis("):
                match = re.search(r"text='([^']*)'", line)
                hyp = match.group(1) if match else ""
                ref = ""
            else:
                row = json.loads(line)
                audio = row.get("audio")
                ref = _norm(row.get("ref") or row.get("text") or "")
                hyp = row.get("hyp") or row.get("pred_text") or ""
                conf = row.get("confidence") or row.get("conf")
                if isinstance(hyp, str) and hyp.startswith("Hypothesis("):
                    match = re.search(r"text='([^']*)'", hyp)
                    hyp = match.group(1) if match else hyp
            hyp = _norm(hyp)
            if confidence_threshold is not None and conf is not None and conf < confidence_threshold:
                continue
            row_out = {"audio": audio, "ref": ref, "hyp": hyp}
            if conf is not None:
                row_out["confidence"] = conf
            rows.append(row_out)
            refs.append(ref)
            hyps.append(hyp)

    w = wer(refs, hyps) if refs else 0.0
    ser_flags = [ref.strip() != hyp.strip() for ref, hyp in zip(refs, hyps)]
    ser = float(np.mean(ser_flags)) if ser_flags else 0.0

    sample_wers = [wer([r], [h]) for r, h in zip(refs, hyps)]
    semantic_sims = compute_semantic_similarity(refs, hyps)

    difficulty_scores = []
    for row, swer, sim, sf in zip(rows, sample_wers, semantic_sims, ser_flags):

        difficulty = (
            DIFFICULTY_WER_WEIGHT * swer
            + DIFFICULTY_SEM_WEIGHT * (1.0 - float(sim))
        )

        row.update(
            {
                "wer": swer,
                "semantic": float(sim),
                "difficulty": difficulty,
                "ser": int(sf),
            }
        )
        difficulty_scores.append(difficulty)

    conf_vals = [r["confidence"] for r in rows if r.get("confidence") is not None]
    if conf_vals:
        confidence_stats = {
            "mean": float(np.mean(conf_vals)),
            "median": float(np.median(conf_vals)),
            "p05": float(np.quantile(conf_vals, 0.05)),
            "p95": float(np.quantile(conf_vals, 0.95)),
        }
        with (out_dir / "confidence_stats.json").open("w", encoding="utf-8") as f:
            json.dump(confidence_stats, f, ensure_ascii=False, indent=2)
        print(
            "Confidence statistics:",
            ", ".join(f"{k}={v:.4f}" for k, v in confidence_stats.items()),
        )

    # ------------------------------------------------------------------
    # Pareto front based on ``difficulty`` and ``confidence``
    # We minimise difficulty while maximising confidence.
    # ------------------------------------------------------------------
    pairs = [
        (r["difficulty"], r.get("confidence"), r)
        for r in rows
        if r.get("confidence") is not None
    ]
    pareto_front: List[dict] = []
    if pairs:
        pairs.sort(key=lambda x: x[0])  # sort by increasing difficulty
        best_conf = float("-inf")
        for w, c, r in pairs:
            if c >= best_conf:
                pareto_front.append(r)
                best_conf = c
        dominated = [r for _, _, r in pairs if r not in pareto_front]
    else:
        dominated = []


    out_dir.mkdir(parents=True, exist_ok=True)

    wers = np.array(sample_wers)
    diffs = np.array(difficulty_scores)
    q_low_d = np.quantile(diffs, tail_fraction)
    q_high_d = np.quantile(diffs, 1 - tail_fraction)
    filtered = [r for r in rows if q_low_d <= r["difficulty"] <= q_high_d]

    filtered.sort(key=lambda r: r["difficulty"])
    cut = int(len(filtered) * 0.3)
    easy = filtered[:cut]
    difficult = filtered[cut:]

    def write_jsonl(path: Path, data: List[dict]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for row in data:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_jsonl(out_dir / "easy.jsonl", easy)
    write_jsonl(out_dir / "difficult.jsonl", difficult)
    write_jsonl(out_dir / "pareto_front.jsonl", pareto_front)
    write_jsonl(
        out_dir / "beyond_pareto.jsonl",
        sorted(
            dominated,
            key=lambda r: (r.get("confidence", float("inf")), -r["difficulty"]),
        )
        if dominated
        else [],
    )

    if plt:
        plt.figure()
        plt.hist(wers, bins=50)
        q_low_w = np.quantile(wers, tail_fraction)
        q_high_w = np.quantile(wers, 1 - tail_fraction)
        plt.axvline(q_low_w, color="red", linestyle="dashed")
        plt.axvline(q_high_w, color="red", linestyle="dashed")
        plt.title("WER distribution")
        plt.xlabel("WER")
        plt.ylabel("Count")
        plt.savefig(out_dir / "wer_distribution.png")

        plt.figure()
        plt.hist(diffs, bins=50)
        plt.axvline(q_low_d, color="red", linestyle="dashed")
        plt.axvline(q_high_d, color="red", linestyle="dashed")
        plt.title("Difficulty distribution")
        plt.xlabel("Difficulty")
        plt.ylabel("Count")
        plt.savefig(out_dir / "difficulty_distribution.png")

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

        if pairs:
            plt.figure()
            all_diff = [p[0] for p in pairs]
            all_conf = [p[1] for p in pairs]
            plt.scatter(all_diff, all_conf, alpha=0.5, label="all")
            front_diff = [r["difficulty"] for r in pareto_front]
            front_conf = [r["confidence"] for r in pareto_front]
            plt.plot(front_diff, front_conf, color="red", marker="o", label="Pareto front")
            plt.xlabel("Difficulty")
            plt.ylabel("Confidence")
            plt.title("Difficulty vs Confidence")
            plt.legend()
            plt.savefig(out_dir / "difficulty_vs_confidence_pareto.png")

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
        analyse_dataset(
            pred_file,
            out_dir,
            args.tail_fraction,
            args.confidence_threshold,

        )


if __name__ == "__main__":
    main()
