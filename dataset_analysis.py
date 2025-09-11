import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from jiwer import wer
from sentence_transformers import SentenceTransformer

# configuration
PREDICTIONS_PATH = Path("preds.jsonl")
OUT_DIR = Path("analysis_output")
TAIL_FRACTION = 0.05


def compute_semantic_similarity(refs: List[str], hyps: List[str]) -> np.ndarray:
    """Return cosine similarity between reference and hypothesis texts."""

    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    emb_ref = model.encode(refs)
    emb_hyp = model.encode(hyps)
    # cosine similarity
    sims = (
        (emb_ref * emb_hyp).sum(axis=1)
        / (np.linalg.norm(emb_ref, axis=1) * np.linalg.norm(emb_hyp, axis=1))
    )
    return sims


def main() -> None:
    rows = []
    refs: List[str] = []
    hyps: List[str] = []
    with PREDICTIONS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            ref = row.get("ref") or row.get("text") or ""
            hyp = row.get("hyp") or row.get("pred_text") or ""
            rows.append({"audio": row.get("audio"), "ref": ref, "hyp": hyp})
            refs.append(ref)
            hyps.append(hyp)

    # Global metrics
    w = wer(refs, hyps) if refs else 0.0
    ser_flags = [ref.strip() != hyp.strip() for ref, hyp in zip(refs, hyps)]
    ser = float(np.mean(ser_flags)) if ser_flags else 0.0

    # Per-sample metrics
    sample_wers = [wer([r], [h]) for r, h in zip(refs, hyps)]
    semantic_sims = compute_semantic_similarity(refs, hyps)

    for row, swer, sim, sf in zip(rows, sample_wers, semantic_sims, ser_flags):
        row.update({"wer": swer, "semantic": float(sim), "ser": int(sf)})

    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Quantile analysis and split
    wers = np.array(sample_wers)
    q_low = np.quantile(wers, TAIL_FRACTION)
    q_high = np.quantile(wers, 1 - TAIL_FRACTION)
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

    # Plots
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


if __name__ == "__main__":
    main()
