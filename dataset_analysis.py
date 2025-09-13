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
# Optional default location of original dataset (manifests + audio)
DEFAULT_SOURCE_DIR = Path("data_wav")


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
    parser.add_argument(
        "--dedup-by-text",
        dest="dedup_by_text",
        action="store_true",
        help="Deduplicate by normalized reference text (keep max difficulty)",
    )
    parser.add_argument(
        "--no-dedup-by-text",
        dest="dedup_by_text",
        action="store_false",
        help="Do not deduplicate by text",
    )
    parser.set_defaults(dedup_by_text=True)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help=(
            "Optional directory containing original manifest and audio. "
            "Used to attach durations matched by audio filename."
        ),
    )
    args = parser.parse_args()
    args.preds_dir = resolve_path(args.preds_dir)
    args.out_dir = resolve_path(args.out_dir)
    if args.source_dir is not None:
        args.source_dir = resolve_path(args.source_dir)
    return args


def _norm(text: str) -> str:
    """Normalize text for fair comparison.

    The normalization performs the following steps:
    1. Replace ``С‘``/``РЃ`` with ``Рµ``/``Р•`` to avoid treating them as
       different letters.
    2. Convert the text to lower case so WER is case-insensitive.
    3. Remove punctuation, keeping word boundaries intact.
    """

    text = text.replace("С‘", "Рµ").replace("РЃ", "Р•").lower()
    # Replace any punctuation with a space and collapse multiple spaces.
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


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
    semantic_weight: float,
    wer_weight: float,
    dedup_by_text: bool,
    source_dir: Path | None,
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
    # Prepare optional duration lookup by basename
    durations_by_basename: dict[str, float] = {}
    if source_dir is not None:
        try:
            ds_name = pred_file.parent.name
            cand_dirs = []
            # Prefer a subfolder in source_dir that matches the predictions subfolder name
            if (source_dir / ds_name).exists():
                cand_dirs.append(source_dir / ds_name)
            cand_dirs.append(source_dir)
            for _dir in cand_dirs:
                manifest = _dir / "manifest.json"
                if not manifest.exists():
                    manifest = _dir / "manifest.jsonl"
                if not manifest.exists():
                    continue
                with manifest.open("r", encoding="utf-8") as mf:
                    for _ln in mf:
                        _ln = _ln.strip()
                        if not _ln:
                            continue
                        try:
                            _obj = json.loads(_ln)
                        except Exception:
                            continue
                        _ap = _obj.get("audio") or _obj.get("audio_filepath")
                        if not _ap:
                            continue
                        _base = Path(str(_ap)).name
                        _dur = _obj.get("duration")
                        if isinstance(_dur, (int, float)) and _dur > 0:
                            durations_by_basename[_base] = float(_dur)
                            continue
                        # Try compute from file relative to the manifest dir
                        _full = (_dir / _ap) if not Path(str(_ap)).is_absolute() else Path(str(_ap))
                        try:
                            import soundfile as _sf  # type: ignore
                            _info = _sf.info(str(_full))
                            if _info.frames and _info.samplerate:
                                durations_by_basename[_base] = float(_info.frames) / float(_info.samplerate)
                                continue
                        except Exception:
                            pass
                        try:
                            import wave as _wave, contextlib as _ctx  # type: ignore
                            with _ctx.closing(_wave.open(str(_full), "rb")) as _wf:
                                _fr = _wf.getframerate(); _nf = _wf.getnframes()
                                if _fr and _nf:
                                    durations_by_basename[_base] = float(_nf) / float(_fr)
                        except Exception:
                            pass
        except Exception:
            durations_by_basename = {}
    with pred_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            audio = None
            # Some prediction dumps contain raw ``Hypothesis(...)`` lines. Extract
            # the ``text='вЂ¦'`` segment if encountered.
            if line.startswith("Hypothesis("):
                match = re.search(r"text='([^']*)'", line)
                raw_hyp = match.group(1) if match else ""
                raw_ref = ""
            else:
                row = json.loads(line)
                audio = row.get("audio")
                raw_ref = row.get("ref") or row.get("text") or ""
                raw_hyp = row.get("hyp") or row.get("pred_text") or ""
                if isinstance(raw_hyp, str) and raw_hyp.startswith("Hypothesis("):
                    match = re.search(r"text='([^']*)'", raw_hyp)
                    raw_hyp = match.group(1) if match else raw_hyp
            # Normalize only for metrics; keep raw values in output
            ref = _norm(raw_ref)
            hyp = _norm(raw_hyp)
            # Try to attach duration using provided manifest mapping by basename
            duration = None
            try:
                base = Path(str(audio)).name if audio else None
                if base and "durations_by_basename" in locals():
                    duration = durations_by_basename.get(base)
            except Exception:
                duration = None
            row_out = {"audio": audio, "ref": raw_ref, "hyp": raw_hyp}
            if duration is not None:
                row_out["duration"] = float(duration)
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

        difficulty = wer_weight * swer + semantic_weight * (1.0 - float(sim))

        row.update(
            {
                "wer": swer,
                "semantic": float(sim),
                "difficulty": difficulty,
                "ser": int(sf),
            }
        )
        difficulty_scores.append(difficulty)

    # Optional deduplication by normalized ref text (default on)
    if dedup_by_text and rows:
        by_key: dict[str, dict] = {}
        for r in rows:
            key = _norm(r.get("ref", ""))
            cur = by_key.get(key)
            if cur is None:
                by_key[key] = r
            else:
                c1 = float(r.get("difficulty", 0.0))
                c2 = float(cur.get("difficulty", 0.0))
                if c1 > c2:
                    by_key[key] = r
                elif c1 < c2:
                    pass
                else:
                    d1 = r.get("duration")
                    d2 = cur.get("duration")
                    if isinstance(d1, (int, float)) and isinstance(d2, (int, float)):
                        by_key[key] = r if float(d1) > float(d2) else cur
                    elif isinstance(d1, (int, float)) and not isinstance(d2, (int, float)):
                        by_key[key] = r
        rows = list(by_key.values())

    # Recompute aggregate WER/SER after dedup
    refs2 = [_norm(r.get("ref", "")) for r in rows]
    hyps2 = [_norm(r.get("hyp", "")) for r in rows]
    w = wer(refs2, hyps2) if refs2 else 0.0
    ser_flags = [int(r.get("ser", 0)) for r in rows]
    ser = float(np.mean(ser_flags)) if ser_flags else 0.0
    semantic_sims = np.array([float(r.get("semantic", 0.0)) for r in rows])

    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Pareto front based on ``WER`` (minimise) and ``semantic`` similarity (maximise)
    # ------------------------------------------------------------------
    pairs = [(r["wer"], r["semantic"], r) for r in rows]
    pareto_front: List[dict] = []
    if pairs:
        pairs.sort(key=lambda x: x[0])  # sort by increasing WER
        best_sem = float("-inf")
        for w, s, r in pairs:
            if s >= best_sem:
                pareto_front.append(r)
                best_sem = s
        dominated = [r for _, _, r in pairs if r not in pareto_front]
    else:
        dominated = []

    wers = np.array([float(r.get("wer", 0.0)) for r in rows])
    diffs = np.array([float(r.get("difficulty", 0.0)) for r in rows])
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

    # Final post-filters as requested:
    # - Exclude any rows with WER == 0 or semantic >= 1.0 from both easy and difficult
    # - Additionally, for difficult keep only rows with semantic > 0.99
    easy_filtered = [
        r for r in easy
        if not (r.get("wer") == 0 or r.get("semantic", 0) >= 1.0)
    ]
    difficult_filtered = [
        r for r in difficult
        if not (r.get("wer") == 0 or r.get("semantic", 0) >= 1.0)
        and r.get("semantic", 0) > 0.99
    ]

    write_jsonl(out_dir / "easy.jsonl", easy_filtered)
    write_jsonl(out_dir / "difficult.jsonl", difficult_filtered)
    write_jsonl(out_dir / "pareto_front.jsonl", pareto_front)
    write_jsonl(
        out_dir / "beyond_pareto.jsonl",
        sorted(dominated, key=lambda r: (r["wer"], -r["semantic"])) if dominated else [],
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
        all_wers = [p[0] for p in pairs]
        all_sem = [p[1] for p in pairs]
        plt.scatter(all_wers, all_sem, alpha=0.5, label="all")
        if pareto_front:
            front_wers = [r["wer"] for r in pareto_front]
            front_sem = [r["semantic"] for r in pareto_front]
            plt.plot(front_wers, front_sem, color="red", marker="o", label="Pareto front")
        plt.xlabel("WER")
        plt.ylabel("Semantic similarity")
        plt.title("WER vs Semantic similarity")
        plt.legend()
        plt.savefig(out_dir / "wer_vs_semantic_pareto.png")

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
            args.semantic_weight,
            args.wer_weight,
            args.dedup_by_text,
            args.source_dir,
        )


if __name__ == "__main__":
    main()
