from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, TYPE_CHECKING

from tqdm import tqdm

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from nemo.collections.asr.models import ASRModel


# Base directory of the script, used for resolving relative paths
BASE_DIR = Path(__file__).resolve().parent

# Hardcoded relative location of the Canary model files
MODEL_PATH = BASE_DIR / ".hf" / "models--nvidia--canary-1b-v2"
# Pin Canary revision for deterministic downloads
CANARY_REVISION = "809ebc8cde9905ef510b7f834cb8e4627220f037"


def resolve_path(p: Path) -> Path:
    """Resolve a path relative to the script directory if it is not absolute."""
    return p if p.is_absolute() else (BASE_DIR / p).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Canary inference on a dataset")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data_wav\\Vikhrmodels___ToneRuDevices_train"),
                        help="Directory containing manifest.jsonl")
    parser.add_argument("--out-dir", type=Path, default=Path("predictions\\Vikhrmodels___ToneRuDevices_train"),
                        help="Directory to save predictions")
    parser.add_argument("--model-id", default="nvidia/canary-1b-v2",
                        help="HuggingFace model identifier")
    parser.add_argument("--source-lang", default="ru",
                        help="Source language for inference")
    parser.add_argument("--target-lang", default="ru",
                        help="Target language for inference")

    ts = parser.add_mutually_exclusive_group()
    ts.add_argument("--timestamps", dest="timestamps", action="store_true",
                    help="Return word/segment timestamps if supported")
    ts.add_argument("--no-timestamps", dest="timestamps", action="store_false")
    parser.set_defaults(timestamps=False)

    parser.add_argument("--batch-size", type=int, default=32,
                        help="Transcription batch size (use 1 for very long files)")

    args = parser.parse_args()
    args.dataset_dir = resolve_path(args.dataset_dir)
    args.out_dir = resolve_path(args.out_dir)
    return args


def load_canary(model_id: str, model_path: Path = MODEL_PATH, revision: str = CANARY_REVISION):
    """Load a Canary model from various sources.

    If ``model_path`` points to a ``.nemo`` file, the model is restored from it.
    If ``model_path`` is a directory, the function first looks for a ``.nemo``
    file inside (useful for a Hugging Face cache) and falls back to using the
    directory as ``cache_dir`` for :func:`ASRModel.from_pretrained`.
    Otherwise the model is downloaded from Hugging Face using ``model_id``.
    """

    from nemo.collections.asr.models import ASRModel

    model_path = Path(model_path)
    if model_path.is_file():
        model = ASRModel.restore_from(str(model_path)).eval()
    elif model_path.is_dir():
        model_name = model_id.split("/")[-1]
        nemo_files = list(model_path.rglob(f"*{model_name}*.nemo")) or list(
            model_path.rglob("*.nemo")
        )
        if nemo_files:
            model = ASRModel.restore_from(str(nemo_files[0])).eval()
        else:
            model = ASRModel.from_pretrained(
                model_name=model_id,
                cache_dir=str(model_path),
                revision=revision,
            ).eval()
    else:
        model = ASRModel.from_pretrained(
            model_name=model_id,
            cache_dir=str(model_path),
            revision=revision,
        ).eval()

    return model


def transcribe_paths(
    model: "ASRModel",
    paths: List[str],
    batch_size: int,
    source_lang: str,
    target_lang: str,
    timestamps: bool,
):
    """Transcribe ``paths`` and optionally return timestamps."""

    results = {}
    batch_size = max(1, int(batch_size))
    import torch, gc

    # Do not force prompt slots for PnC; rely on model defaults

    for i in range(0, len(paths), batch_size):
        batch = paths[i : i + batch_size]
        hyps = model.transcribe(
            batch,
            batch_size=batch_size,
            source_lang=source_lang,
            target_lang=target_lang,
            timestamps=timestamps,
            return_hypotheses=True,
        )
        if isinstance(hyps, tuple):
            hyps = hyps[0]

        for p, h in zip(batch, hyps):
            text = getattr(h, "text", getattr(h, "pred_text", str(h)))

            # Extract timestamps when requested and available
            ts = None
            if timestamps:
                ts = {"word": [], "segment": []}
                # Word-level
                words = getattr(h, "word_timestamps", None) or getattr(h, "words", None) or getattr(h, "word_ts", None)
                if words:
                    for w in words:
                        if isinstance(w, dict):
                            st = w.get("start_time", w.get("start", None))
                            et = w.get("end_time", w.get("end", None))
                            wd = w.get("word", w.get("text", None))
                        else:
                            st = getattr(w, "start_time", getattr(w, "start", None))
                            et = getattr(w, "end_time", getattr(w, "end", None))
                            wd = getattr(w, "word", getattr(w, "text", None))
                        try:
                            if st is not None and et is not None and wd is not None:
                                ts["word"].append({"start": float(st), "end": float(et), "word": str(wd)})
                        except Exception:
                            continue
                # Segment-level
                segs = getattr(h, "segments", None) or getattr(h, "segment_timestamps", None) or getattr(h, "timestamps", None)
                if segs:
                    for s in segs:
                        if isinstance(s, dict):
                            st = s.get("start", s.get("start_time", None))
                            et = s.get("end", s.get("end_time", None))
                            tx = s.get("segment", s.get("text", None))
                        else:
                            st = getattr(s, "start", getattr(s, "start_time", None))
                            et = getattr(s, "end", getattr(s, "end_time", None))
                            tx = getattr(s, "segment", getattr(s, "text", None))
                        try:
                            if st is not None and et is not None and tx is not None:
                                ts["segment"].append({"start": float(st), "end": float(et), "segment": str(tx)})
                        except Exception:
                            continue
                # Drop empty container
                if not ts["word"] and not ts["segment"]:
                    ts = None

            results[p] = {"text": text, **({"timestamp": ts} if ts else {})}
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass
        gc.collect()

    return results


def _estimate_durations(paths):
    import soundfile as sf
    durs = []
    for p in paths:
        try:
            with sf.SoundFile(p) as f:
                durs.append(len(f) / f.samplerate)
        except Exception:
            durs.append(0.0)
    return durs


def main() -> None:
    args = parse_args()
    manifest = args.dataset_dir / "manifest.jsonl"
    items = [json.loads(x) for x in manifest.open("r", encoding="utf-8")]
    uniq_paths = list({it["audio_filepath"] for it in items})
    model = load_canary(args.model_id)
    durs = _estimate_durations(uniq_paths)
    effective_bs = 1 if any(d > 40.0 for d in durs) else args.batch_size
    preds = transcribe_paths(
        model,
        uniq_paths,
        effective_bs,
        args.source_lang,
        args.target_lang,
        args.timestamps,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "predictions.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for it in tqdm(items, desc="write"):
            pred = preds.get(it["audio_filepath"], {})
            hyp = pred.get("text", "")
            row = {
                "audio": it["audio_filepath"],
                "ref": it.get("text", ""),
                "hyp": hyp,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved predictions to {out_path}")


if __name__ == "__main__":
    main()
