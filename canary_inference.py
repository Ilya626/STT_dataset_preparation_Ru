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


def resolve_path(p: Path) -> Path:
    """Resolve a path relative to the script directory if it is not absolute."""
    return p if p.is_absolute() else (BASE_DIR / p).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Canary inference on a dataset")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data_wav\\bond005___podlodka_speech_train"),
                        help="Directory containing manifest.jsonl")
    parser.add_argument("--out-dir", type=Path, default=Path("predictions\\bond005___podlodka_speech_train"),
                        help="Directory to save predictions")
    parser.add_argument("--model-id", default="nvidia/canary-1b-v2",
                        help="HuggingFace model identifier")
    parser.add_argument("--source-lang", default="ru",
                        help="Source language for inference")
    parser.add_argument("--target-lang", default="ru",
                        help="Target language for inference")
    parser.add_argument("--task", default="asr",
                        help="Task type for Canary model")
    parser.add_argument("--pnc", default="true",action="store_true",
                        help="Enable punctuation and casing")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Transcription batch size")
    args = parser.parse_args()
    args.dataset_dir = resolve_path(args.dataset_dir)
    args.out_dir = resolve_path(args.out_dir)
    return args


def load_canary(model_id: str, model_path: Path = MODEL_PATH):
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
        return ASRModel.restore_from(str(model_path)).eval()
    if model_path.is_dir():
        model_name = model_id.split("/")[-1]
        nemo_files = list(model_path.rglob(f"*{model_name}*.nemo")) or list(
            model_path.rglob("*.nemo")
        )
        if nemo_files:
            return ASRModel.restore_from(str(nemo_files[0])).eval()
        return ASRModel.from_pretrained(
            model_name=model_id, cache_dir=str(model_path)
        ).eval()

    return ASRModel.from_pretrained(model_name=model_id).eval()


def transcribe_paths(model: ASRModel, paths: List[str], batch_size: int,
                     source_lang: str, target_lang: str, task: str, pnc: bool):
    results = {}
    batch_size = max(1, int(batch_size))
    import torch, gc

    for i in range(0, len(paths), batch_size):
        batch = paths[i:i + batch_size]
        hyps = model.transcribe(
            batch,
            batch_size=batch_size,
            source_lang=source_lang,
            target_lang=target_lang,
            task=task,
            pnc=pnc,
        )
        for p, h in zip(batch, hyps):
            if isinstance(h, str):
                results[p] = h
            elif isinstance(h, dict):
                results[p] = h.get("text") or h.get("pred_text") or str(h)
            else:
                results[p] = str(h)
        try:
            torch.cuda.empty_cache(); torch.cuda.ipc_collect()
        except Exception:  # pragma: no cover - CPU path
            pass
        gc.collect()
    return results


def main() -> None:
    args = parse_args()
    manifest = args.dataset_dir / "manifest.jsonl"
    items = [json.loads(x) for x in manifest.open("r", encoding="utf-8")]
    uniq_paths = list({it["audio_filepath"] for it in items})
    model = load_canary(args.model_id)
    preds = transcribe_paths(model, uniq_paths, args.batch_size,
                             args.source_lang, args.target_lang,
                             args.task, args.pnc)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "predictions.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for it in tqdm(items, desc="write"):
            hyp = preds.get(it["audio_filepath"], "")
            row = {"audio": it["audio_filepath"], "ref": it.get("text", ""), "hyp": hyp}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved predictions to {out_path}")


if __name__ == "__main__":
    main()
