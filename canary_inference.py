from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, TYPE_CHECKING

from tqdm import tqdm

# Ensure OmegaConf is available; install on the fly if missing
try:  # pragma: no cover - optional dependency
    from omegaconf import OmegaConf
except Exception:  # pragma: no cover - install if absent
    import subprocess, sys

    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "omegaconf"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        from omegaconf import OmegaConf  # type: ignore
    except Exception:  # pragma: no cover - give up quietly
        OmegaConf = None

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

    pnc = parser.add_mutually_exclusive_group()
    pnc.add_argument("--pnc", dest="pnc", action="store_true",
                     help="Enable punctuation and casing (default)")
    pnc.add_argument("--no-pnc", dest="pnc", action="store_false",
                     help="Disable punctuation and casing")
    parser.set_defaults(pnc=True)

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
                model_name=model_id, cache_dir=str(model_path)
            ).eval()
    else:
        model = ASRModel.from_pretrained(model_name=model_id).eval()

    # Enable token-level confidence estimation when possible
    if OmegaConf is not None:
        try:  # pragma: no cover - optional nemo/omegaconf dependency
            cfg = OmegaConf.create({"name": "entropy", "alpha": 0.3})
            model.change_decoding_strategy(
                preserve_token_confidence=True, confidence_method_cfg=cfg
            )
        except Exception:
            pass

    return model


def _safe_len(tokens) -> int:
    """Return a safe length for ``tokens`` which may be tensors or sequences."""
    if tokens is None:
        return 0
    try:  # pragma: no cover - optional torch dependency
        import torch

        if torch.is_tensor(tokens):
            return tokens.numel()
    except Exception:  # pragma: no cover - torch not installed
        pass
    try:
        return len(tokens)
    except Exception:  # pragma: no cover - objects without length
        return 0


def _mean_logprob(token_logprobs):
    """Compute the mean of log probabilities if possible."""
    return _safe_mean(token_logprobs)


def _safe_mean(values):
    """Return the mean of ``values`` if possible."""
    if values is None:
        return None
    try:  # pragma: no cover - optional torch dependency
        import torch

        if torch.is_tensor(values):
            if values.numel() == 0:
                return None
            return float(values.mean().item())
    except Exception:  # pragma: no cover - torch not installed
        pass
    try:
        if len(values) == 0:
            return None
        return float(sum(float(x) for x in values) / len(values))
    except Exception:  # pragma: no cover - objects without length
        return None


def transcribe_paths(
    model: "ASRModel",
    paths: List[str],
    batch_size: int,
    source_lang: str,
    target_lang: str,
    pnc: bool,
    timestamps: bool,
):
    """Transcribe ``paths`` and compute a simple confidence score.

    Confidence is estimated from per-token probabilities when available.
    If the model does not provide them, log probabilities or the overall
    hypothesis score are used as fallbacks.
    """

    results = {}
    batch_size = max(1, int(batch_size))
    import torch, gc

    try:
        if hasattr(model, "prompt") and hasattr(model.prompt, "update_slots"):
            model.prompt.update_slots({"pnc": "<|pnc|>" if pnc else "<|nopnc|>"})
    except Exception:
        pass

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

            conf = None
            token_conf = getattr(h, "token_confidence", None) or getattr(
                h, "word_confidence", None
            )
            if token_conf:
                conf = _safe_mean(token_conf)
            else:
                score = getattr(h, "score", None)
                length = getattr(h, "length", None)
                if score not in (None, 0.0) and isinstance(length, (int, float)) and length > 0:
                    try:
                        import torch

                        if torch.is_tensor(score):
                            score = float(score.detach().item())
                    except Exception:
                        pass
                    conf = float(score) / float(length)

            results[p] = {"text": text, "confidence": conf}
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass
        gc.collect()

    return results


def main() -> None:
    args = parse_args()
    manifest = args.dataset_dir / "manifest.jsonl"
    items = [json.loads(x) for x in manifest.open("r", encoding="utf-8")]
    uniq_paths = list({it["audio_filepath"] for it in items})
    model = load_canary(args.model_id)
    preds = transcribe_paths(
        model,
        uniq_paths,
        args.batch_size,
        args.source_lang,
        args.target_lang,
        args.pnc,
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
            conf = pred.get("confidence")
            if conf is not None:
                row["confidence"] = conf
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved predictions to {out_path}")


if __name__ == "__main__":
    main()
