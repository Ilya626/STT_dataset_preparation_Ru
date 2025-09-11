import argparse
import json
from pathlib import Path
from typing import List

from nemo.collections.asr.models import ASRModel
from tqdm import tqdm


def load_canary(model_id: str, nemo_path: str | None):
    if nemo_path:
        return ASRModel.restore_from(nemo_path).eval()
    return ASRModel.from_pretrained(model_name=model_id).eval()


def transcribe_paths(model: ASRModel, paths: List[str], batch_size: int,
                     source_lang: str, target_lang: str, task: str, pnc: bool):
    results = {}
    batch_size = max(1, int(batch_size))
    import torch, gc
    for i in range(0, len(paths), batch_size):
        batch = paths[i:i + batch_size]
        hyps = model.transcribe(batch, batch_size=batch_size,
                                source_lang=source_lang, target_lang=target_lang,
                                task=task, pnc=pnc)
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


def main():
    parser = argparse.ArgumentParser(description="Run Canary inference and save preds vs refs")
    parser.add_argument("manifest", help="Path to manifest.jsonl produced by download_dataset.py")
    parser.add_argument("--out", default="predictions.jsonl")
    parser.add_argument("--model-id", default="nvidia/canary-1b-v2")
    parser.add_argument("--nemo-path", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--source-lang", default="ru")
    parser.add_argument("--target-lang", default="ru")
    parser.add_argument("--task", default="asr")
    parser.add_argument("--pnc", action="store_true", help="Enable punctuation and casing")
    args = parser.parse_args()

    items = [json.loads(x) for x in Path(args.manifest).open("r", encoding="utf-8")]
    uniq_paths = list({it["audio_filepath"] for it in items})
    model = load_canary(args.model_id, args.nemo_path)
    preds = transcribe_paths(model, uniq_paths, args.batch_size,
                             args.source_lang, args.target_lang, args.task, args.pnc)

    with Path(args.out).open("w", encoding="utf-8") as f:
        for it in tqdm(items, desc="write"):
            hyp = preds.get(it["audio_filepath"], "")
            row = {"audio": it["audio_filepath"], "ref": it.get("text", ""), "hyp": hyp}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved predictions to {args.out}")


if __name__ == "__main__":
    main()
