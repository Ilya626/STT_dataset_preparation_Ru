import json
from pathlib import Path
from typing import List

from nemo.collections.asr.models import ASRModel
from tqdm import tqdm


# configuration
DATASET_DIR = Path("data_wav")
OUT_DIR = Path("predictions")
MODEL_ID = "nvidia/canary-1b-v2"
NEMO_PATH: str | None = None
SOURCE_LANG = "ru"
TARGET_LANG = "ru"
TASK = "asr"
PNC = False

# Batch settings
BATCH_SIZE = 16


def load_canary(model_id: str, nemo_path: str | None):
    if nemo_path:
        return ASRModel.restore_from(nemo_path).eval()
    return ASRModel.from_pretrained(model_name=model_id).eval()


def transcribe_paths(model: ASRModel, paths: List[str],
                     source_lang: str, target_lang: str, task: str, pnc: bool):
    results = {}
    batch_size = max(1, int(BATCH_SIZE))
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
    manifest = DATASET_DIR / "manifest.jsonl"
    items = [json.loads(x) for x in manifest.open("r", encoding="utf-8")]
    uniq_paths = list({it["audio_filepath"] for it in items})
    model = load_canary(MODEL_ID, NEMO_PATH)
    preds = transcribe_paths(model, uniq_paths,
                             SOURCE_LANG, TARGET_LANG, TASK, PNC)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "predictions.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for it in tqdm(items, desc="write"):
            hyp = preds.get(it["audio_filepath"], "")
            row = {"audio": it["audio_filepath"], "ref": it.get("text", ""), "hyp": hyp}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved predictions to {out_path}")


if __name__ == "__main__":
    main()
