import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import load_dataset, Audio
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

# ----- Настройки скачивания -----
# Укажите здесь ссылку на набор данных Hugging Face (например "nvidia/voice")
HF_DATASET_REPO = ""

# Укажите здесь ваш токен доступа Hugging Face или оставьте пустым, если он не требуется
HF_TOKEN = ""

# Укажите каталог кэша, куда будут скачаны данные Hugging Face
HF_CACHE_DIR = "hf_cache"
# --------------------------------

MIN_DUR = 1.0
MAX_DUR = 35.0

def sha1_name(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def ensure_wav_mono16k(data: np.ndarray, sr: int):
    """Return mono float32 audio resampled to 16 kHz."""
    if getattr(data, "ndim", 1) > 1:
        data = data.mean(axis=1)
    data = np.asarray(data, dtype=np.float32)
    if sr != 16000:
        try:
            import librosa
            data = librosa.resample(y=data, orig_sr=sr, target_sr=16000)
            sr = 16000
        except Exception as exc:  # pragma: no cover - fallback path
            raise RuntimeError("Need resample to 16k but librosa not available") from exc
    return data, 16000

def save_wav(path: Path, data: np.ndarray, sr: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, sr, subtype="PCM_16", format="WAV")


def _process_row(i: int, row: dict, name: str, audio_dir: Path,
                 lang_regex: re.Pattern | None):
    text = None
    for key in ["text", "sentence", "transcript", "transcription", "label", "target"]:
        if key in row and row[key]:
            text = str(row[key]).strip()
            break
    if not text:
        return None

    if lang_regex:
        lang_val = None
        for lkey in ["lang", "language", "source_lang", "locale"]:
            if lkey in row and row[lkey]:
                lang_val = str(row[lkey]).lower()
                break
        if lang_val and not lang_regex.search(lang_val):
            return None

    audio = row.get("audio")
    if not audio:
        return None
    arr = np.asarray(audio["array"])
    sr = int(audio["sampling_rate"])
    try:
        arr, sr = ensure_wav_mono16k(arr, sr)
    except Exception:
        return None
    dur = float(len(arr) / sr)
    if not (MIN_DUR <= dur <= MAX_DUR):
        return None

    wav_path = audio_dir / f"{sha1_name(name + '_' + str(i))}.wav"
    save_wav(wav_path, arr, sr)
    return {"audio_filepath": str(wav_path), "text": text, "duration": dur}

def prepare_hf_dataset_to_wav(repo: str, split: str, out_root: Path,
                              lang_regex: re.Pattern | None, hf_token: str | None,
                              cache_dir: Path | None = None, num_workers: int = 10):
    """Download HF dataset split and store as 16k mono wav + manifest."""
    name = f"{repo.replace('/', '___')}_{split}"
    out_dir = out_root / name
    manifest = out_dir / "manifest.jsonl"
    if manifest.exists():
        print(f"[skip] {name}: already exists")
        return {"name": name, "manifest": str(manifest), "dir": str(out_dir), "kept": 0}

    kwargs = {}
    if hf_token:
        kwargs["token"] = hf_token
    if cache_dir:
        kwargs["cache_dir"] = str(cache_dir)
    try:
        ds = load_dataset(repo, split=split, streaming=False, **kwargs)
    except Exception as exc:  # pragma: no cover - network path
        print(f"[skip] {repo}:{split} -> {exc}")
        return None

    try:
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    except Exception:
        pass

    audio_dir = out_dir / "audio"
    out_dir.mkdir(parents=True, exist_ok=True)

    kept = 0
    with manifest.open("w", encoding="utf-8") as fo, ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = set()
        for i, row in enumerate(tqdm(ds, desc=f"{repo}:{split}")):
            futures.add(ex.submit(_process_row, i, row, name, audio_dir, lang_regex))
            if len(futures) >= num_workers * 5:
                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                for fut in done:
                    item = fut.result()
                    if item:
                        fo.write(json.dumps(item, ensure_ascii=False) + "\n")
                        kept += 1
        done, _ = wait(futures)
        for fut in done:
            item = fut.result()
            if item:
                fo.write(json.dumps(item, ensure_ascii=False) + "\n")
                kept += 1
    print(f"[OK] {name}: {kept} records")
    return {"name": name, "manifest": str(manifest), "dir": str(out_dir), "kept": kept}

def main():
    parser = argparse.ArgumentParser(description="Download HF dataset split to wav + manifest")
    parser.add_argument("repo", nargs="?", default=HF_DATASET_REPO,
                    help="HF dataset repo, e.g. nvidia/voice")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", default="data_wav", help="Output directory")
    parser.add_argument("--hf-token", dest="hf_token",
                    default=HF_TOKEN or os.environ.get("HF_TOKEN"),
                    help="HF access token")
    parser.add_argument("--lang-regex", dest="lang_regex", default="(^|[-_])ru([-_]|$)|russian")
    parser.add_argument("--cache-dir", dest="cache_dir", default=HF_CACHE_DIR,
                        help="HF datasets cache directory")
    parser.add_argument("--workers", type=int, default=10,
                        help="Number of concurrent workers for processing")
    args = parser.parse_args()

    regex = re.compile(args.lang_regex, re.IGNORECASE) if args.lang_regex else None
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    prepare_hf_dataset_to_wav(args.repo, args.split, Path(args.out), regex,
                              args.hf_token, cache_dir, args.workers)

if __name__ == "__main__":
    main()
