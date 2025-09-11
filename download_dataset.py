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

def prepare_hf_dataset_to_wav(repo: str, split: str, out_root: Path,
                              lang_regex: re.Pattern | None, hf_token: str | None):
    """Download HF dataset split and store as 16k mono wav + manifest."""
    kwargs = {}
    if hf_token:
        kwargs["token"] = hf_token
    try:
        ds = load_dataset(repo, split=split, streaming=False, **kwargs)
    except Exception as exc:  # pragma: no cover - network path
        print(f"[skip] {repo}:{split} -> {exc}")
        return None

    try:
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    except Exception:
        pass

    name = f"{repo.replace('/', '___')}_{split}"
    out_dir = out_root / name
    audio_dir = out_dir / "audio"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.jsonl"

    kept = 0
    with manifest.open("w", encoding="utf-8") as fo:
        for i, row in enumerate(tqdm(ds, desc=f"{repo}:{split}")):
            text = None
            for key in ["text", "sentence", "transcript", "transcription", "label", "target"]:
                if key in row and row[key]:
                    text = str(row[key]).strip()
                    break
            if not text:
                continue

            if lang_regex:
                lang_val = None
                for lkey in ["lang", "language", "source_lang", "locale"]:
                    if lkey in row and row[lkey]:
                        lang_val = str(row[lkey]).lower()
                        break
                if lang_val and not lang_regex.search(lang_val):
                    continue

            audio = row.get("audio")
            if not audio:
                continue
            arr = np.asarray(audio["array"])
            sr = int(audio["sampling_rate"])
            try:
                arr, sr = ensure_wav_mono16k(arr, sr)
            except Exception:
                continue
            dur = float(len(arr) / sr)
            if not (MIN_DUR <= dur <= MAX_DUR):
                continue

            wav_path = audio_dir / f"{sha1_name(name + '_' + str(i))}.wav"
            save_wav(wav_path, arr, sr)
            item = {"audio_filepath": str(wav_path), "text": text, "duration": dur}
            fo.write(json.dumps(item, ensure_ascii=False) + "\n")
            kept += 1
    print(f"[OK] {name}: {kept} records")
    return {"name": name, "manifest": str(manifest), "dir": str(out_dir), "kept": kept}

def main():
    parser = argparse.ArgumentParser(description="Download HF dataset split to wav + manifest")
    parser.add_argument("repo", help="HF dataset repo, e.g. nvidia/voice")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", default="data_wav", help="Output directory")
    parser.add_argument("--hf-token", dest="hf_token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--lang-regex", dest="lang_regex", default="(^|[-_])ru([-_]|$)|russian")
    args = parser.parse_args()

    regex = re.compile(args.lang_regex, re.IGNORECASE) if args.lang_regex else None
    prepare_hf_dataset_to_wav(args.repo, args.split, Path(args.out), regex, args.hf_token)

if __name__ == "__main__":
    main()
