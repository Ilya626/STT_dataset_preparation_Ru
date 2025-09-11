import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

import numpy as np
import soundfile as sf
import webrtcvad
from datasets import load_dataset, Audio
from tqdm import tqdm

# Reduce memory usage by decoding audio as float32 instead of default float64
_sf_read = sf.read


def _sf_read_float32(*args, **kwargs):
    kwargs.setdefault("dtype", "float32")
    return _sf_read(*args, **kwargs)


sf.read = _sf_read_float32


# ----- Настройки скачивания -----
# Укажите здесь ссылку на набор данных Hugging Face (например "nvidia/voice")
HF_DATASET_REPO = "bond005/taiga_speech_v2"

# Укажите здесь ваш токен доступа Hugging Face или оставьте пустым, если он не требуется
HF_TOKEN = ""

# Укажите каталог кэша, куда будут скачаны данные Hugging Face
HF_CACHE_DIR = "hf_cache"
# --------------------------------

MIN_DUR = 1.0
MAX_DUR = 35.0
VAD_MODE = 2  # 0-3, higher = more aggressive
PAUSE_SEC = 0.3


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


def _find_split(arr: np.ndarray, sr: int, min_pause: float, mode: int) -> float:
    """Find a split point (in seconds) near the middle using VAD."""
    vad = webrtcvad.Vad(mode)
    frame_ms = 30
    frame_len = int(sr * frame_ms / 1000)
    if frame_len == 0:
        return len(arr) / sr / 2
    pcm = (arr * 32768).astype(np.int16).tobytes()
    num_frames = len(arr) // frame_len
    silence_spans = []
    start = None
    min_frames = int(min_pause * 1000 / frame_ms)
    for i in range(num_frames):
        frame = pcm[i * frame_len * 2:(i + 1) * frame_len * 2]
        speech = vad.is_speech(frame, sr)
        if not speech:
            if start is None:
                start = i
        elif start is not None:
            if i - start >= min_frames:
                silence_spans.append((start, i))
            start = None
    if start is not None and num_frames - start >= min_frames:
        silence_spans.append((start, num_frames))
    if silence_spans:
        mid = (len(arr) / sr) / 2
        def center(span):
            return ((span[0] + span[1]) / 2) * frame_ms / 1000
        split_time = min(((center(s), s) for s in silence_spans),
                         key=lambda x: abs(x[0] - mid))[0]
    else:
        split_time = len(arr) / sr / 2
    return split_time


def split_audio(arr: np.ndarray, sr: int, text: str,
                max_dur: float, pause_sec: float,
                vad_mode: int) -> list[tuple[np.ndarray, str]]:
    """Recursively split long audio using VAD."""
    segments = [(arr, text)]
    result = []
    while segments:
        a, t = segments.pop(0)
        dur = len(a) / sr
        if dur <= max_dur:
            result.append((a, t))
            continue
        split_t = _find_split(a, sr, pause_sec, vad_mode)
        idx = int(split_t * sr)
        if idx <= 0 or idx >= len(a):
            idx = len(a) // 2
        a1, a2 = a[:idx], a[idx:]
        if t:
            words = t.split()
            cut = int(len(words) * (len(a1) / len(a)))
            t1 = " ".join(words[:cut]).strip()
            t2 = " ".join(words[cut:]).strip()
        else:
            t1 = t2 = t
        segments = [(a2, t2), (a1, t1)] + segments
    return result


def _process_row(i: int, row: dict, name: str, audio_dir: Path,
                 lang_regex: re.Pattern | None,
                 min_dur: float, max_dur: float,
                 pause_sec: float, vad_mode: int):
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
    arr = np.asarray(audio["array"], dtype=np.float32)
    sr = int(audio["sampling_rate"])
    try:
        arr, sr = ensure_wav_mono16k(arr, sr)
    except Exception:
        return None

    items = []
    for j, (seg, seg_text) in enumerate(split_audio(arr, sr, text,
                                                    max_dur, pause_sec,
                                                    vad_mode)):
        dur = float(len(seg) / sr)
        if not (min_dur <= dur <= max_dur) or not seg_text:
            continue
        wav_path = audio_dir / f"{sha1_name(name + '_' + str(i) + '_' + str(j))}.wav"
        save_wav(wav_path, seg, sr)
        items.append({"audio_filepath": str(wav_path),
                      "text": seg_text,
                      "duration": dur})
    return items or None


def prepare_hf_dataset_to_wav(repo: str, split: str, out_root: Path,
                              lang_regex: re.Pattern | None, hf_token: str | None,
                              cache_dir: Path | None = None, num_workers: int = 10,
                              min_dur: float = MIN_DUR, max_dur: float = MAX_DUR,
                              pause_sec: float = PAUSE_SEC, vad_mode: int = VAD_MODE):
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
        ds = load_dataset(repo, split=split, streaming=True, **kwargs)
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

    total = None
    if getattr(ds, "info", None) and ds.info.splits and split in ds.info.splits:
        total = ds.info.splits[split].num_examples

    with manifest.open("w", encoding="utf-8") as fo, ThreadPoolExecutor(max_workers=num_workers) as ex:
        pbar = tqdm(total=total, desc=f"{repo}:{split}")
        pending = set()
        idx = 0
        for row in ds:
            pending.add(ex.submit(_process_row, idx, row, name, audio_dir,
                                  lang_regex, min_dur, max_dur,
                                  pause_sec, vad_mode))
            idx += 1
            if len(pending) >= num_workers:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for fut in done:
                    items = fut.result()
                    if items:
                        for item in items:
                            fo.write(json.dumps(item, ensure_ascii=False) + "\n")
                            kept += 1
                pbar.update(len(done))

        if pending:
            done, _ = wait(pending)
            for fut in done:
                items = fut.result()
                if items:
                    for item in items:
                        fo.write(json.dumps(item, ensure_ascii=False) + "\n")
                        kept += 1
            pbar.update(len(done))

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
    parser.add_argument("--min-dur", type=float, default=MIN_DUR,
                        help="Minimum duration of kept audio segments")
    parser.add_argument("--max-dur", type=float, default=MAX_DUR,
                        help="Maximum duration of audio segments")
    parser.add_argument("--vad-mode", type=int, default=VAD_MODE,
                        help="Aggressiveness of webrtcvad (0-3)")
    parser.add_argument("--pause-sec", type=float, default=PAUSE_SEC,
                        help="Minimum pause length for splitting")
    args = parser.parse_args()

    regex = re.compile(args.lang_regex, re.IGNORECASE) if args.lang_regex else None
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    prepare_hf_dataset_to_wav(args.repo, args.split, Path(args.out), regex,
                              args.hf_token, cache_dir, args.workers,
                              args.min_dur, args.max_dur,
                              args.pause_sec, args.vad_mode)


if __name__ == "__main__":
    main()

