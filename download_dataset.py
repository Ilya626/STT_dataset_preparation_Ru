#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import webrtcvad
from datasets import load_dataset, Audio
from huggingface_hub import snapshot_download
from tqdm import tqdm

# ---------- Пользовательские настройки по умолчанию ----------
HF_DATASET_REPO = ""          # HF dataset repo
HF_TOKEN = os.environ.get("HF_TOKEN", "")              # или пропишите строкой
HF_CACHE_DIR = "hf_cache"                              # корень кэша/локальной копии
MIN_DUR = 0.3
MAX_DUR = 35.0
VAD_MODE = 2   # 0..3 (выше = агрессивнее)
PAUSE_SEC = 0.3
# -------------------------------------------------------------

# soundfile -> float32 (меньше память)
_sf_read = sf.read
def _sf_read_float32(*args, **kwargs):
    kwargs.setdefault("dtype", "float32")
    return _sf_read(*args, **kwargs)
sf.read = _sf_read_float32


def sha1_name(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def ensure_wav_mono16k(data: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
    """Return mono float32 audio resampled to 16 kHz."""
    if getattr(data, "ndim", 1) > 1:
        data = data.mean(axis=1)
    data = np.asarray(data, dtype=np.float32)
    if sr != 16000:
        try:
            import librosa
            data = librosa.resample(y=data, orig_sr=sr, target_sr=16000)
            sr = 16000
        except Exception as exc:
            raise RuntimeError("Need resample to 16k but librosa not available") from exc
    return data, 16000


def save_wav(path: Path, data: np.ndarray, sr: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    # резюмирование: если уже писали — не перезаписываем
    if path.exists():
        return
    sf.write(str(path), data, sr, subtype="PCM_16", format="WAV")


def _find_split(arr: np.ndarray, sr: int, min_pause: float, mode: int) -> float:
    """Найти точку разреза (сек) около середины с помощью VAD."""
    vad = webrtcvad.Vad(mode)
    frame_ms = 30
    frame_len = int(sr * frame_ms / 1000)
    if frame_len == 0:
        return len(arr) / sr / 2
    pcm = (arr * 32768).astype(np.int16).tobytes()
    num_frames = len(arr) // frame_len
    silence_spans = []
    start = None
    min_frames = max(1, int(min_pause * 1000 / frame_ms))
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
                max_dur: float, pause_sec: float, vad_mode: int) -> List[Tuple[np.ndarray, str]]:
    """Рекурсивно режем длинное аудио по паузам."""
    segments = [(arr, text)]
    result: List[Tuple[np.ndarray, str]] = []
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


def _process_row(i: int, row: Dict, name: str, audio_dir: Path,
                 lang_regex: Optional[re.Pattern],
                 min_dur: float, max_dur: float,
                 pause_sec: float, vad_mode: int):
    # достаём текст
    text = None
    for key in ["text", "sentence", "transcript", "transcription", "label", "target"]:
        if key in row and row[key]:
            text = str(row[key]).strip()
            break
    if not text:
        return None

    # фильтр языка, если задан
    if lang_regex:
        lang_val = None
        for lkey in ["lang", "language", "source_lang", "locale"]:
            if lkey in row and row[lkey]:
                lang_val = str(row[lkey]).lower()
                break
        if lang_val and not lang_regex.search(lang_val):
            return None

    # аудио
    audio = row.get("audio")
    if not audio:
        return None

    # допускаем два формата: dict {'array', 'sampling_rate'} ИЛИ уже декодированную Audio
    if isinstance(audio, dict) and "array" in audio and "sampling_rate" in audio:
        arr = np.asarray(audio["array"], dtype=np.float32)
        sr = int(audio["sampling_rate"])
    else:
        # на всякий случай: если колонка хранит путь
        path = None
        if isinstance(audio, dict) and "path" in audio:
            path = audio["path"]
        elif isinstance(audio, str):
            path = audio
        if not path:
            return None
        arr, sr = sf.read(path)

    try:
        arr, sr = ensure_wav_mono16k(arr, sr)
    except Exception:
        return None

    items = []
    for j, (seg, seg_text) in enumerate(split_audio(arr, sr, text, max_dur, pause_sec, vad_mode)):
        dur = float(len(seg) / sr)
        if not (min_dur <= dur <= max_dur) or not seg_text:
            continue
        wav_path = audio_dir / f"{sha1_name(name + '_' + str(i) + '_' + str(j))}.wav"
        save_wav(wav_path, seg, sr)
        items.append({"audio_filepath": str(wav_path),
                      "text": seg_text,
                      "duration": dur})
    return items or None


def _find_parquet_for_splits(local_root: Path, split_expr: str) -> Dict[str, List[str]]:
    """Подбираем parquet-файлы под запрошенные сплиты (поддержка 'train+validation+test')."""
    all_parquets = [p for p in local_root.rglob("*.parquet")]
    if not all_parquets:
        raise FileNotFoundError(f"No parquet files found under: {local_root}")

    wanted: Dict[str, List[str]] = {}
    parts = [s.strip().lower() for s in split_expr.split("+")]
    for s in parts:
        files = [str(p) for p in all_parquets if s in p.as_posix().lower()]
        # если по имени не нашли, берём всё и пусть будет один общий сплит
        if not files and len(parts) == 1:
            files = [str(p) for p in all_parquets]
        if not files:
            raise FileNotFoundError(f"No parquet matched split='{s}' under: {local_root}")
        wanted[s] = sorted(files)
    return wanted


def _load_local_parquet_as_dataset(files: List[str]):
    """Грузим локальные parquet-файлы в Dataset (без streaming)."""
    ds_dict = load_dataset("parquet", data_files={"_": files})
    ds = ds_dict["_"]
    # пытаемся привести 'audio' к типу Audio (автодекод дорожек по path)
    if "audio" in ds.column_names:
        try:
            ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        except Exception:
            pass
    return ds


def _process_split(ds, repo: str, split_name: str, out_root: Path,
                   lang_regex: Optional[re.Pattern],
                   min_dur: float, max_dur: float,
                   pause_sec: float, vad_mode: int,
                   num_workers: int, prefetch: int) -> Dict:
    name = f"{repo.replace('/', '___')}_{split_name}"
    out_dir = out_root / name
    manifest = out_dir / "manifest.jsonl"
    audio_dir = out_dir / "audio"
    out_dir.mkdir(parents=True, exist_ok=True)

    kept = 0
    total = len(ds)

    max_pending = max(num_workers, num_workers * prefetch)
    with open(manifest, "a", encoding="utf-8") as fo, ProcessPoolExecutor(max_workers=num_workers) as ex:
        pbar = tqdm(total=total, desc=f"{repo}:{split_name}")
        pending = set()
        idx = 0
        for row in ds:
            pending.add(ex.submit(_process_row, idx, row, name, audio_dir,
                                  lang_regex, min_dur, max_dur,
                                  pause_sec, vad_mode))
            idx += 1
            if len(pending) >= max_pending:
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


def prepare_hf_dataset_to_wav_local(repo: str, split_expr: str, out_root: Path,
                                    lang_regex: Optional[re.Pattern],
                                    hf_token: Optional[str],
                                    cache_dir: Optional[Path],
                                    num_workers: int = 10, prefetch: int = 2,
                                    min_dur: float = MIN_DUR, max_dur: float = MAX_DUR,
                                    pause_sec: float = PAUSE_SEC, vad_mode: int = VAD_MODE):

    # 1) Полная локальная копия датасета (со всеми файлами)
    #    Примечание: отключаем symlink на Windows.
    local_root = snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        token=hf_token or True,                   # возьмёт токен из env/кэша, если не передали
        local_dir=str(cache_dir or HF_CACHE_DIR),
        local_dir_use_symlinks=False,
        max_workers=8,                            # параллельные загрузки
    )

    local_root = Path(local_root)

    # 2) Ищем parquet-шарды под нужные сплиты и грузим их локально
    splits_to_files = _find_parquet_for_splits(local_root, split_expr)

    results = []
    for s, files in splits_to_files.items():
        ds = _load_local_parquet_as_dataset(files)
        res = _process_split(ds, repo, s, out_root,
                             lang_regex, min_dur, max_dur,
                             pause_sec, vad_mode,
                             num_workers, prefetch)
        results.append(res)

    return results


def main():
    parser = argparse.ArgumentParser(description="Download HF dataset locally then convert to 16k wav + manifest")
    parser.add_argument("repo", nargs="?", default=HF_DATASET_REPO, help="HF dataset repo, e.g. nvidia/voice")
    parser.add_argument("--split", default="train", help="Split or multiple like 'train+validation+test'")
    parser.add_argument("--out", default="data_wav", help="Output directory")
    parser.add_argument("--hf-token", dest="hf_token", default=HF_TOKEN, help="HF access token")
    parser.add_argument("--lang-regex", dest="lang_regex", default="(^|[-_])ru([-_]|$)|russian")
    parser.add_argument("--cache-dir", dest="cache_dir", default=HF_CACHE_DIR, help="HF cache / local snapshot dir")
    parser.add_argument("--workers", type=int, default=10, help="Number of worker processes")
    parser.add_argument("--prefetch", type=int, default=2, help="Prefetch factor per worker")
    parser.add_argument("--min-dur", type=float, default=MIN_DUR)
    parser.add_argument("--max-dur", type=float, default=MAX_DUR)
    parser.add_argument("--vad-mode", type=int, default=VAD_MODE)
    parser.add_argument("--pause-sec", type=float, default=PAUSE_SEC)
    args = parser.parse_args()

    regex = re.compile(args.lang_regex, re.IGNORECASE) if args.lang_regex else None
    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    prepare_hf_dataset_to_wav_local(
        repo=args.repo,
        split_expr=args.split,
        out_root=Path(args.out),
        lang_regex=regex,
        hf_token=args.hf_token,
        cache_dir=cache_dir,
        num_workers=args.workers,
        prefetch=args.prefetch,
        min_dur=args.min_dur,
        max_dur=args.max_dur,
        pause_sec=args.pause_sec,
        vad_mode=args.vad_mode,
    )


if __name__ == "__main__":
    main()
