#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import gradio as gr

# Ensure repository root (parent of this file) is on sys.path so that
# `import canary_inference` works when running this script directly.
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Reuse our existing inference helpers
from canary_inference import load_canary, transcribe_paths

# Helper: ensure mono 16 kHz WAV (defined BEFORE main so it's available when UI starts)
def _ensure_mono_16k(path: str) -> str:
    """Load arbitrary audio file, downmix to mono 16k, save to a temp wav, return its path.
    On failure, falls back to original path.
    """
    try:
        import librosa, soundfile as sf, tempfile, numpy as np
        y, sr = librosa.load(path, sr=16000, mono=True)
        if y is None:
            return path
        if isinstance(y, np.ndarray) and y.size == 0:
            return path
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.close()
        sf.write(tmp.name, y, 16000)
        return tmp.name
    except Exception:
        return path


def _clean_text_ru(text: str) -> str:
    """Best-effort cleanup for mojibake-like characters in Russian output.

    - Replace U+2047 (⁇) that sometimes appears instead of dash or 'ё'.
      Heuristics:
        * If between Russian letters (allowing spaces), treat as 'ё'.
        * If at sentence boundary (after .!? or start), treat as long dash '—'.
    """
    try:
        import re

        # Between letters -> likely 'ё'
        def repl_yo(m):
            left = m.group(1)
            right = m.group(2)
            return f"{left}ё{right}"

        text = re.sub(r"([А-Яа-яЁё])\s*\u2047\s*([А-Яа-яЁё])", repl_yo, text)

        # At sentence boundary -> em dash
        text = re.sub(r"(^|[\.!?…])\s*\u2047\s*", r"\1 — ", text)

        return text
    except Exception:
        return text


def _pick_segment(choice: str, paths: list[str]) -> Optional[str]:
    try:
        if not choice or not paths:
            return None
        idx_str = str(choice).split(":", 1)[0].strip()
        idx = int(idx_str) - 1
        if 0 <= idx < len(paths):
            return paths[idx]
        return None
    except Exception:
        return None


def _smooth_hysteresis(probs, thr_on=0.6, thr_off=0.4):
    """превращает вероятности в bool-маску с гистерезисом."""
    out = []
    state = False
    for p in probs:
        if not state and p >= thr_on:
            state = True
        elif state and p <= thr_off:
            state = False
        out.append(state)
    import numpy as np
    return np.array(out, dtype=bool)


def _speech_blocks_16k(
    path: str,
    vad_backend: str = "silero",  # "none" | "silero" | "marblenet"
    min_silence: float = 1.8,
    min_speech: float = 0.8,
    pad: float = 0.25,
    max_block_sec: float = 240.0,
) -> Tuple[list[str], list[Tuple[float, float]]]:
    """Detect large non-speech and return padded speech blocks.

    - Prefer Silero VAD; fallback to MarbleNet (if available) then energy.
    - Do not cut inside speech blocks. For very long blocks, split with small overlap.
    """
    try:
        import librosa, soundfile as sf, numpy as np, tempfile

        y, sr = librosa.load(path, sr=16000, mono=True)
        if y is None or (isinstance(y, np.ndarray) and y.size == 0):
            return [path], [(0.0, 0.0)]

        duration = len(y) / sr
        frame_ms = 30
        frame_len = int(sr * frame_ms / 1000)
        n_frames = max(1, int(np.ceil(len(y) / frame_len)))

        # Compute speech mask per frame (True for speech)
        speech_mask = np.zeros(n_frames, dtype=bool)

        backend = (vad_backend or "").lower().strip()
        used_backend = "none"

        # Try Silero via torch.hub
        if backend in ("silero", "auto"):
            try:
                import torch  # type: ignore

                model, utils = torch.hub.load(
                    'snakers4/silero-vad', 'silero_vad', trust_repo=True
                )
                (get_speech_ts, _, read_audio, *_) = utils
                wav_t = torch.from_numpy(y).float()
                # Silero expects PCM 16k float tensor
                ts = get_speech_ts(
                    wav_t, model,
                    sampling_rate=16000,
                    min_silence_duration_ms=int(min_silence * 1000),
                    min_speech_duration_ms=int(min_speech * 1000),
                )
                # Convert to frame mask
                for seg in ts:
                    a = max(0, int(seg['start'] // frame_len))
                    b = min(n_frames, int(np.ceil(seg['end'] / frame_len)))
                    if b > a:
                        speech_mask[a:b] = True
                used_backend = "silero"
            except Exception:
                pass

        # Try MarbleNet VAD from NeMo if requested and silero not used
        if used_backend == "none" and backend in ("marblenet", "auto"):
            try:
                from nemo.collections.asr.models import EncDecFrameClassificationModel
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
                vad_model = EncDecFrameClassificationModel.from_pretrained(
                    model_name="vad_multilingual_frame_marblenet"
                ).to(device).eval()
                with torch.no_grad():
                    sig = torch.tensor(y, dtype=torch.float32, device=device).unsqueeze(0)
                    probs = (
                        vad_model.get_frame_probs(signal=sig)
                        .squeeze(0)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                ratio = int(round(30 / 20))
                if ratio > 1:
                    probs = np.maximum.reduce([probs[i::ratio] for i in range(ratio)])
                speech_mask = _smooth_hysteresis(
                    probs,
                    thr_on=max(0.0, min(1.0, 0.6)),
                    thr_off=max(0.0, min(1.0, 0.4)),
                )
                n_frames = len(speech_mask)
                used_backend = "marblenet"
            except Exception:
                pass

        # Fallbacks: WebRTC VAD → energy
        if used_backend == "none":
            try:
                import webrtcvad

                vad = webrtcvad.Vad(2)
                pcm16 = (y * 32767.0).astype(np.int16)
                for i in range(n_frames):
                    beg = i * frame_len
                    end = min(len(pcm16), beg + frame_len)
                    frame = pcm16[beg:end].tobytes()
                    if len(frame) < frame_len * 2:
                        frame = frame + b"\x00" * (frame_len * 2 - len(frame))
                    is_speech = vad.is_speech(frame, sr)
                    speech_mask[i] = bool(is_speech)
                used_backend = "webrtcvad"
            except Exception:
                pass
        if used_backend == "none":
            # Energy-based fallback
            hop = frame_len
            rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop)[0]
            thr = np.percentile(rms, 30.0)
            speech_mask = rms > max(thr, 1e-8)
            used_backend = "energy"

        # Merge consecutive frames into speech/non-speech runs
        runs: list[Tuple[int, int, bool]] = []
        i = 0
        while i < n_frames:
            flag = bool(speech_mask[i])
            j = i + 1
            while j < n_frames and bool(speech_mask[j]) == flag:
                j += 1
            runs.append((i, j, flag))
            i = j

        # Keep only long speech blocks and drop short speech islands
        speech_blocks_frames: list[Tuple[int, int]] = []
        min_speech_frames = int(np.ceil(min_speech * 1000 / frame_ms))
        min_silence_frames = int(np.ceil(min_silence * 1000 / frame_ms))
        for a, b, is_speech in runs:
            if is_speech:
                # Check preceding and following silence lengths to avoid micro cuts
                dur_frames = b - a
                if dur_frames >= min_speech_frames:
                    speech_blocks_frames.append((a, b))

        # Apply padding and convert to sample indices
        pad_samp = int(pad * sr)
        blocks_samples: list[Tuple[int, int]] = []
        for a, b in speech_blocks_frames:
            sa = max(0, a * frame_len - pad_samp)
            sb = min(len(y), b * frame_len + pad_samp)
            if sb - sa > int(0.2 * sr):
                blocks_samples.append((sa, sb))

        # If no blocks detected, return full audio
        if not blocks_samples:
            blocks_samples = [(0, len(y))]

        # Enforce max_block_sec with local overlap ~0.4s
        max_len = int(max_block_sec * sr)
        ovlp = int(0.4 * sr)
        final_samples: list[Tuple[int, int]] = []
        for sa, sb in blocks_samples:
            cur = sa
            while cur < sb:
                end = min(sb, cur + max_len)
                if end - cur < int(0.2 * sr):
                    break
                final_samples.append((cur, end))
                if end >= sb:
                    break
                cur = max(cur + max_len - ovlp, cur + 1)

        # Export to temp WAV files
        out_paths: list[str] = []
        times: list[Tuple[float, float]] = []
        for a, b in final_samples:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            tmp.close()
            sf.write(tmp.name, y[a:b], sr)
            out_paths.append(tmp.name)
            times.append((a / sr, b / sr))

        return out_paths, times
    except Exception:
        return [path], [(0.0, 0.0)]


def merge_by_timestamps(seg_paths: list[str], res: dict, dedup_window: float = 0.8):
    """Merge Canary segments across blocks by time with simple dedup on seams."""
    merged = []
    last_end = 0.0
    texts = []
    for p in seg_paths:
        out = res.get(p, {}) or {}
        ts = (out.get("timestamp") or {})
        seg_ts = ts.get("segment") or []
        if seg_ts:
            for seg in seg_ts:
                try:
                    st = float(seg.get("start") if isinstance(seg, dict) else getattr(seg, "start", 0.0))
                    et = float(seg.get("end") if isinstance(seg, dict) else getattr(seg, "end", 0.0))
                    txt = (seg.get("segment") if isinstance(seg, dict) else getattr(seg, "segment", "")) or seg.get("text") if isinstance(seg, dict) else getattr(seg, "text", "")
                except Exception:
                    continue
                if et <= last_end + dedup_window:
                    continue
                merged.append({"start": st, "end": et, "text": txt})
                texts.append(txt)
                last_end = max(last_end, et)
            continue

        # Fallback to word-level if no segment-level present
        words = ts.get("word") or []
        kept_words = []
        for w in words:
            try:
                st = float(w.get("start") if isinstance(w, dict) else getattr(w, "start", 0.0))
                et = float(w.get("end") if isinstance(w, dict) else getattr(w, "end", 0.0))
                wd = (w.get("word") if isinstance(w, dict) else getattr(w, "word", "")) or (w.get("text") if isinstance(w, dict) else getattr(w, "text", ""))
            except Exception:
                continue
            if et <= last_end + dedup_window:
                continue
            kept_words.append((st, et, wd))
        if kept_words:
            st = kept_words[0][0]
            et = kept_words[-1][1]
            txt = " ".join(w for (_, _, w) in kept_words)
            merged.append({"start": st, "end": et, "text": txt})
            texts.append(txt)
            last_end = max(last_end, et)
        else:
            # No timing info; just append text as-is
            txt = out.get("text", "")
            if txt:
                merged.append({"start": last_end, "end": last_end, "text": txt})
                texts.append(txt)
    return " ".join(t for t in texts if t), merged


# Two-pass decoding helpers
def _needs_redo(txt: str, dur: float, lang: str = "ru") -> bool:
    t = (txt or "").strip()
    if dur > 8.0 and len(t) < 10:
        return True
    if "�" in t or "⁇" in t or "??" in t or ("…" in t and len(t) < 5):
        return True
    import re
    if lang == "ru":
        cyr = len(re.findall(r"[А-Яа-яЁё]", t))
        if cyr < 0.6 * max(1, len(t)):
            return True
    if re.search(r"\b(\w+)\b(?:\s+\1\b){2,}", t, flags=re.IGNORECASE):
        return True
    return False


def _set_beam(model, beam_size=6):
    from copy import deepcopy

    dec = getattr(model, "cfg", None)
    if dec and hasattr(dec, "decoding"):
        dc = deepcopy(dec.decoding)
        if hasattr(dc, "strategy"):
            dc.strategy = "beam"
        if hasattr(dc, "beam"):
            dc.beam.beam_size = int(beam_size)
        if hasattr(dc, "beam_size"):
            dc.beam_size = int(beam_size)
        if hasattr(model, "change_decoding_strategy"):
            model.change_decoding_strategy(dc)


def _set_greedy(model):
    from copy import deepcopy

    dec = getattr(model, "cfg", None)
    if dec and hasattr(dec, "decoding"):
        dc = deepcopy(dec.decoding)
        if hasattr(dc, "strategy"):
            dc.strategy = "greedy"
        if hasattr(model, "change_decoding_strategy"):
            model.change_decoding_strategy(dc)


# Global model holder to avoid deepcopy/pickle issues in gr.State
_MODEL = None


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Gradio UI for Canary ASR (.nemo/HF)")
    ap.add_argument("--nemo", default="", help="Path to BASE .nemo (preferred)")
    ap.add_argument("--nemo_alt", default="", help="Path to ALT .nemo (optional)")
    ap.add_argument("--model_id", default="nvidia/canary-1b-v2", help="HF model id (fallback if no .nemo)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--source_lang", default="ru")
    ap.add_argument("--target_lang", default="ru")
    # VAD / long-form controls
    ap.add_argument("--vad_backend", default="silero", choices=["silero", "marblenet", "none"], help="VAD backend for gating")
    ap.add_argument("--min_silence", type=float, default=1.8, help="Long silence threshold (s)")
    ap.add_argument("--min_speech", type=float, default=0.8, help="Minimal speech block length (s)")
    ap.add_argument("--pad", type=float, default=0.25, help="Padding around speech blocks (s)")
    ap.add_argument("--max_block_sec", type=float, default=240.0, help="Hard cap block length (s)")
    ap.add_argument("--dedup_window", type=float, default=0.8, help="Seam dedup window (s)")
    return ap


def _ensure_model(nemo_path: str, model_id: str):
    path = Path(nemo_path) if nemo_path else None
    if path and path.exists():
        return load_canary(model_id=model_id, model_path=path)
    # Try to resolve a .nemo next to HF cache if path points to dir
    return load_canary(model_id=model_id)


def transcribe_file(
    audio_path: Optional[str],
    chat_state: List[Tuple[str, str]],
    source_lang: str,
    target_lang: str,
    beam_size: int,
    vad_backend: str,
    min_silence: float,
    min_speech: float,
    pad: float,
    max_block_sec: float,
    dedup_window: float,
) -> Tuple[
    List[Tuple[str, str]],  # chat
    object,                 # dataset samples update
    list[list[object]],     # df rows
    object,                 # dropdown update (choices+value)
    Optional[str],          # first audio path for player
    list[str],              # seg paths state
    Optional[str],          # clear audio input
    list[list[object]],     # merged df rows
]:
    global _MODEL
    if not audio_path:
        return chat_state, [], [], gr.update(choices=[], value=None), None, [], None
    if _MODEL is None:
        chat_state = chat_state + [("Error", "Model not loaded")]
        return chat_state, [], [], gr.update(choices=[], value=None), None, [], None
    # Ensure mono 16 kHz WAV to avoid MultiCut errors on stereo inputs
    prep_path = _ensure_mono_16k(audio_path)
    # Generate large speech blocks with VAD gating
    seg_paths, seg_times = _speech_blocks_16k(
        prep_path,
        vad_backend=(vad_backend or "silero"),
        min_silence=float(min_silence or 1.8),
        min_speech=float(min_speech or 0.8),
        pad=float(pad or 0.25),
        max_block_sec=float(max_block_sec or 240.0),
    )

    # Decoding: greedy by default; bump beam only if >1
    if int(beam_size or 1) > 1:
        try:
            from copy import deepcopy
            dec = getattr(_MODEL, "cfg", None)
            if dec and hasattr(dec, "decoding"):
                dc = deepcopy(dec.decoding)
                if hasattr(dc, "beam") and hasattr(dc.beam, "beam_size"):
                    dc.beam.beam_size = int(beam_size)
                if hasattr(dc, "beam_size"):
                    dc.beam_size = int(beam_size)
                if hasattr(_MODEL, "change_decoding_strategy"):
                    _MODEL.change_decoding_strategy(dc)
        except Exception:
            pass

    # PnC is always enabled; request timestamps for robust stitching; keep batch_size=1 inside file
    res = transcribe_paths(
        _MODEL,
        seg_paths,
        batch_size=1,
        source_lang=source_lang,
        target_lang=target_lang,
        timestamps=True,
    )
    # Собираем к кандидаты на редекод
    redo = []
    for (p, (st, et)) in zip(seg_paths, seg_times):
        txt = (res.get(p, {}) or {}).get("text", "")
        if _needs_redo(txt, dur=et - st, lang=source_lang):
            redo.append(p)

    if redo:
        try:
            _set_beam(_MODEL, beam_size=6)
            res2 = transcribe_paths(
                _MODEL,
                redo,
                batch_size=1,
                source_lang=source_lang,
                target_lang=target_lang,
                timestamps=True,
            )
            res.update(res2)
        finally:
            _set_greedy(_MODEL)
    # Merge by timestamps to drop seam duplicates
    full, merged_segments = merge_by_timestamps(seg_paths, res, dedup_window=float(dedup_window or 0.8))
    full = _clean_text_ru(full)
    chat_state = chat_state + [(f"Audio: {Path(audio_path).name}", full)]
    # Build dataset rows with audio + timing + transcript
    dataset_rows: list[list[object]] = []
    for idx, (p, (st, et)) in enumerate(zip(seg_paths, seg_times), start=1):
        pred = res.get(p, {})
        # Clean up possible mojibake
        pred_text = _clean_text_ru(pred.get("text", ""))
        dataset_rows.append([
            p,
            f"{st:.2f}-{et:.2f}s",
            pred_text,
        ])
    # Return dataset update so Gradio refreshes the Dataset.samples
    try:
        ds_update = gr.update(samples=dataset_rows)
    except Exception:
        ds_update = dataset_rows

    # Build dataframe rows + dropdown choices, including Canary span per VAD block
    df_rows: list[list[object]] = []
    choices: list[str] = []
    for idx, (p, (st, et)) in enumerate(zip(seg_paths, seg_times), start=1):
        pred = res.get(p, {})
        # Pull Canary span if present
        ts = (pred.get("timestamp") or {})
        seg_ts = ts.get("segment") or []
        if seg_ts:
            cst = float(seg_ts[0].get("start") if isinstance(seg_ts[0], dict) else getattr(seg_ts[0], "start", 0.0))
            cet = float(seg_ts[-1].get("end") if isinstance(seg_ts[-1], dict) else getattr(seg_ts[-1], "end", 0.0))
            cspan = f"{cst:.2f}-{cet:.2f}s"
        else:
            cspan = ""
        df_rows.append([
            idx,
            round(st, 2),
            round(et, 2),
            _clean_text_ru(pred.get("text", "")),
            cspan,
        ])
        choices.append(f"{idx}: {st:.2f}-{et:.2f}s")

    first_audio = seg_paths[0] if seg_paths else None
    dd_update = gr.update(choices=choices, value=(choices[0] if choices else None))

    # Build merged segments DF rows
    merged_df_rows: list[list[object]] = []
    for i, seg in enumerate(merged_segments, start=1):
        merged_df_rows.append([i, round(float(seg.get("start", 0.0)), 2), round(float(seg.get("end", 0.0)), 2), _clean_text_ru(seg.get("text", ""))])

    return chat_state, ds_update, df_rows, dd_update, first_audio, seg_paths, None, merged_df_rows


def reset_inputs() -> Tuple[None, List[Tuple[str, str]], object, list[list[object]], object, None, list[str], list[list[object]]]:
    try:
        empty_ds = gr.update(samples=[])
    except Exception:
        empty_ds = []
    return None, [], empty_ds, [], gr.update(choices=[], value=None), None, [], []


def switch_model(sel: float, base_path: str, alt_path: str, model_id: str):
    global _MODEL
    # sel is 0.0 or 1.0 from slider
    import gc
    try:
        import torch
    except Exception:
        torch = None
    # Unload previous
    _MODEL = None
    if torch and hasattr(torch, "cuda"):
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass
    gc.collect()

    use_alt = int(sel) == 1
    chosen = (alt_path if use_alt else base_path) or ""
    label = "ALT" if use_alt else "BASE"
    try:
        _MODEL = _ensure_model(chosen, model_id)
        status = f"Loaded [{label}] {Path(chosen).name if chosen else model_id}"
    except Exception as e:
        status = f"[ERR] Load failed for {label}: {e}"
        _MODEL = None
    return status


def main():
    args = build_parser().parse_args()
    global _MODEL
    _MODEL = _ensure_model(args.nemo, args.model_id)

    with gr.Blocks(title="Canary ASR (Gradio)") as demo:
        gr.Markdown("""
        # Canary ASR Demo
        Загрузите WAV/MP3 слева и нажмите Transcribe. Справа появится чат с расшифровками.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                audio = gr.Audio(sources=["upload"], type="filepath", label="Audio file")
                gr.Markdown("### Models")
                base_path = gr.Textbox(value=str(Path(args.nemo).resolve()) if args.nemo else "", label="BASE .nemo path", placeholder="/path/to/base.nemo")
                alt_path = gr.Textbox(value=str(Path(args.nemo_alt).resolve()) if args.nemo_alt else "", label="ALT .nemo path", placeholder="/path/to/alt.nemo")
                model_toggle = gr.Slider(0, 1, value=0, step=1, label="Model selector (0=BASE, 1=ALT)")
                status = gr.Markdown(value=f"Loaded [BASE] {Path(args.nemo).name if args.nemo else args.model_id}")
                # PnC is always enabled; keep a disabled checkbox for clarity
                _pnc_display = gr.Checkbox(value=True, label="Punctuation + Casing (PnC)", interactive=False)
                src = gr.Textbox(value=args.source_lang, label="Source lang", interactive=True)
                tgt = gr.Textbox(value=args.target_lang, label="Target lang", interactive=True)
                beam = gr.Slider(minimum=1, maximum=10, value=1, step=1, label="Beam size")

                gr.Markdown("### VAD + Long-form settings")
                vad_backend = gr.Dropdown(choices=["silero", "marblenet", "none"], value=args.vad_backend, label="VAD backend")
                min_sil = gr.Number(value=args.min_silence, label="min_silence (s)")
                min_sp = gr.Number(value=args.min_speech, label="min_speech (s)")
                pad_s = gr.Number(value=args.pad, label="pad (s)")
                max_blk = gr.Number(value=args.max_block_sec, label="max_block_sec (s)")
                dedup = gr.Number(value=args.dedup_window, label="dedup_window (s)")

                trans_btn = gr.Button("Transcribe", variant="primary")
                reset_btn = gr.Button("Reset", variant="secondary")
            with gr.Column(scale=2):
                chat = gr.Chatbot(label="Transcripts", height=500)
                gr.Markdown("### Сегменты и расшифровки")
                segments_ds = gr.Dataset(
                    components=[
                        gr.Audio(type="filepath"),
                        gr.Textbox(label="Время", interactive=False),
                        gr.Textbox(label="Текст", lines=2),
                    ],
                    headers=["Аудио", "Время", "Транскрипт"],
                    label="Нажмите play на нужном сегменте",
                    samples=[],
                    type="values",
                )
                # Player + controls + table for segments
                seg_player = gr.Audio(label="Segment", type="filepath")
                seg_pick = gr.Dropdown(label="Segment", choices=[], value=None)
                seg_df = gr.Dataframe(
                    headers=["idx", "start_s", "end_s", "text", "canary_span"],
                    datatype=["number", "number", "number", "str", "str"],
                    interactive=False,
                    label="Per-VAD-block transcripts (+ Canary span)",
                )

                gr.Markdown("### Финальная склейка по таймстемпам")
                merged_df = gr.Dataframe(
                    headers=["idx", "start_s", "end_s", "text"],
                    datatype=["number", "number", "number", "str"],
                    interactive=False,
                    label="Merged segments (Canary)",
                )

        state = gr.State([])  # list of (user, bot)
        seg_paths_state = gr.State([])

        trans_btn.click(
            fn=lambda a, s, sl, tl, bm, vb, msil, msp, pd, mblk, ddw: transcribe_file(
                a, s, sl, tl, int(bm), vb, float(msil), float(msp), float(pd), float(mblk), float(ddw)
            ),
            inputs=[audio, state, src, tgt, beam, vad_backend, min_sil, min_sp, pad_s, max_blk, dedup],
            outputs=[chat, segments_ds, seg_df, seg_pick, seg_player, seg_paths_state, audio, merged_df],
        )
        reset_btn.click(fn=reset_inputs, inputs=None, outputs=[audio, chat, segments_ds, seg_df, seg_pick, seg_player, seg_paths_state, merged_df])

        seg_pick.change(
            fn=lambda choice, paths: _pick_segment(choice, paths),
            inputs=[seg_pick, seg_paths_state],
            outputs=[seg_player],
        )

        # Switch model only when slider changes (explicit action)
        model_toggle.change(
            fn=lambda sel, bp, ap, mid: switch_model(sel, bp, ap, mid),
            inputs=[model_toggle, base_path, alt_path, gr.State(args.model_id)],
            outputs=[status],
        )

    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
