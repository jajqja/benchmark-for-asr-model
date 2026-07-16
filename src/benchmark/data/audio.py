"""Nạp audio -> waveform mono float32 ở sample rate mong muốn.

Dùng chung cho mọi adapter để đảm bảo tiền xử lý audio ĐỒNG NHẤT (yếu tố công bằng):
mọi model nhận cùng một waveform, chỉ khác cách decode.
"""
from __future__ import annotations

import os

import numpy as np


def load_audio(path: str, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """Trả về (waveform float32 mono trong [-1,1], target_sr).

    Đọc bằng soundfile; downmix stereo -> mono; resample nếu lệch sample rate.
    """
    import soundfile as sf

    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim == 2:  # (frames, channels) -> mono
        wav = wav.mean(axis=1)
    wav = np.ascontiguousarray(wav, dtype=np.float32)
    if sr != target_sr:
        wav = _resample(wav, sr, target_sr)
    return wav, target_sr


def _resample(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample chất lượng cao. Ưu tiên librosa, fallback tuyến tính (kèm cảnh báo)."""
    try:
        import librosa
        return librosa.resample(wav, orig_sr=orig_sr, target_sr=target_sr).astype(np.float32)
    except Exception:
        # Fallback nội suy tuyến tính — chỉ dùng khi thiếu librosa, chất lượng thấp hơn.
        import warnings
        warnings.warn("librosa không có; dùng resample tuyến tính (chất lượng thấp hơn).")
        duration = wav.shape[0] / orig_sr
        n_target = int(round(duration * target_sr))
        x_old = np.linspace(0.0, duration, num=wav.shape[0], endpoint=False)
        x_new = np.linspace(0.0, duration, num=n_target, endpoint=False)
        return np.interp(x_new, x_old, wav).astype(np.float32)


def duration_sec(wav: np.ndarray, sr: int) -> float:
    return round(len(wav) / sr, 3)


def write_temp_wav(wav: np.ndarray, sr: int) -> str:
    """Ghi waveform ra file wav tạm (PCM16), trả về đường dẫn.

    Dùng cho các framework chỉ nhận audio_path (chunkformer, NeMo). Người gọi
    có trách nhiệm xoá file sau khi dùng.
    """
    import tempfile
    import soundfile as sf

    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(path, wav, sr, subtype="PCM_16")
    return path
