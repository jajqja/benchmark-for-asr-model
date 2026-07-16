"""Adapter cho nvidia/nemotron-3.5-asr-streaming-0.6b (framework NeMo).

API tham chiếu từ model card:
    import nemo.collections.asr as nemo_asr
    m = nemo_asr.models.ASRModel.from_pretrained("nvidia/nemotron-3.5-asr-streaming-0.6b")

Chấm ở chế độ OFFLINE: dùng m.transcribe([...]) trên toàn bộ utterance (model
cache-aware streaming vẫn transcribe offline được). Truyền file path (ổn định qua
nhiều phiên bản NeMo hơn là truyền numpy).
"""
from __future__ import annotations

import os

import numpy as np

from benchmark.adapters.base import ASRModel
from benchmark.data.audio import write_temp_wav


class NemotronNeMoAdapter(ASRModel):
    name = "nemotron_06b"
    sample_rate = 16000

    def load(self) -> None:
        import nemo.collections.asr as nemo_asr

        model_id = self.params.get("model_id", "nvidia/nemotron-3.5-asr-streaming-0.6b")
        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_id)
        self.model.eval()
        device = self.params.get("device", "cuda")
        try:
            import torch
            if device.startswith("cuda") and torch.cuda.is_available():
                self.model = self.model.to(device)
        except Exception:
            pass
        self.batch_size = self.params.get("batch_size", 1)
        # Model đa ngôn ngữ: BẮT BUỘC chỉ định ngôn ngữ qua target_lang (vd 'vi-VN'),
        # nếu không NeMo báo "Unknown prompt key: 'None'".
        self.target_lang = self.params.get("target_lang", "vi-VN")

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        path = write_temp_wav(audio, sample_rate)
        try:
            out = self.model.transcribe(
                [path], batch_size=self.batch_size, target_lang=self.target_lang)
        finally:
            os.remove(path)
        return _to_text(out)

    def teardown(self) -> None:
        self.model = None
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


def _to_text(out) -> str:
    """transcribe() trả list; phần tử có thể là str hoặc Hypothesis(.text)."""
    if isinstance(out, (list, tuple)) and out:
        item = out[0]
    else:
        item = out
    if isinstance(item, str):
        return item.strip()
    text = getattr(item, "text", None)
    return (text if isinstance(text, str) else str(item)).strip()
