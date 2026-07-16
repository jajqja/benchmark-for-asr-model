"""Adapter cho khanhld/chunkformer-ctc-large-vie.

API tham chiếu từ model card (pip install chunkformer):
    from chunkformer import ChunkFormerModel
    model = ChunkFormerModel.from_pretrained("khanhld/chunkformer-ctc-large-vie")
    text = model.endless_decode(audio_path=..., chunk_size=64,
                                left_context_size=128, right_context_size=128,
                                total_batch_duration=14400, return_timestamps=True)

endless_decode nhận audio_path -> ghi waveform ra wav tạm rồi truyền vào.
Import `chunkformer` được đặt trong load() để module import được ngay cả khi chưa
cài framework (tránh vỡ run_inference lúc nạp adapter động).
"""
from __future__ import annotations

import os

import numpy as np

from benchmark.adapters.base import ASRModel
from benchmark.data.audio import write_temp_wav


class ChunkFormerAdapter(ASRModel):
    name = "chunkformer_large"
    sample_rate = 16000

    def load(self) -> None:
        from chunkformer import ChunkFormerModel

        model_id = self.params.get("model_id", "khanhld/chunkformer-ctc-large-vie")
        self.model = ChunkFormerModel.from_pretrained(model_id)
        device = self.params.get("device", "cuda")
        # Cố gắng chuyển sang GPU nếu API hỗ trợ (không phải bản nào cũng có .to()).
        for mover in (getattr(self.model, "to", None), getattr(self.model, "cuda", None)):
            if device.startswith("cuda") and callable(mover):
                try:
                    mover(device) if mover.__name__ == "to" else mover()
                    break
                except Exception:
                    pass
        self._decode_kwargs = dict(
            chunk_size=self.params.get("chunk_size", 64),
            left_context_size=self.params.get("left_context_size", 128),
            right_context_size=self.params.get("right_context_size", 128),
            total_batch_duration=self.params.get("total_batch_duration", 14400),
            return_timestamps=False,
        )

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        path = write_temp_wav(audio, sample_rate)
        try:
            out = self.model.endless_decode(audio_path=path, **self._decode_kwargs)
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
    """endless_decode có thể trả str hoặc list segment {text,...} tuỳ phiên bản."""
    if isinstance(out, str):
        return out.strip()
    if isinstance(out, dict):
        return str(out.get("text", "")).strip()
    if isinstance(out, (list, tuple)):
        parts = [seg.get("text", "") if isinstance(seg, dict) else str(seg) for seg in out]
        return " ".join(p for p in parts if p).strip()
    return str(out).strip()
