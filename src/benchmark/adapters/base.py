"""Interface adapter cho mọi model ASR.

Mục tiêu: che giấu sự khác biệt framework (k2, NeMo, chunkformer, custom) sau một
interface duy nhất. run_inference chỉ cần gọi load() -> transcribe() -> teardown(),
không cần biết bên trong là framework nào.

Adapter chỉ chịu trách nhiệm: waveform -> text (RAW, chưa chuẩn hoá). Việc chuẩn hoá
và chấm điểm nằm ở lõi scoring (Pha 1), tách biệt hoàn toàn.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ASRModel(ABC):
    """Lớp cơ sở cho adapter. Kế thừa và cài đặt load() + transcribe()."""

    #: nhãn model (đặt qua config), dùng cho log & đường dẫn output
    name: str = "asr-model"
    #: sample rate model kỳ vọng; audio sẽ được resample về mức này trước khi đưa vào
    sample_rate: int = 16000

    def __init__(self, name: str | None = None, sample_rate: int | None = None, **params):
        if name is not None:
            self.name = name
        if sample_rate is not None:
            self.sample_rate = sample_rate
        self.params = params

    @abstractmethod
    def load(self) -> None:
        """Nạp trọng số/model lên thiết bị. Gọi một lần trước khi transcribe."""

    @abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        """Nhận diện 1 utterance.

        audio: mảng float32 mono, giá trị trong [-1, 1], ở `sample_rate`.
               run_inference đảm bảo sample_rate == self.sample_rate.
        Trả về: text thô (chưa chuẩn hoá).
        """

    def after_utterance(self, index: int) -> None:
        """Hook gọi SAU mỗi utterance (ngoài vùng đo giờ) — index bắt đầu từ 1.

        Dùng cho bảo trì định kỳ không được tính vào infer_time, ví dụ giải phóng
        bộ nhớ tích lũy (onnxruntime arena). Mặc định no-op.
        """
        pass

    def teardown(self) -> None:
        """Giải phóng tài nguyên (tùy chọn)."""
        pass
