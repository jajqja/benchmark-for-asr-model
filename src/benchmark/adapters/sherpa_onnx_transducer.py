"""Adapter dùng chung cho RNNT transducer chạy qua sherpa-onnx.

Bao cả 2 model:
  * gipformer-65M-rnnt  -> mode="offline" (sherpa_onnx.OfflineRecognizer)
  * Zipformer-30M streaming -> mode="online" (sherpa_onnx.OnlineRecognizer)

(Cách dùng của gipformer suy ra từ infer_onnx.py: OfflineRecognizer.from_transducer
với encoder/decoder/joiner/tokens; zipformer streaming dùng OnlineRecognizer.)

Đường dẫn model: nếu param là path local đã tồn tại -> dùng luôn; ngược lại coi là
tên file trong `repo_id` và tải bằng huggingface_hub. Nếu để null, tự dò trong repo
theo tiền tố (encoder*/decoder*/joiner*/tokens.txt).
"""
from __future__ import annotations

import os

import numpy as np

from benchmark.adapters.base import ASRModel

_TAIL_SEC = 0.5  # đệm im lặng cuối để model streaming xả nốt kết quả


class SherpaOnnxTransducerAdapter(ASRModel):
    name = "sherpa-onnx-transducer"
    sample_rate = 16000

    def load(self) -> None:
        import sherpa_onnx

        paths = self._resolve_paths()
        self.mode = self.params.get("mode", "offline")
        common = dict(
            tokens=paths["tokens"],
            encoder=paths["encoder"],
            decoder=paths["decoder"],
            joiner=paths["joiner"],
            num_threads=self.params.get("num_threads", 4),
            sample_rate=self.sample_rate,
            feature_dim=self.params.get("feature_dim", 80),
            decoding_method=self.params.get("decoding_method", "greedy_search"),
            provider=self.params.get("provider", "cuda"),
        )
        if self.mode == "online":
            self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(**common)
        elif self.mode == "offline":
            self.recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(**common)
        else:
            raise ValueError(f"mode không hợp lệ: {self.mode!r} (offline|online)")

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        if self.mode == "offline":
            s = self.recognizer.create_stream()
            s.accept_waveform(sample_rate, audio)
            self.recognizer.decode_streams([s])
            return s.result.text.strip()
        # online: nạp waveform + đệm im lặng, chờ decode hết.
        s = self.recognizer.create_stream()
        s.accept_waveform(sample_rate, audio)
        tail = np.zeros(int(_TAIL_SEC * sample_rate), dtype=np.float32)
        s.accept_waveform(sample_rate, tail)
        s.input_finished()
        while self.recognizer.is_ready(s):
            self.recognizer.decode_stream(s)
        return self.recognizer.get_result(s).strip()

    def teardown(self) -> None:
        self.recognizer = None

    # ------------------------------------------------------------------
    def _resolve_paths(self) -> dict:
        keys = ("encoder", "decoder", "joiner", "tokens")
        repo_id = self.params.get("repo_id")
        given = {k: self.params.get(k) for k in keys}

        # Nếu tất cả đã cho và là path local tồn tại -> dùng thẳng.
        if all(given[k] and os.path.exists(given[k]) for k in keys):
            return given

        if not repo_id:
            missing = [k for k in keys if not (given[k] and os.path.exists(given[k]))]
            raise ValueError(
                f"Thiếu path local cho {missing} và không có repo_id để tải. "
                "Khai báo repo_id + tên file trong config, hoặc trỏ path local."
            )

        from huggingface_hub import hf_hub_download, HfApi

        # Dò tên file còn thiếu theo tiền tố.
        if any(given[k] is None for k in keys):
            files = HfApi().list_repo_files(repo_id)
            prefix = {"encoder": "encoder", "decoder": "decoder",
                      "joiner": "joiner", "tokens": "tokens"}
            for k in keys:
                if given[k] is None:
                    cands = [f for f in files if os.path.basename(f).startswith(prefix[k])
                             and (f.endswith(".onnx") or f.endswith(".txt"))]
                    if len(cands) != 1:
                        raise ValueError(
                            f"Không tự dò được file cho {k!r} trong {repo_id} "
                            f"(ứng viên: {cands}). Khai báo tên file rõ trong config."
                        )
                    given[k] = cands[0]

        return {k: (given[k] if os.path.exists(given[k])
                    else hf_hub_download(repo_id, given[k])) for k in keys}
