"""Adapter cho nvidia/nemotron-3.5-asr-streaming-0.6b (NeMo, cache-aware streaming).

QUAN TRỌNG: `model.transcribe()` KHÔNG dùng được — đây là EncDecRNNTBPEModelWithPrompt,
prompt ngôn ngữ được đọc từ dataloader nên transcribe() luôn báo
"Unknown prompt key: 'None'". `set_inference_prompt()` chỉ tác động đường streaming.

=> Đi đúng đường cache-aware streaming: set_inference_prompt(target_lang) +
conformer_stream_step() theo
examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py.

Benchmark offline = nạp TOÀN BỘ waveform vào CacheAwareStreamingAudioBuffer rồi chạy
hết các chunk, lấy transcription khi buffer rỗng (keep_all_outputs=True).

CHƯA chạy thử (cần GPU + NeMo) — smoke-test `run_inference --limit 3` trước khi chạy full.
"""
from __future__ import annotations

import os

import numpy as np

from benchmark.adapters.base import ASRModel
from benchmark.data.audio import write_temp_wav


def _calc_drop_extra_pre_encoded(model, step_num, pad_and_drop_preencoded):
    # Bước đầu không cần bỏ frame sau downsampling (trừ khi bật pad_and_drop).
    if step_num == 0 and not pad_and_drop_preencoded:
        return 0
    return model.encoder.streaming_cfg.drop_extra_pre_encoded


def _text_of(item) -> str:
    t = getattr(item, "text", None)
    return (t if isinstance(t, str) else str(item)).strip()


class NemotronStreamingAdapter(ASRModel):
    name = "nemotron_06b"
    sample_rate = 16000

    def load(self) -> None:
        import nemo.collections.asr as nemo_asr
        import torch

        model_id = self.params.get("model_id", "nvidia/nemotron-3.5-asr-streaming-0.6b")
        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_id)
        self.model.eval()
        self._torch = torch

        device = self.params.get("device", "cuda")
        if device.startswith("cuda") and torch.cuda.is_available():
            self.model = self.model.to(device)

        # BẮT BUỘC: chốt ngôn ngữ cho đường streaming.
        self.model.set_inference_prompt(target_lang=self.params.get("target_lang", "vi-VN"))

        # Cửa sổ ngữ cảnh (lookahead). Để None => dùng mặc định của model.
        att = self.params.get("att_context_size")
        if att is not None:
            self.model.encoder.set_default_att_context_size(att_context_size=list(att))

        self.online_normalization = self.params.get("online_normalization", False)
        self.pad_and_drop_preencoded = self.params.get("pad_and_drop_preencoded", False)
        self._compute_dtype = next(self.model.parameters()).dtype

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        from nemo.collections.asr.parts.utils.streaming_utils import (
            CacheAwareStreamingAudioBuffer,
        )

        torch = self._torch
        path = write_temp_wav(audio, sample_rate)
        try:
            buf = CacheAwareStreamingAudioBuffer(
                model=self.model,
                online_normalization=self.online_normalization,
                pad_and_drop_preencoded=self.pad_and_drop_preencoded,
            )
            buf.append_audio_file(path, stream_id=-1)

            cache_ch, cache_t, cache_ch_len = self.model.encoder.get_initial_cache_state(
                batch_size=1)
            prev_hyp = pred_out_stream = None
            transcribed = None
            for step_num, (chunk_audio, chunk_len) in enumerate(buf):
                with torch.inference_mode():
                    chunk_audio = chunk_audio.to(self._compute_dtype)
                    (pred_out_stream, transcribed, cache_ch, cache_t, cache_ch_len,
                     prev_hyp) = self.model.conformer_stream_step(
                        processed_signal=chunk_audio,
                        processed_signal_length=chunk_len,
                        cache_last_channel=cache_ch,
                        cache_last_time=cache_t,
                        cache_last_channel_len=cache_ch_len,
                        keep_all_outputs=buf.is_buffer_empty(),
                        previous_hypotheses=prev_hyp,
                        previous_pred_out=pred_out_stream,
                        drop_extra_pre_encoded=_calc_drop_extra_pre_encoded(
                            self.model, step_num, self.pad_and_drop_preencoded),
                        return_transcription=True,
                    )
        finally:
            os.remove(path)
        return _text_of(transcribed[0]) if transcribed else ""

    def teardown(self) -> None:
        self.model = None
        try:
            self._torch.cuda.empty_cache()
        except Exception:
            pass
