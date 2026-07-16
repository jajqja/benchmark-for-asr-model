# ASR Benchmark (tiếng Việt)

Benchmark offline (WER/CER/RTF) cho các model ASR tiếng Việt trên dữ liệu riêng.
Thiết kế chi tiết: xem `.claude/DESIGN.md`.

## Models

- hynt/Zipformer-30M-RNNT-Streaming-6000h: https://huggingface.co/hynt/Zipformer-30M-RNNT-Streaming-6000h
- nvidia/nemotron-3.5-asr-streaming-0.6b: https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b
- khanhld/chunkformer-ctc-large-vie: https://huggingface.co/khanhld/chunkformer-ctc-large-vie
- g-group-ai-lab/gipformer-65M-rnnt: https://huggingface.co/g-group-ai-lab/gipformer-65M-rnnt

## Kiến trúc

Tách 2 giai đoạn để tránh xung đột dependency giữa các framework:

1. **Inference** (framework-specific, cần GPU) → `results/hypotheses/<model>/<dataset>.jsonl`
2. **Scoring** (framework-agnostic, không GPU) → `results/metrics/...` → `leaderboard`

```
data/manifests/*.jsonl ──┐
                         ├─► run_inference ─► hypotheses ─► run_scoring ─► build_report ─► leaderboard
configs/models/*.yaml ───┘        (GPU)                        (CPU)          (CPU)
```

## Quy trình

```bash
# 0. Dựng manifest từ audio + transcript (xem configs/datasets/*.yaml)
python scripts/prepare_manifest.py --config configs/datasets/auto_autoele.yaml

# 0b. (Khuyến nghị) Lấy subset ~200 utt giữ tỉ lệ domain — cả 13 variant ~10h thay vì ~27h
python scripts/subset_manifest.py --in data/manifests/auto_autoele.jsonl \
  --out data/manifests/auto_autoele_sub200.jsonl --n 200 --seed 42

# 1. Inference 1 variant (resume-được). Mỗi nhóm model 1 env riêng — xem requirements/<nhóm>.txt
PYTHONPATH=src python -m benchmark.runners.run_inference \
  --model-config configs/models/gipformer_65m_fp32.yaml \
  --manifest data/manifests/auto_autoele.jsonl \
  --out results/hypotheses/gipformer_65m_fp32/auto_autoele.jsonl

# 2. Chấm điểm + leaderboard + biểu đồ đánh đổi cho mọi hypotheses đã có
bash scripts/run_all.sh   # -> results/reports/{leaderboard.md,leaderboard.csv,tradeoff.png}
```

Trên Colab A100: dùng `notebooks/00_setup_colab.ipynb` → `01_run_model.ipynb` (mỗi model) → `02_report.ipynb`.

## Độ đo

WER, CER, WER/CER không dấu, SER, tỉ lệ I/D/S, RTF, throughput, peak VRAM — cắt lát theo `domain`.

## Variant — đánh đổi độ chính xác ↔ tốc độ

Mỗi nhóm model có nhiều variant (mỗi variant = 1 file config = 1 dòng leaderboard riêng),
để vẽ đường cong accuracy ↔ speed:

- **zipformer**: `chunk16` / `chunk32` / `chunk64` — chunk nhỏ = độ trễ thấp, kém chính xác hơn.
- **gipformer**: `int8` / `fp32` — int8 nhanh & nhẹ, fp32 chính xác hơn.
- **chunkformer**: `c16` / `c32` / `c64` — `chunk_size` khi decode; lớn = nhiều ngữ cảnh, chính xác hơn nhưng chậm/tốn VRAM.
- **nemotron**: `la0` / `la13` — lookahead `att_context_size=[56, r]` (2 cực trị; nemotron chậm nên chỉ đo 2 điểm). Bộ hỗ trợ đầy đủ: 0/3/6/13.

→ 10 variant, đủ vẽ đường cong accuracy ↔ speed cả trong-nhóm lẫn giữa các nhóm.

Notebook `01_run_model.ipynb` chọn `FAMILY` rồi tự chạy TẤT CẢ variant của nhóm trong cùng một env.

## Lưu ý theo model

- **nemotron**: model đa ngôn ngữ `EncDecRNNTBPEModelWithPrompt` — `model.transcribe()` KHÔNG
  dùng được (báo `Unknown prompt key: 'None'` vì prompt ngôn ngữ đọc từ dataloader). Adapter đi
  đường cache-aware streaming: `set_inference_prompt(target_lang)` + `conformer_stream_step`.
  Phải đặt `target_lang` (mặc định `vi-VN`; `auto` = tự nhận diện) trong config. Output có thẻ
  ngôn ngữ ở cuối (vd `<vi-VN>`) → adapter tự cắt (`strip_lang_tags: true`).
- **zipformer**: repo chỉ có `bpe.model`, không có `tokens.txt` → sinh trước rồi trỏ `tokens:`
  vào file đó (config trỏ sẵn `models/zipformer_tokens.txt`; cell sinh có trong `01_run_model.ipynb`).
- **gipformer**: repo có cả bản int8 lẫn fp32 → mỗi bản 1 config variant.

## Ghi chú chung

- Chuẩn hoá text (`configs/normalization/vi.yaml`) áp CHUNG cho ref + mọi hyp để so sánh công bằng.
- Acronym/brand (INVT, Siemens...) model đọc phiên âm còn reference viết dạng chữ → tính là lỗi;
  coi là giới hạn chung (ảnh hưởng đều mọi model, ~0.5% token, không đổi thứ hạng).
- Dữ liệu và manifest là RIÊNG TƯ, không commit (xem `.gitignore`).
