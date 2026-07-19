"""Pha 2/3 — chạy inference 1 model trên 1 dataset -> hypotheses.jsonl.

Đặc điểm quan trọng cho Colab:
  * RESUME-ĐƯỢC: bỏ qua utt_id đã có trong file output -> session bị ngắt vẫn chạy tiếp.
  * WARMUP: chạy vài utterance đầu KHÔNG tính giờ (loại chi phí khởi tạo CUDA/lazy-init).
  * Đo infer_time từng utt (đã loại thời gian load model) + peak VRAM.

Adapter được nạp ĐỘNG từ config model (khoá `adapter: module.path.ClassName`), nên
runner này không import bất kỳ framework ASR nào — tránh xung đột dependency.

Ví dụ:
  PYTHONPATH=src python -m benchmark.runners.run_inference \
    --model-config configs/models/chunkformer_large.yaml \
    --manifest data/manifests/my_data_test.jsonl \
    --out results/hypotheses/chunkformer_large/my_data_test.jsonl \
    --warmup 3
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from benchmark.data.manifest import load_manifest  # noqa: E402
from benchmark.data.audio import load_audio, duration_sec  # noqa: E402


def _load_yaml(path: str) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _coerce(v: str):
    """Ép kiểu chuỗi CLI: int/float/bool/None -> đúng kiểu, còn lại giữ str."""
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none"):
        return None
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def _build_adapter(cfg: dict):
    """Nạp động adapter từ 'adapter: pkg.mod.ClassName' + params trong config."""
    dotted = cfg["adapter"]
    module_path, cls_name = dotted.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_path), cls_name)
    return cls(
        name=cfg.get("name", "asr-model"),
        sample_rate=cfg.get("sample_rate", 16000),
        **(cfg.get("params") or {}),
    )


def _done_utt_ids(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    done.add(json.loads(line)["utt_id"])
                except Exception:
                    pass
    return done


def _gpu_reset():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
    except Exception:
        pass


def _gpu_peak_mb():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            return round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1)
    except Exception:
        pass
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description="Chạy inference ASR -> hypotheses.jsonl")
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmup", type=int, default=3, help="số utt chạy trước, không tính giờ")
    ap.add_argument("--limit", type=int, default=None, help="giới hạn số utt (debug/smoke-test)")
    ap.add_argument("--no-resume", action="store_true", help="bỏ qua resume, ghi đè từ đầu")
    ap.add_argument("--set", action="append", default=[], metavar="key=value",
                    help="ghi đè params trong config, vd --set provider=cpu (lặp nhiều lần)")
    args = ap.parse_args(argv)

    cfg = _load_yaml(args.model_config)
    for kv in args.set:
        k, _, v = kv.partition("=")
        cfg.setdefault("params", {})[k.strip()] = _coerce(v.strip())
    manifest = load_manifest(args.manifest)
    if args.limit:
        manifest = manifest[: args.limit]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if args.no_resume and os.path.exists(args.out):
        os.remove(args.out)
    done = _done_utt_ids(args.out)
    todo = [r for r in manifest if r["utt_id"] not in done]
    print(f"[{cfg.get('name')}] tổng={len(manifest)} đã xong={len(done)} "
          f"cần chạy={len(todo)}", file=sys.stderr)
    if not todo:
        print("Không còn gì để chạy.", file=sys.stderr)
        return

    model = _build_adapter(cfg)
    t0 = time.time()
    model.load()
    print(f"  load model: {time.time() - t0:.1f}s", file=sys.stderr)
    target_sr = model.sample_rate

    # Warmup: chạy vài utt đầu để loại chi phí lazy-init khỏi phép đo RTF.
    for r in todo[: args.warmup]:
        try:
            wav, sr = load_audio(r["audio"], target_sr)
            model.transcribe(wav, sr)
        except Exception as e:
            print(f"  warmup lỗi {r['utt_id']}: {e}", file=sys.stderr)

    _gpu_reset()
    n_ok = n_err = 0
    # mở 'a' để resume ghi tiếp; flush từng dòng để mất session không mất dữ liệu.
    with open(args.out, "a", encoding="utf-8") as fout:
        for i, r in enumerate(todo, 1):
            uid = r["utt_id"]
            try:
                wav, sr = load_audio(r["audio"], target_sr)
                dur = duration_sec(wav, sr)
                t = time.time()
                text = model.transcribe(wav, sr)
                infer = time.time() - t
                fout.write(json.dumps(
                    {"utt_id": uid, "text": text, "audio_dur": dur,
                     "infer_time": round(infer, 4)}, ensure_ascii=False) + "\n")
                fout.flush()
                n_ok += 1
            except Exception as e:
                n_err += 1
                print(f"  [{i}/{len(todo)}] LỖI {uid}: {e}", file=sys.stderr)
            # bảo trì định kỳ (vd giải phóng RAM) — NGOÀI vùng đo giờ ở trên
            model.after_utterance(i)
            if i % 50 == 0:
                print(f"  {i}/{len(todo)} (ok={n_ok} err={n_err})", file=sys.stderr)

    model.teardown()
    peak = _gpu_peak_mb()
    print(f"Xong: ok={n_ok} err={n_err} -> {args.out}", file=sys.stderr)
    if peak is not None:
        print(f"  peak VRAM: {peak} MB", file=sys.stderr)
        # ghi sidecar để build_report gộp thông tin hiệu năng phần cứng
        with open(args.out + ".meta.json", "w", encoding="utf-8") as f:
            json.dump({"model": cfg.get("name"), "peak_vram_mb": peak,
                       "num_ok": n_ok, "num_err": n_err}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
