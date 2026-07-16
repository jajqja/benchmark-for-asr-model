"""Chấm điểm framework-agnostic: manifest (reference) + hypotheses -> metrics.json.

Không cần GPU, không phụ thuộc jiwer. Chạy lại bao nhiêu lần cũng được để tinh chỉnh
chuẩn hoá mà không phải inference lại.

Ví dụ:
  PYTHONPATH=src python -m benchmark.runners.run_scoring \
    --manifest data/manifests/my_data_test.jsonl \
    --hyp results/hypotheses/zipformer_30m/my_data_test.jsonl \
    --norm-config configs/normalization/vi.yaml \
    --model zipformer_30m \
    --out results/metrics/zipformer_30m/my_data_test.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

# Cho phép chạy trực tiếp lẫn qua -m: đảm bảo `src` nằm trong sys.path.
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from benchmark.data.manifest import load_manifest, load_hypotheses  # noqa: E402
from benchmark.metrics.wer import compute_wer, compute_cer, compute_ser  # noqa: E402
from benchmark.normalize.vi_normalizer import ViNormalizer  # noqa: E402


def _load_yaml(path: str) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _quality_block(pairs) -> dict:
    """Gộp WER + CER + SER + breakdown I/D/S cho một danh sách (ref, hyp)."""
    wer = compute_wer(pairs)
    cer = compute_cer(pairs)
    ser = compute_ser(pairs)
    return {
        "wer": wer["wer"],
        "cer": cer["wer"],  # compute_cer trả cùng khoá 'wer' (áp trên ký tự)
        "ser": ser["ser"],
        "sub_rate": wer["sub_rate"],
        "del_rate": wer["del_rate"],
        "ins_rate": wer["ins_rate"],
        "ref_words": wer["ref_len"],
        "ref_chars": cer["ref_len"],
    }


def score(manifest_path, hyp_path, norm_cfg, empty_hyp="empty"):
    manifest = load_manifest(manifest_path)
    hyps = load_hypotheses(hyp_path)

    normalizer = ViNormalizer.from_config(norm_cfg)
    normalizer_nd = normalizer.with_diacritics_removed()

    matched, missing = [], []
    for row in manifest:
        uid = row["utt_id"]
        h = hyps.get(uid)
        if h is None:
            missing.append(uid)
            if empty_hyp == "skip":
                continue
            hyp_text = ""  # coi như model không nhận ra -> toàn deletion
        else:
            hyp_text = h.get("text", "") or ""
        matched.append({
            "utt_id": uid,
            "domain": row.get("domain", "all"),
            "ref": row["text"],
            "hyp": hyp_text,
            "audio_dur": (h.get("audio_dur") if h else None) or row.get("duration"),
            "infer_time": h.get("infer_time") if h else None,
        })

    def pairs(items, norm):
        return [(norm(x["ref"]), norm(x["hyp"])) for x in items]

    overall = _quality_block(pairs(matched, normalizer))
    overall_nd = _quality_block(pairs(matched, normalizer_nd))
    overall["wer_no_diacritics"] = overall_nd["wer"]
    overall["cer_no_diacritics"] = overall_nd["cer"]

    # Hiệu năng: RTF = tổng thời gian infer / tổng thời lượng audio.
    tot_infer = sum(x["infer_time"] for x in matched if x["infer_time"] is not None)
    tot_audio = sum(x["audio_dur"] for x in matched if x["audio_dur"] is not None)
    have_timing = any(x["infer_time"] is not None for x in matched)
    efficiency = {
        "rtf": (tot_infer / tot_audio) if (have_timing and tot_audio > 0) else None,
        "throughput_x_realtime": (tot_audio / tot_infer) if (have_timing and tot_infer > 0) else None,
        "total_audio_sec": tot_audio,
        "total_infer_sec": tot_infer if have_timing else None,
    }

    # Cắt lát theo domain.
    by_domain = defaultdict(list)
    for x in matched:
        by_domain[x["domain"]].append(x)
    domains = {
        d: _quality_block(pairs(items, normalizer))
        for d, items in sorted(by_domain.items())
    }

    return {
        "num_utterances": len(matched),
        "num_missing_hyp": len(missing),
        "missing_utt_ids": missing[:50],  # cắt bớt để file gọn
        "overall": overall,
        "efficiency": efficiency,
        "by_domain": domains,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="Chấm điểm ASR từ manifest + hypotheses")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--hyp", required=True)
    ap.add_argument("--norm-config", default="configs/normalization/vi.yaml")
    ap.add_argument("--model", default=None, help="nhãn model, ghi vào output")
    ap.add_argument("--out", default=None, help="đường dẫn metrics.json (mặc định: stdout)")
    ap.add_argument("--missing", choices=["empty", "skip"], default="empty",
                    help="hyp thiếu: 'empty'=coi như câm (deletion) | 'skip'=bỏ qua")
    args = ap.parse_args(argv)

    norm_cfg = _load_yaml(args.norm_config)
    result = score(args.manifest, args.hyp, norm_cfg, empty_hyp=args.missing)
    result["model"] = args.model
    result["manifest"] = args.manifest
    result["hyp"] = args.hyp
    result["norm_config"] = norm_cfg

    ov = result["overall"]
    eff = result["efficiency"]
    print(f"[{args.model or 'model'}] utt={result['num_utterances']} "
          f"missing={result['num_missing_hyp']}", file=sys.stderr)
    print(f"  WER={ov['wer']:.4f}  CER={ov['cer']:.4f}  "
          f"WER(no-dấu)={ov['wer_no_diacritics']:.4f}  SER={ov['ser']:.4f}", file=sys.stderr)
    print(f"  I/D/S rate = {ov['ins_rate']:.4f} / {ov['del_rate']:.4f} / {ov['sub_rate']:.4f}",
          file=sys.stderr)
    if eff["rtf"] is not None:
        print(f"  RTF={eff['rtf']:.4f}  ({eff['throughput_x_realtime']:.1f}x realtime)",
              file=sys.stderr)

    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(payload)
        print(f"  -> {args.out}", file=sys.stderr)
    else:
        print(payload)


if __name__ == "__main__":
    main()
