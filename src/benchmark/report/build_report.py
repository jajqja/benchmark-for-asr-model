"""Pha 4 — gom mọi metrics.json thành leaderboard (markdown + csv).

Đọc toàn bộ file JSON do run_scoring sinh ra (mặc định quét results/metrics/**.json),
xếp hạng theo WER tăng dần, xuất bảng tổng + (tuỳ chọn) bảng theo domain.

Ví dụ:
  PYTHONPATH=src python -m benchmark.report.build_report \
    --metrics-dir results/metrics \
    --out-md results/reports/leaderboard.md \
    --out-csv results/reports/leaderboard.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os

# (khoá nội bộ, tiêu đề cột, có phải %? , số chữ số)
COLUMNS = [
    ("model", "Model", False, None),
    ("dataset", "Dataset", False, None),
    ("n", "N", False, 0),
    ("wer", "WER%", True, 2),
    ("cer", "CER%", True, 2),
    ("wer_nd", "WER-nodấu%", True, 2),
    ("ser", "SER%", True, 2),
    ("ins", "I%", True, 2),
    ("dele", "D%", True, 2),
    ("sub", "S%", True, 2),
    ("rtf", "RTF", False, 4),
    ("xrt", "xRT", False, 1),
    ("vram", "VRAM_MB", False, 1),
    ("missing", "Missing", False, 0),
]


def _dataset_name(meta: dict) -> str:
    m = meta.get("manifest") or ""
    base = os.path.basename(m)
    return base[:-6] if base.endswith(".jsonl") else (base or "?")


def load_rows(metric_files):
    rows = []
    for path in metric_files:
        with open(path, "r", encoding="utf-8") as f:
            m = json.load(f)
        ov = m.get("overall", {})
        eff = m.get("efficiency", {})
        rows.append({
            "model": m.get("model") or "?",
            "dataset": _dataset_name(m),
            "n": m.get("num_utterances", 0),
            "wer": ov.get("wer"),
            "cer": ov.get("cer"),
            "wer_nd": ov.get("wer_no_diacritics"),
            "ser": ov.get("ser"),
            "ins": ov.get("ins_rate"),
            "dele": ov.get("del_rate"),
            "sub": ov.get("sub_rate"),
            "rtf": eff.get("rtf"),
            "xrt": eff.get("throughput_x_realtime"),
            "vram": eff.get("peak_vram_mb"),
            "missing": m.get("num_missing_hyp", 0),
            "_by_domain": m.get("by_domain", {}),
        })
    # xếp hạng: WER tăng dần (None xuống cuối)
    rows.sort(key=lambda r: (r["wer"] is None, r["wer"] if r["wer"] is not None else 0))
    return rows


def _fmt(key, val, is_pct, ndigits):
    if val is None:
        return "-"
    if is_pct:
        return f"{val * 100:.{ndigits}f}"
    if ndigits is not None:
        return f"{val:.{ndigits}f}" if ndigits > 0 else f"{int(round(val))}"
    return str(val)


def to_markdown(rows) -> str:
    head = "| " + " | ".join(c[1] for c in COLUMNS) + " |"
    sep = "| " + " | ".join("---" for _ in COLUMNS) + " |"
    lines = [head, sep]
    for r in rows:
        cells = [_fmt(k, r.get(k), pct, nd) for (k, _, pct, nd) in COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def to_domain_markdown(rows) -> str:
    """Bảng WER theo domain, mỗi model 1 dòng."""
    domains = sorted({d for r in rows for d in r["_by_domain"]})
    if not domains:
        return ""
    head = "| Model | Dataset | " + " | ".join(f"{d} WER%" for d in domains) + " |"
    sep = "| --- | --- | " + " | ".join("---" for _ in domains) + " |"
    lines = ["", "## WER theo domain", "", head, sep]
    for r in rows:
        cells = []
        for d in domains:
            w = r["_by_domain"].get(d, {}).get("wer")
            cells.append(f"{w * 100:.2f}" if w is not None else "-")
        lines.append(f"| {r['model']} | {r['dataset']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_csv(rows, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([c[1] for c in COLUMNS])
        for r in rows:
            w.writerow([_fmt(k, r.get(k), pct, nd) for (k, _, pct, nd) in COLUMNS])


def main(argv=None):
    ap = argparse.ArgumentParser(description="Gom metrics -> leaderboard")
    ap.add_argument("--metrics-dir", default="results/metrics",
                    help="thư mục chứa metrics.json (quét đệ quy)")
    ap.add_argument("--metrics", nargs="*", help="danh sách file metrics.json cụ thể (thay cho --metrics-dir)")
    ap.add_argument("--out-md", default="results/reports/leaderboard.md")
    ap.add_argument("--out-csv", default="results/reports/leaderboard.csv")
    ap.add_argument("--title", default="ASR Benchmark — Leaderboard")
    args = ap.parse_args(argv)

    files = args.metrics or sorted(glob.glob(os.path.join(args.metrics_dir, "**", "*.json"),
                                             recursive=True))
    if not files:
        raise SystemExit(f"Không tìm thấy metrics.json trong {args.metrics_dir}")

    rows = load_rows(files)
    md = (f"# {args.title}\n\n"
          f"_{len(rows)} kết quả · xếp theo WER tăng dần · % = nhân 100_\n\n"
          + to_markdown(rows) + "\n" + to_domain_markdown(rows) + "\n")

    for p in (args.out_md, args.out_csv):
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(md)
    write_csv(rows, args.out_csv)

    print(md)
    print(f"\n-> {args.out_md}\n-> {args.out_csv}")


if __name__ == "__main__":
    main()
