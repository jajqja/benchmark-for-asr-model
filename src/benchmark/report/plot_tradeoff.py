"""Pha 4 — vẽ scatter đánh đổi độ chính xác ↔ tốc độ từ leaderboard.csv.

Mỗi dataset 1 hàng, 2 panel:
  * OFFLINE (chunkformer, gipformer): X = RTF (chi phí tính toán), thang log.
  * STREAMING (zipformer, nemotron): X = latency (độ trễ thuật toán, suy từ
    lookahead/chunk trong config `latency_ms`). Với model streaming, RTF KHÔNG phải
    trục đánh đổi đúng (lookahead lớn thường RTF thấp hơn NHƯNG latency cao hơn).
Trục Y = WER% -> góc DƯỚI-TRÁI là tốt nhất. Màu theo family; variant cùng family nối đường.

Bảng màu: Okabe-Ito (CVD-safe), đã validate (mọi cặp ΔE >= 8 dưới mô phỏng mù màu).

Ví dụ:
  PYTHONPATH=src python -m benchmark.report.plot_tradeoff \
    --csv results/reports/leaderboard.csv \
    --out results/reports/tradeoff.png
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
from collections import defaultdict

# Gán màu CỐ ĐỊNH theo nhóm (thứ tự cố định, không xoay vòng).
FAMILY_COLORS = {
    "zipformer": "#0072B2",   # blue
    "gipformer": "#D55E00",   # vermillion
    "chunkformer": "#009E73",  # green
    "nemotron": "#E69F00",    # orange
}
_FALLBACK = ["#CC79A7", "#56B4E9", "#F0E442", "#000000"]

INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#e6e6e6"
# offset nhãn so le (points) để các điểm gần trùng không đè chữ lên nhau
_LABEL_OFFSETS = [(6, 5), (6, -11), (-34, 5), (-34, -11)]


def _family(model: str) -> str:
    return model.split("_")[0]


def _variant_tag(model: str) -> str:
    return model.split("_")[-1]


# frame duration (ms) để suy latency từ tên variant khi config không khai latency_ms
_FRAME_MS = {"nemotron": 80, "zipformer": 40}  # zipformer là ước lượng


def _derive_latency(model: str):
    """Suy latency_ms từ tag: nemotron_*_la<N> -> N×80; zipformer_*_chunk<N> -> N×40."""
    import re
    fam = _family(model)
    m = re.search(r"(?:la|chunk)(\d+)$", model)
    if m and fam in _FRAME_MS:
        return int(m.group(1)) * _FRAME_MS[fam]
    return None


def _latency_of(model, latency_map):
    v = latency_map.get(model)
    return v if v is not None else _derive_latency(model)


def _num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def load_points(csv_path):
    """-> {dataset: [ {model, family, tag, rtf, wer}, ... ]}"""
    by_ds = defaultdict(list)
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rtf, wer = _num(r.get("RTF")), _num(r.get("WER%"))
            if rtf is None or wer is None:
                continue  # thiếu timing -> bỏ (RTF chỉ có nghĩa khi đo trên GPU)
            by_ds[r.get("Dataset", "all")].append({
                "model": r["Model"], "family": _family(r["Model"]),
                "tag": _variant_tag(r["Model"]), "rtf": rtf, "wer": wer,
            })
    return by_ds


def _color(family, assigned):
    if family in FAMILY_COLORS:
        return FAMILY_COLORS[family]
    if family not in assigned:
        assigned[family] = _FALLBACK[len(assigned) % len(_FALLBACK)]
    return assigned[family]


def load_latency_map(models_dir="configs/models"):
    """Đọc latency_ms từ config từng variant -> {model_name: latency_ms}.

    Chỉ những model streaming có khai latency_ms. Family của các model đó = family
    streaming (vẽ theo trục latency thay vì RTF).
    """
    import yaml
    out = {}
    for p in glob.glob(os.path.join(models_dir, "*.yaml")):
        try:
            cfg = yaml.safe_load(open(p, encoding="utf-8")) or {}
        except Exception:
            continue
        lat = (cfg.get("params") or {}).get("latency_ms")
        if lat is not None and cfg.get("name"):
            out[cfg["name"]] = float(lat)
    return out


def _draw_panel(ax, pts, xget, xlabel, title, xscale, assigned, seen, plt):
    """Vẽ 1 panel: điểm theo family, nối đường nội bộ, nhãn variant so le."""
    by_fam = defaultdict(list)
    for p in pts:
        if xget(p) is not None:
            by_fam[p["family"]].append(p)
    for fam in sorted(by_fam):
        fam_pts = sorted(by_fam[fam], key=xget)
        color = _color(fam, assigned)
        if fam not in seen:
            seen.append(fam)
        xs = [xget(p) for p in fam_pts]
        ys = [p["wer"] for p in fam_pts]
        ax.plot(xs, ys, "-", color=color, lw=1.5, alpha=0.55, zorder=2)
        ax.scatter(xs, ys, s=70, color=color, edgecolors="white",
                   linewidths=1.5, zorder=3, label=fam)
        for k, p in enumerate(fam_pts):
            dx, dy = _LABEL_OFFSETS[k % len(_LABEL_OFFSETS)]
            ax.annotate(p["tag"], (xget(p), p["wer"]), textcoords="offset points",
                        xytext=(dx, dy), fontsize=8, color=MUTED)
    ax.set_title(title, fontsize=11, color=INK, pad=8)
    ax.set_xlabel(xlabel, fontsize=10, color=MUTED)
    ax.set_ylabel("WER %  (nhỏ = chính xác hơn)", fontsize=10, color=MUTED)
    if xscale == "log":
        ax.set_xscale("log")
    ax.grid(True, color=GRID, lw=0.8, zorder=0, which="both")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.margins(0.15)


def plot(by_ds, out_path, title, xscale="log", latency_map=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    latency_map = latency_map or {}
    streaming_fams = {_family(m) for m in latency_map}
    datasets = sorted(by_ds)
    # 2 cột: [offline vs RTF] | [streaming vs latency] — bỏ cột streaming nếu không có
    two_col = bool(streaming_fams)
    ncols = 2 if two_col else 1
    fig, axes = plt.subplots(len(datasets), ncols,
                             figsize=(6.4 * ncols, 5.0 * len(datasets)), squeeze=False)
    assigned, seen = {}, []

    for row, ds in enumerate(datasets):
        pts = by_ds[ds]
        offline = [p for p in pts if p["family"] not in streaming_fams]
        streaming = [p for p in pts if p["family"] in streaming_fams]
        rtf_lab = ("RTF, thang log  (nhỏ = nhanh hơn →)" if xscale == "log"
                   else "RTF  (nhỏ = nhanh hơn →)")

        _draw_panel(axes[row][0], offline if two_col else pts, lambda p: p["rtf"],
                    rtf_lab, f"{ds} · Offline — chi phí (RTF)" if two_col else ds,
                    xscale, assigned, seen, plt)
        if two_col:
            _draw_panel(axes[row][1], streaming,
                        lambda p: _latency_of(p["model"], latency_map),
                        "Độ trễ / latency (ms)  (← nhỏ = phản hồi nhanh)",
                        f"{ds} · Streaming — độ trễ (latency)",
                        "linear", assigned, seen, plt)

    handles, labels = [], []
    for axr in axes:
        for ax in axr:
            h, l = ax.get_legend_handles_labels()
            for hi, li in zip(h, l):
                if li not in labels:
                    handles.append(hi); labels.append(li)
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(title, y=1.05, fontsize=13, color=INK)
    fig.text(0.5, -0.02,
             "Góc dưới-trái = tốt nhất. Offline: RTF = chi phí tính toán. "
             "Streaming: latency = độ trễ thuật toán (zipformer latency là ước lượng).",
             ha="center", fontsize=8.5, color=MUTED)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"-> {out_path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Vẽ scatter accuracy vs speed")
    ap.add_argument("--csv", default="results/reports/leaderboard.csv")
    ap.add_argument("--out", default="results/reports/tradeoff.png")
    ap.add_argument("--title", default="ASR: đánh đổi độ chính xác ↔ tốc độ")
    ap.add_argument("--xscale", choices=["log", "linear"], default="log",
                    help="thang trục RTF (log = mặc định, tách cụm điểm dày)")
    ap.add_argument("--models-dir", default="configs/models",
                    help="đọc latency_ms để vẽ panel streaming vs latency")
    args = ap.parse_args(argv)

    by_ds = load_points(args.csv)
    if not by_ds:
        raise SystemExit(f"Không có điểm nào có RTF+WER trong {args.csv} "
                         "(cần chạy inference trên GPU để có RTF).")
    latency_map = load_latency_map(args.models_dir)
    plot(by_ds, args.out, args.title, xscale=args.xscale, latency_map=latency_map)


if __name__ == "__main__":
    main()
