"""Pha 4 — vẽ scatter đánh đổi độ chính xác ↔ tốc độ từ leaderboard.csv.

Trục X = RTF (nhỏ = nhanh), trục Y = WER% (nhỏ = chính xác) -> góc DƯỚI-TRÁI là tốt nhất.
Màu theo NHÓM model (family); các variant cùng nhóm nối bằng đường -> đường cong nội bộ.
Facet theo dataset (mỗi dataset 1 subplot).

Bảng màu: Okabe-Ito (CVD-safe), đã validate (mọi cặp ΔE >= 8 dưới mô phỏng mù màu).

Ví dụ:
  PYTHONPATH=src python -m benchmark.report.plot_tradeoff \
    --csv results/reports/leaderboard.csv \
    --out results/reports/tradeoff.png
"""
from __future__ import annotations

import argparse
import csv
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


def plot(by_ds, out_path, title, xscale="log"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = sorted(by_ds)
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 5.2), squeeze=False)
    axes = axes[0]
    assigned = {}
    seen_families = []

    for ax, ds in zip(axes, datasets):
        pts = by_ds[ds]
        by_fam = defaultdict(list)
        for p in pts:
            by_fam[p["family"]].append(p)

        for fam in sorted(by_fam):
            fam_pts = sorted(by_fam[fam], key=lambda p: p["rtf"])
            color = _color(fam, assigned)
            if fam not in seen_families:
                seen_families.append(fam)
            xs = [p["rtf"] for p in fam_pts]
            ys = [p["wer"] for p in fam_pts]
            # đường cong nội bộ nối các variant (thin)
            ax.plot(xs, ys, "-", color=color, lw=1.5, alpha=0.55, zorder=2)
            # điểm: marker >=8px, viền trắng 1.5px (surface ring khi chồng)
            ax.scatter(xs, ys, s=70, color=color, edgecolors="white",
                       linewidths=1.5, zorder=3, label=fam)
            # nhãn variant từng điểm, offset SO LE để bớt đè (điểm gần trùng nhau)
            for k, p in enumerate(fam_pts):
                dx, dy = _LABEL_OFFSETS[k % len(_LABEL_OFFSETS)]
                ax.annotate(p["tag"], (p["rtf"], p["wer"]),
                            textcoords="offset points", xytext=(dx, dy),
                            fontsize=8, color=MUTED)

        ax.set_title(ds, fontsize=12, color=INK, pad=8)
        xlab = "RTF, thang log  (nhỏ = nhanh hơn →)" if xscale == "log" \
            else "RTF  (nhỏ = nhanh hơn →)"
        ax.set_xlabel(xlab, fontsize=10, color=MUTED)
        ax.set_ylabel("WER %  (nhỏ = chính xác hơn)", fontsize=10, color=MUTED)
        if xscale == "log":
            ax.set_xscale("log")
        ax.grid(True, color=GRID, lw=0.8, zorder=0, which="both")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.margins(0.15)

    # legend chung (>=2 nhóm luôn có), 1 hàng trên cùng
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(title, y=1.06, fontsize=13, color=INK)
    fig.text(0.5, -0.02, "Góc dưới-trái = nhanh & chính xác nhất. "
             "RTF chỉ hợp lệ khi đo cùng phần cứng (A100).",
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
    args = ap.parse_args(argv)

    by_ds = load_points(args.csv)
    if not by_ds:
        raise SystemExit(f"Không có điểm nào có RTF+WER trong {args.csv} "
                         "(cần chạy inference trên GPU để có RTF).")
    plot(by_ds, args.out, args.title, xscale=args.xscale)


if __name__ == "__main__":
    main()
