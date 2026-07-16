"""Pha 4 — phân tích lỗi 1 model trên 1 dataset.

Soi ĐÂU model sai để hiểu hành vi và (về sau) curate lexicon brand nếu muốn:
  * utterance WER cao nhất (ref vs hyp cạnh nhau)
  * cặp thay thế hay gặp nhất  (ref_word -> hyp_word)  <- lộ lỗi hệ thống (brand, dấu thanh)
  * từ hay bị NUỐT (deletion) và hay bị THÊM (insertion)

Chuẩn hoá dùng CHUNG với scoring (configs/normalization/vi.yaml) để nhất quán.

Ví dụ:
  PYTHONPATH=src python -m benchmark.report.error_analysis \
    --manifest data/manifests/auto_autoele.jsonl \
    --hyp results/hypotheses/gipformer_65m/auto_autoele.jsonl \
    --model gipformer_65m --top 20 \
    --out-md results/reports/errors_gipformer_auto_autoele.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from benchmark.data.manifest import load_manifest, load_hypotheses  # noqa: E402
from benchmark.metrics.wer import align, align_ops  # noqa: E402
from benchmark.normalize.vi_normalizer import ViNormalizer  # noqa: E402


def _load_yaml(path: str) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def analyze(manifest_path, hyp_path, norm_cfg, top=20, empty_hyp="empty"):
    manifest = load_manifest(manifest_path)
    hyps = load_hypotheses(hyp_path)
    norm = ViNormalizer.from_config(norm_cfg)

    per_utt = []
    subs, dels, inss = Counter(), Counter(), Counter()
    for row in manifest:
        h = hyps.get(row["utt_id"])
        if h is None and empty_hyp == "skip":
            continue
        hyp_text = (h.get("text", "") if h else "") or ""
        ref_n, hyp_n = norm(row["text"]), norm(hyp_text)
        r_toks, h_toks = ref_n.split(), hyp_n.split()
        counts = align(r_toks, h_toks)
        wer = counts.rates()["wer"]
        for op, rt, ht in align_ops(r_toks, h_toks):
            if op == "sub":
                subs[(rt, ht)] += 1
            elif op == "del":
                dels[rt] += 1
            elif op == "ins":
                inss[ht] += 1
        per_utt.append({
            "utt_id": row["utt_id"], "domain": row.get("domain", "all"),
            "wer": wer, "ref_words": counts.ref_len,
            "ref": ref_n, "hyp": hyp_n,
        })

    per_utt.sort(key=lambda x: (x["wer"], x["ref_words"]), reverse=True)
    return {
        "num_utterances": len(per_utt),
        "worst": per_utt[:top],
        "top_substitutions": [{"ref": r, "hyp": h, "count": c}
                              for (r, h), c in subs.most_common(top)],
        "top_deletions": [{"word": w, "count": c} for w, c in dels.most_common(top)],
        "top_insertions": [{"word": w, "count": c} for w, c in inss.most_common(top)],
    }


def to_markdown(res, model, dataset) -> str:
    L = [f"# Error analysis — {model} / {dataset}", "",
         f"_{res['num_utterances']} utterance · chuẩn hoá giống scoring_", ""]

    L += ["## Utterance WER cao nhất", "",
          "| WER% | #từ | utt_id | ref → hyp |", "| --- | --- | --- | --- |"]
    for u in res["worst"]:
        ref = u["ref"][:120] + ("…" if len(u["ref"]) > 120 else "")
        hyp = u["hyp"][:120] + ("…" if len(u["hyp"]) > 120 else "")
        L.append(f"| {u['wer']*100:.1f} | {u['ref_words']} | `{u['utt_id'][:22]}` | "
                 f"**ref:** {ref}<br>**hyp:** {hyp} |")

    L += ["", "## Cặp thay thế hay gặp (ref → hyp)", "",
          "| # | ref | → | hyp |", "| --- | --- | --- | --- |"]
    for s in res["top_substitutions"]:
        L.append(f"| {s['count']} | {s['ref']} | → | {s['hyp']} |")

    L += ["", "## Từ hay bị NUỐT (deletion)", "", "| # | từ |", "| --- | --- |"]
    for d in res["top_deletions"]:
        L.append(f"| {d['count']} | {d['word']} |")

    L += ["", "## Từ hay bị THÊM (insertion)", "", "| # | từ |", "| --- | --- |"]
    for i in res["top_insertions"]:
        L.append(f"| {i['count']} | {i['word']} |")

    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Phân tích lỗi ASR")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--hyp", required=True)
    ap.add_argument("--norm-config", default="configs/normalization/vi.yaml")
    ap.add_argument("--model", default="model")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--out-md", default=None)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--missing", choices=["empty", "skip"], default="empty")
    args = ap.parse_args(argv)

    res = analyze(args.manifest, args.hyp, _load_yaml(args.norm_config),
                  top=args.top, empty_hyp=args.missing)
    dataset = os.path.basename(args.manifest)[:-6] if args.manifest.endswith(".jsonl") \
        else os.path.basename(args.manifest)
    md = to_markdown(res, args.model, dataset)

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print(f"-> {args.out_json}", file=sys.stderr)
    if args.out_md:
        os.makedirs(os.path.dirname(args.out_md) or ".", exist_ok=True)
        with open(args.out_md, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"-> {args.out_md}", file=sys.stderr)
    if not args.out_md and not args.out_json:
        print(md)


if __name__ == "__main__":
    main()
