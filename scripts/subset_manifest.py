"""Lấy mẫu con manifest, GIỮ TỈ LỆ domain (stratified). Tái lập được nhờ --seed.

Dùng để giảm thời gian benchmark mà vẫn đại diện: WER trên ~200 utterance hội thoại
(hàng chục nghìn từ) đủ ổn định để SO SÁNH các model.

Ví dụ:
  python scripts/subset_manifest.py \
    --in data/manifests/auto_autoele.jsonl \
    --out data/manifests/auto_autoele_sub200.jsonl \
    --n 200 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict


def read_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def stratified_sample(rows, n, seed, key="domain"):
    rng = random.Random(seed)
    by_group = defaultdict(list)
    for r in rows:
        by_group[r.get(key, "all")].append(r)

    total = len(rows)
    picked = []
    # phân bổ theo tỉ lệ, làm tròn; nhóm nào cũng lấy >=1 nếu có mặt
    for g, items in sorted(by_group.items()):
        k = max(1, round(n * len(items) / total)) if items else 0
        k = min(k, len(items))
        picked += rng.sample(items, k)

    # tinh chỉnh về đúng n (dôi thì bỏ bớt ngẫu nhiên, thiếu thì bù từ phần còn lại)
    if len(picked) > n:
        picked = rng.sample(picked, n)
    elif len(picked) < n:
        chosen = {r["utt_id"] for r in picked}
        rest = [r for r in rows if r["utt_id"] not in chosen]
        picked += rng.sample(rest, min(n - len(picked), len(rest)))

    picked.sort(key=lambda r: r["utt_id"])
    return picked


def main(argv=None):
    ap = argparse.ArgumentParser(description="Lấy mẫu con manifest giữ tỉ lệ domain")
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=None,
                    help="số utt sau lấy mẫu; bỏ trống = giữ tất cả (chỉ lọc theo duration)")
    ap.add_argument("--max-duration", type=float, default=None,
                    help="bỏ utt dài hơn N giây (vd model offline OOM ở audio dài)")
    ap.add_argument("--min-duration", type=float, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stratify-by", default="domain")
    args = ap.parse_args(argv)

    rows = read_jsonl(args.inp)
    before = len(rows)
    if args.max_duration is not None:
        rows = [r for r in rows if (r.get("duration") or 0) <= args.max_duration]
    if args.min_duration is not None:
        rows = [r for r in rows if (r.get("duration") or 0) >= args.min_duration]
    if before != len(rows):
        print(f"lọc duration: {len(rows)}/{before} utt còn lại", file=sys.stderr)

    if args.n is None or args.n >= len(rows):
        sub = sorted(rows, key=lambda r: r["utt_id"])
    else:
        sub = stratified_sample(rows, args.n, args.seed, args.stratify_by)

    with open(args.out, "w", encoding="utf-8") as f:
        for r in sub:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # báo cáo phân bố
    def dist(rs):
        c = defaultdict(int)
        for r in rs:
            c[r.get(args.stratify_by, "all")] += 1
        return dict(sorted(c.items()))
    dur = sum(r.get("duration") or 0 for r in sub) / 3600
    print(f"{len(sub)} utt -> {args.out}  ({dur:.1f}h audio)", file=sys.stderr)
    print(f"  gốc   : {dist(rows)}", file=sys.stderr)
    print(f"  subset: {dist(sub)}", file=sys.stderr)


if __name__ == "__main__":
    main()
