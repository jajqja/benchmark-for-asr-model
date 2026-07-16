"""Lọc file hypotheses xuống đúng tập utt_id của một manifest.

Dùng khi đã lỡ chạy inference trên tập LỚN nhưng muốn chấm trên tập CON: tập lớn đã
chứa sẵn các utt của tập con, nên chỉ cần lọc — KHÔNG phải chạy lại.

Ví dụ:
  python scripts/filter_hyp.py \
    --hyp   results/hypotheses/zipformer_30m_chunk16/auto_autoele.jsonl \
    --manifest data/manifests/auto_autoele_sub200.jsonl \
    --out   results/hypotheses/zipformer_30m_chunk16/auto_autoele_sub200.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(description="Lọc hypotheses theo utt_id của manifest")
    ap.add_argument("--hyp", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    keep = {json.loads(l)["utt_id"] for l in open(args.manifest, encoding="utf-8") if l.strip()}
    kept = missing = 0
    seen = set()
    with open(args.hyp, encoding="utf-8") as fin, open(args.out, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["utt_id"] in keep:
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
                seen.add(r["utt_id"])
                kept += 1
    missing = len(keep - seen)
    print(f"giữ {kept}/{len(keep)} utt -> {args.out}", file=sys.stderr)
    if missing:
        print(f"CẢNH BÁO: {missing} utt trong manifest KHÔNG có trong hypotheses "
              "(tập lớn chưa chạy đủ?)", file=sys.stderr)


if __name__ == "__main__":
    main()
