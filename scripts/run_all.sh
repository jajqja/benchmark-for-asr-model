#!/usr/bin/env bash
# Chấm điểm mọi hypotheses đã có + dựng leaderboard. Không cần GPU.
# Dùng sau khi run_inference đã sinh results/hypotheses/<model>/<dataset>.jsonl.
#
#   bash scripts/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src

NORM=configs/normalization/vi.yaml

shopt -s nullglob
for hyp in results/hypotheses/*/*.jsonl; do
  case "$hyp" in *.meta.json) continue;; esac
  model=$(basename "$(dirname "$hyp")")
  ds=$(basename "$hyp" .jsonl)
  manifest="data/manifests/${ds}.jsonl"
  if [ ! -f "$manifest" ]; then
    echo "BỎ QUA $hyp — không thấy $manifest" >&2; continue
  fi
  echo "=== chấm $model / $ds ==="
  python -m benchmark.runners.run_scoring \
    --manifest "$manifest" --hyp "$hyp" --norm-config "$NORM" \
    --model "$model" --out "results/metrics/${model}/${ds}.json"
  python -m benchmark.report.error_analysis \
    --manifest "$manifest" --hyp "$hyp" --norm-config "$NORM" \
    --model "$model" --top 25 \
    --out-md "results/reports/errors_${model}_${ds}.md"
done

echo "=== leaderboard ==="
python -m benchmark.report.build_report \
  --metrics-dir results/metrics \
  --out-md results/reports/leaderboard.md \
  --out-csv results/reports/leaderboard.csv
