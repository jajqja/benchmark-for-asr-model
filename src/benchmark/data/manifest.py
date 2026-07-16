"""Đọc/validate manifest và hypotheses (định dạng jsonl)."""
from __future__ import annotations

import json

REQUIRED_MANIFEST_KEYS = ("utt_id", "audio", "text")


def read_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: JSON không hợp lệ: {e}") from e
    return rows


def write_jsonl(path: str, rows) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def load_manifest(path: str) -> list[dict]:
    rows = read_jsonl(path)
    seen: set[str] = set()
    for i, r in enumerate(rows):
        missing = [k for k in REQUIRED_MANIFEST_KEYS if k not in r]
        if missing:
            raise ValueError(f"{path}: dòng {i + 1} thiếu trường {missing}")
        uid = r["utt_id"]
        if uid in seen:
            raise ValueError(f"{path}: utt_id trùng lặp: {uid!r}")
        seen.add(uid)
        r.setdefault("domain", "all")
        r.setdefault("duration", None)
    return rows


def load_hypotheses(path: str) -> dict[str, dict]:
    """Trả về map utt_id -> record. Yêu cầu tối thiểu: utt_id, text."""
    rows = read_jsonl(path)
    out: dict[str, dict] = {}
    for i, r in enumerate(rows):
        if "utt_id" not in r or "text" not in r:
            raise ValueError(f"{path}: dòng {i + 1} thiếu utt_id/text")
        out[r["utt_id"]] = r
    return out
