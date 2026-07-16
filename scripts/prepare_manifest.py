"""Pha 0 — ghép audio + transcript rời thành manifest jsonl.

Hai chế độ (khai báo trong configs/datasets/*.yaml, mục `prepare`):
  per_file : mỗi audio có 1 file transcript cùng tên gốc (0001.wav -> 0001.txt)
  table    : một file bảng tsv/csv chứa cột id + text; audio khớp theo id

Ví dụ:
  python scripts/prepare_manifest.py --config configs/datasets/my_data.yaml
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import wave

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from benchmark.data.manifest import write_jsonl  # noqa: E402


def _audio_duration(path: str) -> float | None:
    """Đọc thời lượng (giây). Ưu tiên soundfile, fallback module `wave` cho .wav."""
    try:
        import soundfile as sf
        info = sf.info(path)
        return round(info.frames / info.samplerate, 3)
    except Exception:
        pass
    if path.lower().endswith(".wav"):
        try:
            with wave.open(path, "rb") as w:
                return round(w.getnframes() / w.getframerate(), 3)
        except Exception:
            return None
    return None


def _read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return " ".join(f.read().split())


def _load_yaml(path: str) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _list_audio(audio_dir: str, ext: str) -> dict[str, str]:
    """Trả về map stem -> đường dẫn audio."""
    out: dict[str, str] = {}
    for name in sorted(os.listdir(audio_dir)):
        if name.lower().endswith(ext.lower()):
            stem = os.path.splitext(name)[0]
            out[stem] = os.path.join(audio_dir, name)
    return out


def build_per_file(p: dict, default_domain: str):
    audio = _list_audio(p["audio_dir"], p.get("audio_ext", ".wav"))
    tdir = p["transcript_dir"]
    text_ext = p.get("transcript_ext", ".txt")
    rows, missing_txt = [], []
    for stem, apath in audio.items():
        tpath = os.path.join(tdir, stem + text_ext)
        if not os.path.exists(tpath):
            missing_txt.append(stem)
            continue
        rows.append(_make_row(stem, apath, _read_text_file(tpath), default_domain))
    return rows, missing_txt


def build_table(p: dict, default_domain: str):
    table_path = p["table_path"]
    id_col = p.get("id_column", "id")
    text_col = p.get("text_column", "text")
    audio = _list_audio(p["audio_dir"], p.get("audio_ext", ".wav"))
    delim = "\t" if table_path.lower().endswith(".tsv") else ","
    rows, missing_audio = [], []
    with open(table_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delim)
        for r in reader:
            uid = r[id_col].strip()
            apath = audio.get(uid)
            if apath is None:
                missing_audio.append(uid)
                continue
            rows.append(_make_row(uid, apath, " ".join(r[text_col].split()), default_domain))
    return rows, missing_audio


def _make_row(uid, apath, text, domain):
    rel = os.path.relpath(apath, _ROOT)
    return {
        "utt_id": uid,
        "audio": rel,
        "duration": _audio_duration(apath),
        "text": text,
        "domain": domain,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="Ghép audio + transcript -> manifest jsonl")
    ap.add_argument("--config", required=True, help="configs/datasets/*.yaml")
    ap.add_argument("--out", default=None, help="ghi đè đường dẫn manifest trong config")
    args = ap.parse_args(argv)

    cfg = _load_yaml(args.config)
    p = cfg.get("prepare", {})
    out_path = args.out or cfg["manifest"]
    default_domain = p.get("default_domain", "all")
    mode = p.get("mode", "per_file")

    if mode == "per_file":
        rows, missing = build_per_file(p, default_domain)
        miss_label = "audio thiếu transcript"
    elif mode == "table":
        rows, missing = build_table(p, default_domain)
        miss_label = "id trong bảng thiếu audio"
    else:
        ap.error(f"mode không hợp lệ: {mode!r}")

    no_dur = sum(1 for r in rows if r["duration"] is None)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    n = write_jsonl(out_path, rows)

    print(f"Ghi {n} utterance -> {out_path}", file=sys.stderr)
    if no_dur:
        print(f"CẢNH BÁO: {no_dur} file không đọc được duration (RTF sẽ thiếu).", file=sys.stderr)
    if missing:
        print(f"CẢNH BÁO: {len(missing)} {miss_label}; ví dụ: {missing[:10]}", file=sys.stderr)


if __name__ == "__main__":
    main()
