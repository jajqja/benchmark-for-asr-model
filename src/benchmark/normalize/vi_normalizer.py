"""Chuẩn hoá text tiếng Việt dùng chung cho reference và hypothesis.

Nguyên tắc: cùng một cấu hình được áp cho CẢ ref lẫn hyp của MỌI model, nếu không
so sánh WER giữa các model sẽ không công bằng.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Ký tự thay dấu câu: mọi ký tự KHÔNG phải chữ/số/khoảng trắng -> khoảng trắng.
# \w trong chế độ unicode giữ lại chữ có dấu tiếng Việt.
_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")

_UNITS = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]
_SCALES = ["", "nghìn", "triệu", "tỷ"]  # hỗ trợ tới < 10^12
_DIGITS_RE = re.compile(r"\d+")


def remove_diacritics(text: str) -> str:
    """Bỏ dấu thanh/dấu phụ tiếng Việt: 'việt' -> 'viet', 'đ' -> 'd'."""
    text = text.replace("đ", "d").replace("Đ", "D")
    nfd = unicodedata.normalize("NFD", text)
    stripped = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    return unicodedata.normalize("NFC", stripped)


def _read_group(n: int, show_hundred: bool) -> list[str]:
    """Đọc một nhóm 3 chữ số (0..999). `show_hundred`=True buộc đọc phần trăm."""
    tram, chuc, donvi = n // 100, (n % 100) // 10, n % 10
    parts: list[str] = []
    if tram > 0 or show_hundred:
        parts += [_UNITS[tram], "trăm"]
    if chuc == 0:
        if donvi > 0:
            if tram > 0 or show_hundred:
                parts.append("linh")
            parts.append(_UNITS[donvi])
    else:
        parts.append("mười" if chuc == 1 else f"{_UNITS[chuc]} mươi")
        if donvi == 1 and chuc >= 2:
            parts.append("mốt")
        elif donvi == 5 and chuc >= 1:
            parts.append("lăm")
        elif donvi > 0:
            parts.append(_UNITS[donvi])
    return parts


def num_to_vietnamese(n: int) -> str:
    """Đổi số nguyên không âm (< 10^12) sang dạng đọc tiếng Việt. EXPERIMENTAL."""
    if n == 0:
        return "không"
    groups: list[int] = []
    while n > 0:
        groups.append(n % 1000)
        n //= 1000
    words: list[str] = []
    for idx in range(len(groups) - 1, -1, -1):
        g = groups[idx]
        if g == 0:
            continue
        show_hundred = idx != len(groups) - 1  # nhóm không phải nhóm đầu -> đọc đủ trăm
        words += _read_group(g, show_hundred)
        if _SCALES[idx]:
            words.append(_SCALES[idx])
    return " ".join(w for w in words if w)


def _normalize_numbers(text: str) -> str:
    return _DIGITS_RE.sub(lambda m: num_to_vietnamese(int(m.group())), text)


@dataclass
class ViNormalizer:
    lowercase: bool = True
    unicode_form: str | None = "NFC"
    strip_punct: bool = True
    collapse_whitespace: bool = True
    normalize_numbers: bool = False
    remove_diacritics: bool = False

    def __call__(self, text: str) -> str:
        if text is None:
            return ""
        if self.unicode_form:
            text = unicodedata.normalize(self.unicode_form, text)
        if self.lowercase:
            text = text.lower()
        if self.normalize_numbers:
            text = _normalize_numbers(text)
        if self.strip_punct:
            text = _PUNCT_RE.sub(" ", text)
        if self.remove_diacritics:
            text = remove_diacritics(text)
        if self.collapse_whitespace:
            text = _WS_RE.sub(" ", text).strip()
        return text

    def with_diacritics_removed(self) -> "ViNormalizer":
        """Trả về bản sao có bật remove_diacritics — dùng cho 'WER không dấu'."""
        return ViNormalizer(
            lowercase=self.lowercase,
            unicode_form=self.unicode_form,
            strip_punct=self.strip_punct,
            collapse_whitespace=self.collapse_whitespace,
            normalize_numbers=self.normalize_numbers,
            remove_diacritics=True,
        )

    @classmethod
    def from_config(cls, cfg: dict) -> "ViNormalizer":
        fields = {
            "lowercase", "unicode_form", "strip_punct",
            "collapse_whitespace", "normalize_numbers", "remove_diacritics",
        }
        return cls(**{k: v for k, v in (cfg or {}).items() if k in fields})
