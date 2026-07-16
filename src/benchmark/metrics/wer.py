"""WER / CER tự implement bằng căn chỉnh Levenshtein (không phụ thuộc jiwer).

Trả về cả breakdown Insertion / Deletion / Substitution. Có thể đối chiếu với
jiwer về sau nếu muốn — nhưng lõi scoring không cần cài gì để chạy.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class EditCounts:
    hits: int = 0          # số token khớp
    sub: int = 0           # thay thế
    ins: int = 0           # chèn (thừa trong hyp)
    dele: int = 0          # xoá (thiếu trong hyp)
    ref_len: int = 0       # tổng số token reference

    def __add__(self, other: "EditCounts") -> "EditCounts":
        return EditCounts(
            self.hits + other.hits,
            self.sub + other.sub,
            self.ins + other.ins,
            self.dele + other.dele,
            self.ref_len + other.ref_len,
        )

    @property
    def errors(self) -> int:
        return self.sub + self.ins + self.dele

    def rates(self) -> dict:
        """Tỉ lệ theo số token reference. WER = (S+D+I)/N_ref."""
        n = self.ref_len if self.ref_len > 0 else 1
        return {
            "wer": self.errors / n,
            "sub_rate": self.sub / n,
            "del_rate": self.dele / n,
            "ins_rate": self.ins / n,
            "hits": self.hits,
            "sub": self.sub,
            "del": self.dele,
            "ins": self.ins,
            "ref_len": self.ref_len,
        }


def align(ref: list[str], hyp: list[str]) -> EditCounts:
    """Căn chỉnh 2 chuỗi token, đếm S/D/I/hits qua Levenshtein + backtrace.

    Quy ước ràng buộc: nếu hyp có token đứng một mình không khớp -> insertion;
    nếu ref có token bị bỏ -> deletion; khớp lệch -> substitution.
    """
    n, m = len(ref), len(hyp)
    # d[i][j] = số phép biến đổi tối thiểu để khớp ref[:i] với hyp[:j]
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = i
    for j in range(1, m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        ri = ref[i - 1]
        di = d[i]
        dim1 = d[i - 1]
        for j in range(1, m + 1):
            if ri == hyp[j - 1]:
                di[j] = dim1[j - 1]
            else:
                di[j] = 1 + min(dim1[j - 1], dim1[j], di[j - 1])

    # Backtrace để phân loại lỗi.
    counts = EditCounts(ref_len=n)
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and d[i][j] == d[i - 1][j - 1]:
            counts.hits += 1
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + 1:
            counts.sub += 1
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            counts.dele += 1  # ref có, hyp thiếu
            i -= 1
        else:
            counts.ins += 1   # hyp thừa
            j -= 1
    return counts


def align_ops(ref: list[str], hyp: list[str]) -> list[tuple]:
    """Như align() nhưng trả về danh sách phép căn chỉnh theo thứ tự xuôi.

    Mỗi phần tử: (op, ref_tok|None, hyp_tok|None) với op ∈ {equal, sub, del, ins}.
    Dùng cho error-analysis (gom cặp thay thế, từ bị nuốt/thêm).
    """
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = i
    for j in range(1, m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        ri, di, dim1 = ref[i - 1], d[i], d[i - 1]
        for j in range(1, m + 1):
            di[j] = dim1[j - 1] if ri == hyp[j - 1] else 1 + min(dim1[j - 1], dim1[j], di[j - 1])

    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and d[i][j] == d[i - 1][j - 1]:
            ops.append(("equal", ref[i - 1], hyp[j - 1])); i, j = i - 1, j - 1
        elif i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + 1:
            ops.append(("sub", ref[i - 1], hyp[j - 1])); i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append(("del", ref[i - 1], None)); i -= 1
        else:
            ops.append(("ins", None, hyp[j - 1])); j -= 1
    ops.reverse()
    return ops


def _tokenize_words(text: str) -> list[str]:
    return text.split()


def _tokenize_chars(text: str) -> list[str]:
    # CER tính trên từng ký tự, giữ nguyên khoảng trắng đơn (đã collapse ở normalize).
    return list(text)


def _corpus(pairs, tokenize) -> dict:
    total = EditCounts()
    for ref, hyp in pairs:
        total = total + align(tokenize(ref), tokenize(hyp))
    return total.rates()


def compute_wer(pairs) -> dict:
    """pairs: iterable các (ref_text, hyp_text) đã chuẩn hoá."""
    return _corpus(pairs, _tokenize_words)


def compute_cer(pairs) -> dict:
    return _corpus(pairs, _tokenize_chars)


def compute_ser(pairs) -> dict:
    """Sentence Error Rate: % câu không khớp hoàn toàn sau chuẩn hoá."""
    pairs = list(pairs)
    if not pairs:
        return {"ser": 0.0, "num_sentences": 0}
    wrong = sum(1 for ref, hyp in pairs if ref != hyp)
    return {"ser": wrong / len(pairs), "num_sentences": len(pairs)}
