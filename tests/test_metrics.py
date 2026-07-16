"""Kiểm chứng WER/CER và normalizer. Chạy: PYTHONPATH=src python tests/test_metrics.py"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from benchmark.metrics.wer import align, compute_wer, compute_cer, compute_ser
from benchmark.normalize.vi_normalizer import ViNormalizer, num_to_vietnamese, remove_diacritics


def approx(a, b, eps=1e-9):
    assert abs(a - b) < eps, f"{a} != {b}"


def test_align_basic():
    # ref 6 từ, hyp thiếu 1 từ ("the") -> 1 deletion
    c = align("the cat sat on the mat".split(), "the cat sat on mat".split())
    assert (c.sub, c.ins, c.dele) == (0, 0, 1), (c.sub, c.ins, c.dele)
    assert c.ref_len == 6
    approx(c.rates()["wer"], 1 / 6)


def test_align_sub_and_ins():
    c = align("a b c".split(), "a x c d".split())  # b->x (sub), +d (ins)
    assert (c.sub, c.ins, c.dele) == (1, 1, 0), (c.sub, c.ins, c.dele)
    approx(c.rates()["wer"], 2 / 3)


def test_wer_corpus():
    pairs = [("xin chào việt nam", "xin chào việt nam"),
             ("hôm nay trời đẹp", "hôm nay trời xấu")]
    r = compute_wer(pairs)
    approx(r["wer"], 1 / 8)  # 1 sub trên 8 từ


def test_cer():
    r = compute_cer([("abc", "abd")])  # 1 ký tự sai / 3
    approx(r["wer"], 1 / 3)


def test_ser():
    r = compute_ser([("a b", "a b"), ("c d", "c x")])
    approx(r["ser"], 0.5)


def test_perfect():
    pairs = [("một hai ba", "một hai ba")]
    approx(compute_wer(pairs)["wer"], 0.0)


def test_normalizer():
    n = ViNormalizer()
    assert n("Xin chào, Việt Nam!") == "xin chào việt nam"
    assert n("  nhiều   khoảng    trắng ") == "nhiều khoảng trắng"


def test_remove_diacritics():
    assert remove_diacritics("việt nam") == "viet nam"
    assert remove_diacritics("đường") == "duong"
    nd = ViNormalizer().with_diacritics_removed()
    assert nd("Đường Việt!") == "duong viet"


def test_number_norm():
    n = ViNormalizer(normalize_numbers=True)
    assert num_to_vietnamese(0) == "không"
    assert num_to_vietnamese(15) == "mười lăm"
    assert num_to_vietnamese(21) == "hai mươi mốt"
    assert num_to_vietnamese(100) == "một trăm"
    assert num_to_vietnamese(123) == "một trăm hai mươi ba"
    assert num_to_vietnamese(1000) == "một nghìn"
    assert num_to_vietnamese(1000000) == "một triệu"
    assert n("có 5 con mèo") == "có năm con mèo"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
