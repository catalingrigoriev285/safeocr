from datetime import date

from safeocr.pii.cnp import control_digit, find_cnps, is_valid_cnp, make_cnp


def test_make_cnp_is_valid():
    cnp = make_cnp(1, date(1985, 3, 12), 40, 123)
    assert len(cnp) == 13
    assert is_valid_cnp(cnp)


def test_control_digit_10_becomes_1():
    # search a prefix whose weighted sum % 11 == 10
    for serial in range(1000):
        first12 = f"1850312400{serial:03d}"[:12]
        total = sum(int(d) * int(w) for d, w in zip(first12, "279146358279"))
        if total % 11 == 10:
            assert control_digit(first12) == 1
            return
    raise AssertionError("no prefix with remainder 10 found")


def test_rejects_bad_checksum_date_and_county():
    good = make_cnp(2, date(1990, 7, 1), 12, 5)
    bad_check = good[:12] + str((int(good[12]) + 1) % 10)
    assert not is_valid_cnp(bad_check)
    assert not is_valid_cnp("1901332120057")  # month 13
    assert not is_valid_cnp("123")


def test_find_cnps_in_noisy_token():
    cnp = make_cnp(5, date(2004, 2, 29), 46, 77)
    assert list(find_cnps(f"CNP:{cnp}")) == [cnp]
    assert list(find_cnps("1234567890123")) == []
