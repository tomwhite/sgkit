import numpy as np
import pytest

from sgkit.io.utils import INT_MISSING
from sgkit.io.vcf.vcf_writer_utils import (
    chars_to_str,
    interleave,
    interleave_buf_size,
    ints_to_decimal_chars,
    itoa,
)


@pytest.mark.parametrize(
    "i",
    [
        0,
        1,
        9,
        10,
        11,
        99,
        100,
        101,
        999,
        1000,
        1001,
        9999,
        10000,
        10001,
        np.iinfo(np.int32).max,
        np.iinfo(np.int32).min,
    ],
)
def test_itoa(i):
    buf = np.empty(30, dtype=np.int8)

    a = str(i)
    p = itoa(buf, 0, i)
    assert p == len(a)
    assert chars_to_str(buf[:p]) == a

    if i > 0:
        i = -i
        a = str(i)
        p = itoa(buf, 0, i)
        assert p == len(a)
        assert chars_to_str(buf[:p]) == a


def test_itoa_out_of_range():
    buf = np.empty(30, dtype=np.int8)
    with pytest.raises(ValueError, match=r"itoa only supports 32-bit integers"):
        itoa(buf, 0, np.iinfo(np.int32).max * 10)


def _check_indexes(buf, indexes, separator):
    if separator == ord(" "):
        s = chars_to_str(buf)
        words = []
        for i in range(len(indexes) - 1):
            words.append(s[indexes[i] : indexes[i + 1]].strip())
        assert words == s.split(" ")


@pytest.mark.parametrize(
    "separator, result",
    [
        (-1, "012.456789101112"),
        (ord(" "), "0 1 2 . 4 5 6 7 8 9 10 11 12"),
    ],
)
def test_ints_to_decimal_chars(separator, result):
    a = np.arange(13)
    a[3] = INT_MISSING

    buf = np.empty(a.shape[0] * 20, dtype=np.int8)
    indexes = np.empty(a.shape[0] + 1, dtype=np.int32)
    p = ints_to_decimal_chars(buf, 0, indexes, a, separator=separator)
    buf = buf[:p]

    assert chars_to_str(buf) == result
    _check_indexes(buf, indexes, separator)


@pytest.mark.parametrize(
    "separator, result",
    [
        (-1, "0,21,43.,11"),
        (ord(" "), "0,21,43 .,1 1"),
    ],
)
def test_ints_to_decimal_chars_2d(separator, result):
    a = np.array(
        [
            [0, 21, 43],
            [-1, 1, -2],
            [1, -2, -2],
        ],
        dtype=np.int32,
    )

    buf = np.empty(a.shape[0] * 20, dtype=np.int8)
    indexes = np.empty(a.shape[0] + 1, dtype=np.int32)
    p = ints_to_decimal_chars(buf, 0, indexes, a, separator=separator)
    buf = buf[:p]

    assert chars_to_str(buf) == result
    _check_indexes(buf, indexes, separator)


def test_interleave():
    a = np.arange(6)
    b = np.arange(6, 12)
    c = np.arange(12, 18)

    assert a.shape[0] == b.shape[0] == c.shape[0]

    n = a.shape[0]

    a_buf = np.empty(n * 20, dtype=np.int8)
    b_buf = np.empty(n * 20, dtype=np.int8)
    c_buf = np.empty(n * 20, dtype=np.int8)

    indexes = np.empty((3, n + 1), dtype=np.int32)

    a_p = ints_to_decimal_chars(a_buf, 0, indexes[0], a)
    b_p = ints_to_decimal_chars(b_buf, 0, indexes[1], b)
    c_p = ints_to_decimal_chars(c_buf, 0, indexes[2], c)

    a_ch = a_buf[:a_p]
    b_ch = b_buf[:b_p]
    c_ch = c_buf[:c_p]

    assert chars_to_str(a_ch) == "012345"
    assert chars_to_str(b_ch) == "67891011"
    assert chars_to_str(c_ch) == "121314151617"

    buf_size = interleave_buf_size(indexes, a_buf, b_buf, c_buf)
    buf = np.empty(buf_size, dtype=np.int8)

    p = interleave(buf, 0, indexes, ord(":"), ord(" "), a_buf, b_buf, c_buf)
    buf = buf[:p]

    assert chars_to_str(buf) == "0:6:12 1:7:13 2:8:14 3:9:15 4:10:16 5:11:17"


def test_interleave_speed():
    n_samples = 100000
    a = np.arange(0, n_samples)
    b = np.arange(1, n_samples + 1)
    c = np.arange(2, n_samples + 2)

    assert a.shape[0] == b.shape[0] == c.shape[0]

    n = a.shape[0]

    a_buf = np.empty(n * 20, dtype=np.int8)
    b_buf = np.empty(n * 20, dtype=np.int8)
    c_buf = np.empty(n * 20, dtype=np.int8)

    indexes = np.empty((3, n + 1), dtype=np.int32)

    buf_size = interleave_buf_size(indexes, a_buf, b_buf, c_buf)
    buf = np.empty(buf_size, dtype=np.int8)

    import time

    start = time.time()

    reps = 200
    bytes_written = 0
    for _ in range(reps):

        print(".", end="")

        ints_to_decimal_chars(a_buf, 0, indexes[0], a)
        ints_to_decimal_chars(b_buf, 0, indexes[1], b)
        ints_to_decimal_chars(c_buf, 0, indexes[2], c)

        p = interleave(buf, 0, indexes, ord(":"), ord(" "), a_buf, b_buf, c_buf)

        bytes_written += len(chars_to_str(buf[:p]))

    end = time.time()
    print(f"bytes written: {bytes_written}")
    print(f"duration: {end-start}")
    print(f"speed: {bytes_written/(1000000*(end-start))} MB/s")
