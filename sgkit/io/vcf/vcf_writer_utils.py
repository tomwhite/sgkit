from numba import jit

from sgkit.io.utils import INT_FILL, INT_MISSING

COMMA = ord(",")
DOT = ord(".")
MINUS = ord("-")
ZERO = ord("0")


@jit(nopython=True)
def itoa(buf, p, value):
    """
    Convert an int32 ``value`` to its decimal representation, and store in a NumPy array (``buf``) at position ``p``.

    Returns the position in the buffer after the last character written.
    """
    if value < 0:
        buf[p] = MINUS
        p += 1
        value = -value
    # special case small values
    if value < 10:
        buf[p] = value + ZERO
        p += 1
    else:
        # this is significantly faster than `k = math.floor(math.log10(value))`
        if value < 100:
            k = 1
        elif value < 1000:
            k = 2
        elif value < 10000:
            k = 3
        elif value < 100000:
            k = 4
        elif value < 1000000:
            k = 5
        elif value < 10000000:
            k = 6
        elif value < 100000000:
            k = 7
        elif value < 1000000000:
            k = 8
        elif value < 10000000000:
            k = 9
        else:
            # exceeds int32
            raise ValueError("itoa only supports 32-bit integers")

        # iterate backwards in buf
        p += k
        buf[p] = (value % 10) + ZERO
        for _ in range(k):
            p -= 1
            value = value // 10
            buf[p] = (value % 10) + ZERO
        p += k + 1

    return p


def chars_to_str(a):
    """Convert a NumPy array of characters to a Python string"""
    return memoryview(a).tobytes().decode()


@jit(nopython=True)
def ints_to_decimal_chars(buf, p, indexes, a, separator=-1):
    """
    Convert a 1D or 2D array of ints (``a``) to their decimal representations, and store in a NumPy array (``buf``)
    starting at position ``p``.

    Values -1 and -2 are special and signify missing (represented by ``.``), and fill (end of row), respectively.

    The ``indexes`` array is updated to contain the start positions of each decimal written to the buffer,
    plus the end position after the last character written. This is used in the ``interleave`` function.

    For a 1D array, values are separated by the optional ``separator`` (default empty).

    For a 2D array, values in each row are separated by commas, and rows are separated by the optional
    ``separator`` (default empty).

    Returns the position in the buffer after the last character written.
    """
    n = 0  # total number of strings
    if a.ndim == 1:
        for i in range(a.shape[0]):
            indexes[n] = p
            if a[i] == INT_MISSING:
                buf[p] = DOT
                p += 1
            else:
                p = itoa(buf, p, a[i])
            if separator != -1:
                buf[p] = separator
                p += 1
            n += 1
    elif a.ndim == 2:
        for i in range(a.shape[0]):
            indexes[n] = p
            for j in range(a.shape[1]):
                if a[i, j] == INT_MISSING:
                    buf[p] = DOT
                    p += 1
                elif a[i, j] == INT_FILL:
                    break
                else:
                    p = itoa(buf, p, a[i, j])
                buf[p] = COMMA
                p += 1
            p -= 1
            n += 1
            if separator != -1:
                buf[p] = separator
                p += 1
    else:
        raise ValueError("Array must have dimension 1 or 2")
    if separator != -1:  # remove last separator
        p -= 1
    indexes[n] = p  # add index for end
    return p


@jit(nopython=True)
def interleave(buf, p, indexes, separator, samples_separator, *arrays):
    """
    Interleave character arrays into sample groups, and store in a NumPy array (``buf``)
    starting at position ``p``.

    The ``indexes`` array has one row for each array, and contains the start index for each
    separate string value in the array.

    Values are separated by ``separator`` within each sample group, and by ``samples_separator``
    between each sample group.

    Returns the position in the buffer after the last character written.
    """
    n = indexes.shape[0]
    assert n == len(arrays)
    for j in range(indexes.shape[1] - 1):
        for i in range(n):
            arr = arrays[i]
            sub = arr[indexes[i, j] : indexes[i, j + 1]]
            len_sub = sub.shape[0]
            buf[p : p + len_sub] = sub
            p = p + len_sub
            buf[p] = separator
            p += 1
        buf[p - 1] = samples_separator
    p -= 1  # remove last separator
    return p


def interleave_buf_size(indexes, *arrays):
    """Return the buffer size needed by ``interleave``."""
    # separators + array buffers
    return indexes.size + sum(a.shape[0] for a in arrays)
