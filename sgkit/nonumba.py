# A simple attempt to replace numba when running on Pyodide.
# Note that guvectorize is not supported.

# These functions simple disable jitting


def jit(
    signature_or_function=None,
    locals={},
    cache=False,
    pipeline_class=None,
    boundscheck=None,
    **options
):
    wrapper = _jit()
    if signature_or_function is None:
        return wrapper
    else:
        return wrapper(signature_or_function)


def _jit():
    def wrapper(func):
        return func

    return wrapper


def njit(*args, **kwargs):
    return jit(*args, **kwargs)


# It is not possible to easily rewrite guvectorize'd functions in pure Python, so raise an exception when they are called.


def guvectorize(*args, **kwargs):
    def always_raise(*a, **k):
        raise NotImplementedError("No pure Python implementation of guvectorize")

    def wrapper(func):
        return always_raise

    return wrapper
