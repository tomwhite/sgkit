import allel
import dask.array as da
import numpy as np
import pytest

from sgkit.window import moving_statistic


@pytest.mark.parametrize("length, chunks, size, step", [(12, 6, 4, 4), (12, 6, 4, 2)])
def test_moving_statistic_dask(length, chunks, size, step):
    values = da.from_array(np.arange(length), chunks=chunks)

    stat = moving_statistic(values, np.sum, size=size, step=step, dtype=values.dtype)

    values_sa = np.arange(length)
    stat_sa = allel.moving_statistic(values_sa, np.sum, size=size, step=step)

    np.testing.assert_equal(stat.compute(), stat_sa)


@pytest.mark.parametrize("length, chunks, size, step", [(12, 6, 4, 4), (12, 6, 4, 2)])
def test_moving_statistic_dask_2d(length, chunks, size, step):
    arr = np.arange(length * 3).reshape(length, 3)

    def sum_cols(x):
        return np.sum(x, axis=0)

    values = da.from_array(arr, chunks=chunks)
    stat = moving_statistic(values, sum_cols, size=size, step=step, dtype=values.dtype)

    values_sa = arr
    stat_sa = allel.moving_statistic(values_sa, sum_cols, size=size, step=step)

    np.testing.assert_equal(stat.compute(), stat_sa)
