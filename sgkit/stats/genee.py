import dask.array as da
import numpy as np
import pandas as pd
from dask.dataframe import DataFrame
from dask.utils import import_required
from sklearn.mixture import GaussianMixture
from xarray import Dataset

from sgkit.stats.ld import map_windows_as_dataframe
from sgkit.typing import ArrayLike


def genee(ds: Dataset, ld: ArrayLike, *, reg_covar: float = 0.000001) -> DataFrame:
    """Compute gene-ε as described in Cheng, et al. 2020 [1].

    Parameters
    ----------
    ds
        Dataset containing beta values (OLS betas or regularized betas).
    ld
        2D array of LD values.
    reg_covar
        Non-negative regularization added to the diagonal of covariance.
        Passed to scikit-learn ``GaussianMixture``.

    Warnings
    --------
    Unlike the implementation in [2], this function will always use the
    second mixture component with the largest variance, rather than
    the first mixture component with the largest variance if it is composed
    of more than 50% of the SNPs.

    Returns
    -------
    A dataframe containing the following fields:

    - ``test_q``: test statistic
    - ``q_var``: test variance
    - ``pval``: p-value

    References
    ----------
    [1] - W. Cheng, S. Ramachandran, and L. Crawford (2020).
    Estimation of non-null SNP effect size distributions enables the detection of enriched genes underlying complex traits.
    PLOS Genetics. 16(6): e1008855.

    [2] - https://github.com/ramachandran-lab/genee
    """

    import_required(
        "chiscore",
        "Running `genee` requires the `chiscore` python library.\n\n"
        "Please either conda or pip install as follows:\n\n"
        "  conda install -c conda-forge chiscore\n"
        "  python -m pip install chiscore",
    )

    betas = np.expand_dims(ds["beta"].values, 1)
    epsilon_effect = genee_EM(betas=betas, reg_covar=reg_covar)

    betas = da.asarray(betas)
    ld = da.asarray(ld)

    meta = [
        ("test_q", np.float32),
        ("q_var", np.float32),
        ("pval", np.float32),
    ]
    return map_windows_as_dataframe(
        genee_loop_chunk,
        betas,
        ld,
        window_starts=ds["window_start"].values,
        window_stops=ds["window_stop"].values,
        meta=meta,
        epsilon_effect=epsilon_effect,
    )


def genee_EM(betas, reg_covar=0.000001):
    # based on https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html#sphx-glr-auto-examples-mixture-plot-gmm-selection-py
    lowest_bic = np.inf
    for n_components in range(1, 10):
        gmm = GaussianMixture(
            n_components=n_components, reg_covar=reg_covar, random_state=0
        ).fit(betas)
        bic = gmm.bic(betas)
        if bic < lowest_bic:
            lowest_bic = bic
            best_gmm = gmm

    covars = best_gmm.covariances_.squeeze()
    if best_gmm.n_components == 1:  # pragma: no cover
        epsilon_effect = covars[0]
    else:
        # TODO: handle case where first component composed more than 50% SNPs
        # https://github.com/ramachandran-lab/genee/blob/a357a956241df93f16e07664e24f3aeac65f4177/genee/R/genee_EM.R#L28-L29
        covars_decreasing = np.sort(covars)[::-1]
        epsilon_effect = covars_decreasing[1]

    return epsilon_effect


def genee_loop_chunk(
    args,
    chunk_window_starts,
    chunk_window_stops,
    abs_chunk_start,
    chunk_max_window_start,
    epsilon_effect,
):
    # Iterate over each window in this chunk
    # Note that betas and ld are just the chunked versions here
    betas, ld = args
    rows = list()
    for ti in range(len(chunk_window_starts)):
        window_start = chunk_window_starts[ti]
        window_stop = chunk_window_stops[ti]
        rows.append(
            genee_test(slice(window_start, window_stop), ld, betas, epsilon_effect)
        )
    cols = [
        ("test_q", np.float32),
        ("q_var", np.float32),
        ("pval", np.float32),
    ]
    df = pd.DataFrame(rows, columns=[c[0] for c in cols])
    for k, v in dict(cols).items():
        df[k] = df[k].astype(v)
    return df


def genee_test(gene, ld, betas, epsilon_effect):
    ld_g = ld[gene, gene]
    x = ld_g * epsilon_effect
    e_values = np.linalg.eigvals(x)

    betas_g = betas[gene]
    test_statistics = betas_g.T @ betas_g
    t_var = np.diag((ld_g * epsilon_effect) @ (ld_g * epsilon_effect)).sum()

    from chiscore import liu_sf

    (q, _, _, _) = liu_sf(
        test_statistics, e_values, np.ones(len(e_values)), np.zeros(len(e_values))
    )
    p_value_g = q
    p_value_g = np.real_if_close(p_value_g)
    if p_value_g <= 0.0:  # pragma: no cover
        p_value_g = 1e-20

    return test_statistics.squeeze().item(), t_var, p_value_g.squeeze().item()
