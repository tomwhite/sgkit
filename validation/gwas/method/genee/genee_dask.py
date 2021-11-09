# type: ignore

import dask.dataframe as dd
import numpy as np
import pandas as pd
from chiscore import liu_sf
from scipy.sparse import coo_matrix
from sklearn.mixture import GaussianMixture


def genee_ols(sumstats, ld, variant_groups, gene_start, gene_stop):
    # compute epsilon_effect locally
    beta = np.expand_dims(sumstats["beta"].compute(), 1)
    epsilon_effect = genee_EM(beta)

    # Join LD to other data frames
    df = dd.merge(ld, variant_groups, left_on="i", right_index=True)
    df = dd.merge(df, sumstats, left_on="i", right_index=True)

    n_variants = len(sumstats)

    # TODO: some variant information is passed in via grp (beta), while some is in the closure (gene_start, gene_stop) - make more consistent

    def genee_test_group(grp):
        # get group index and range (in variant index space)
        variant_group = grp["variant_group"].iloc[0]
        start_row = gene_start[variant_group]
        stop_row = gene_stop[variant_group]

        # convert ld dataframe to an array for the range covered by this gene (group)
        data = grp["val"]
        row = grp["i"].to_numpy() - start_row
        col = grp["j"]
        shape = (gene_stop[variant_group] - gene_start[variant_group], n_variants)
        ld = coo_matrix((data, (row, col)), shape=shape)
        ld = ld.tocsc()[:, start_row:stop_row]
        ld = ld.toarray()

        # for beta, lots of values are repeated, so throw them away
        beta = grp["beta"].to_numpy().reshape(shape)[:, 0]

        test_q, q_var, pval = genee_test(ld, beta, epsilon_effect)

        return pd.DataFrame(
            [[test_q, q_var, pval]], columns=["test_q", "q_var", "pval"]
        )

    # Group
    grouped = df.groupby("variant_group")

    meta = pd.DataFrame(
        {
            "test_q": pd.Series(dtype="float"),
            "q_var": pd.Series(dtype="float"),
            "pval": pd.Series(dtype="float"),
        }
    )

    # and apply test to each group
    return grouped.apply(genee_test_group, meta=meta)


def genee_EM(betas):
    # based on https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html#sphx-glr-auto-examples-mixture-plot-gmm-selection-py
    lowest_bic = np.infty
    for n_components in range(1, 10):
        # setting reg_covar=0 makes results closer to R's mclust for real data, but makes it worse for simulated!
        gmm = GaussianMixture(n_components=n_components, random_state=0).fit(betas)
        bic = gmm.bic(betas)
        if bic < lowest_bic:
            lowest_bic = bic
            best_gmm = gmm

    if best_gmm.n_components == 1:
        epsilon_effect = best_gmm.covariances_.squeeze()[0]
    else:
        # TODO; handle case where first component composed more than 50% SNP
        epsilon_effect = best_gmm.covariances_.squeeze()[1]

    return epsilon_effect


def genee_test(ld_g, betas_g, epsilon_effect):
    x = ld_g * epsilon_effect
    e_values = np.linalg.eigvals(x)

    test_statsics = betas_g.T @ betas_g
    t_var = np.diag((ld_g * epsilon_effect) @ (ld_g * epsilon_effect)).sum()

    (q, _, _, _) = liu_sf(
        test_statsics, e_values, np.ones(len(e_values)), np.zeros(len(e_values))
    )
    p_value_g = q
    p_value_g = np.real_if_close(p_value_g)
    if p_value_g <= 0.0:
        p_value_g = 1e-20

    return test_statsics.squeeze().item(), t_var, p_value_g.squeeze().item()
