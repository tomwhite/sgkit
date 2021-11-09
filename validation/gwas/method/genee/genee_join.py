# type: ignore

import numpy as np
import pandas as pd
from chiscore import liu_sf
from scipy.sparse import coo_matrix
from sklearn.mixture import GaussianMixture


def genee_ols(sumstats, ld, variant_groups):
    beta = np.expand_dims(sumstats["beta"], 1)
    epsilon_effect = genee_EM(beta)

    # Join LD to other data frames
    df = pd.merge(ld, variant_groups, left_on="i", right_index=True)
    df = pd.merge(df, sumstats, left_on="i", right_index=True)

    def genee_test_group(grp):
        shape = (grp["variant_group_length"].iloc[0], 1700)
        start_row = np.min(grp["i"].to_numpy())  # TODO: should pass in offsets
        stop_row = np.max(grp["i"].to_numpy()) + 1  # TODO: should pass in offsets
        row = grp["i"].to_numpy() - start_row
        ld = coo_matrix((grp["val"], (row, grp["j"])), shape=shape)
        ld = ld.tocsc()[:, start_row:stop_row]
        ld = ld.toarray()
        beta = (
            grp["beta"].to_numpy().reshape(shape)[:, 0]
        )  # lots of values are repeated, so throw them away

        test_q, q_var, pval = genee_test(ld, beta, epsilon_effect)

        return pd.DataFrame(
            [[test_q, q_var, pval]], columns=["test_q", "q_var", "pval"]
        )

    # Group
    grouped = df.groupby("variant_group")

    # and apply test to each group
    return grouped.apply(genee_test_group)


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
