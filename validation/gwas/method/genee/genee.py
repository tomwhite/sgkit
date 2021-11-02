# type: ignore
from pathlib import Path

import numpy as np
import pandas as pd
from chiscore import liu_sf
from sklearn.mixture import GaussianMixture


def genee_ols(betas_ols, ld, prior_weight, gene_list):
    epsilon_effect = genee_EM(betas=betas_ols)
    records = genee_loop(
        betas=betas_ols,
        ld=ld,
        epsilon_effect=epsilon_effect,
        prior_weight=prior_weight,
        gene_list=gene_list,
    )
    return pd.DataFrame.from_records(records, columns=["test_q", "q_var", "pval"])


def genee_EM(betas):
    # based on https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html#sphx-glr-auto-examples-mixture-plot-gmm-selection-py
    lowest_bic = np.infty
    for n_components in range(1, 10):
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


def genee_loop(betas, ld, epsilon_effect, prior_weight, gene_list):
    return [
        genee_test(gene, ld, betas, epsilon_effect, prior_weight) for gene in gene_list
    ]


def genee_test(gene, ld, betas, epsilon_effect, prior_weight):
    ld_g = ld.iloc[gene, gene].to_numpy()
    weight_matrix = np.diag(prior_weight[gene])
    x = (ld_g * epsilon_effect) @ weight_matrix
    e_values = np.linalg.eigvals(x)

    betas_g = betas[gene]
    test_statsics = betas_g.T @ weight_matrix @ betas_g
    t_var = np.diag((ld_g * epsilon_effect) @ (ld_g * epsilon_effect)).sum()

    (q, _, _, _) = liu_sf(
        test_statsics, e_values, np.ones(len(e_values)), np.zeros(len(e_values))
    )
    p_value_g = q
    if p_value_g <= 0.0:
        p_value_g = 1e-20

    return test_statsics.squeeze().item(), t_var, p_value_g.squeeze().item()


if __name__ == "__main__":

    # pd.set_option("display.max_rows", 500)

    data_dir = Path(__file__).parent / "data" / "Simulated_Data_Example"

    mydata = pd.read_csv(data_dir / "mydata.csv", index_col=0)
    ld = pd.read_csv(data_dir / "ld.csv", index_col=0)

    betas = np.expand_dims(mydata.V4.to_numpy(), 1)
    prior_weight = np.ones(len(betas))

    # This was manually extracted using `write.csv(t(sapply(gene_list, unlist)), "gene_list.csv")` in R
    gene_list = "1:16,17:36,37:51,52:66,67:72,73:83,84:93,94:110,111:116,117:135,136:155,156:164,165:172,173:185,186:199,200:216,217:229,230:242,243:262,263:267,268:276,277:291,292:305,306:310,311:321,322:330,331:335,336:347,348:353,354:366,367:382,383:391,392:403,404:414,415:434,435:445,446:463,464:478,479:489,490:505,506:523,524:541,542:557,558:569,570:574,575:583,584:592,593:599,600:611,612:619,620:634,635:643,644:648,649:660,661:678,679:696,697:716,717:726,727:738,739:755,756:775,776:792,793:810,811:830,831:846,847:863,864:879,880:886,887:905,906:911,912:924,925:943,944:963,964:970,971:976,977:984,985:989,990:1007,1008:1016,1017:1035,1036:1052,1053:1068,1069:1082,1083:1087,1088:1107,1108:1113,1114:1130,1131:1140,1141:1157,1158:1166,1167:1185,1186:1197,1198:1210,1211:1226,1227:1235,1236:1243,1244:1260,1261:1276,1277:1295,1296:1300"
    gene_list = gene_list.split(",")
    gene_list = [
        list(range(int(g.split(":")[0]) - 1, int(g.split(":")[1]))) for g in gene_list
    ]

    result = genee_ols(betas, ld, prior_weight, gene_list)
    print(result)

    expected = pd.read_csv(data_dir / "result.csv", index_col=0)
    expected = expected.reset_index()
    print(expected)

    ratio = expected.div(result).abs().mul(100).sub(100).abs()
    print(ratio)

    print(ratio.mean(axis=0))

    # test_q is identical to float precision
    # q_var is different by a constant factor (some systematic difference between R's mclust, and scikit-learn's GaussianMixture?)
    # pval is within 5% for 98 out of 100 genes (the other two had pval < 10^-8), due to use of Lui not Imhof

    print(sum(ratio["pval"] < 5))
    # print(result[ratio["pval"] > 5])
    # print(expected[ratio["pval"] > 5])
