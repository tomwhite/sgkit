# type: ignore
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from chiscore import liu_sf
from sklearn.mixture import GaussianMixture

sys.path.append("/Users/tom/workspace/sgkit")
import sgkit as sg
from sgkit.model import create_genotype_call_dataset
from sgkit.utils import encode_array
from sgkit.window import window_by_gene


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


def to_sgkit(mydata):
    print(mydata.V1.to_numpy())
    print(mydata.V2.to_numpy())
    print(mydata.V3.to_numpy())

    variant_contig, variant_contig_names = encode_array(mydata.V1.to_numpy())
    variant_contig = variant_contig.astype("int16")
    variant_contig_names = list(variant_contig_names)
    variant_position = mydata.V3.to_numpy()
    variant_id = mydata.V2.to_numpy()
    variant_allele = np.array([["A"]] * len(variant_contig), dtype="S1")  # not used
    sample_id = ["SAMPLE1"]
    ds = create_genotype_call_dataset(
        variant_contig_names=variant_contig_names,
        variant_contig=variant_contig,
        variant_position=variant_position,
        variant_allele=variant_allele,
        sample_id=sample_id,
        variant_id=variant_id,
    )
    return ds


if __name__ == "__main__":

    # pd.set_option("display.max_rows", 500)

    data_dir = Path(__file__).parent / "data" / "Real_Data_Example"

    mydata = pd.read_csv(data_dir / "mydata.csv", index_col=0)
    ld = pd.read_csv(data_dir / "ld.csv", index_col=0)
    glist_hg19 = pd.read_csv(
        Path(__file__).parent / "data" / "glist.hg19.csv", index_col=0
    )

    with open(data_dir / "gene_list.txt") as f:
        gene_list = [
            [int(v) for v in line.rstrip().split(",")] for line in f.readlines()
        ]
    print(gene_list)

    print(glist_hg19)

    print(mydata)
    ds = to_sgkit(mydata)

    ds["gene_contig_name"] = (["genes"], glist_hg19.V1.to_numpy())
    ds["gene_start"] = (["genes"], glist_hg19.V2.to_numpy())
    ds["gene_stop"] = (["genes"], glist_hg19.V3.to_numpy())
    ds["gene_id"] = (["genes"], glist_hg19.V4.to_numpy())

    print(ds)

    ds2 = window_by_gene(ds)
    print(ds2)

    window_start = ds2["window_start"].values.tolist()
    window_stop = ds2["window_stop"].values.tolist()
    print(window_start)
    print(window_stop)

    gene_list_starts = [g[0] - 1 for g in gene_list]
    gene_list_stops = [g[-1] for g in gene_list]
    print(gene_list_starts)
    print(gene_list_stops)

    print(window_start == gene_list_starts)
    print(window_stop == gene_list_stops)
