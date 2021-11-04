# type: ignore
import sys
from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd
from chiscore import liu_sf
from sklearn.mixture import GaussianMixture

sys.path.append("/Users/tom/workspace/sgkit")
from sgkit.model import create_genotype_call_dataset
from sgkit.stats.ld import map_windows_as_dataframe
from sgkit.utils import encode_array


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
    ld_g = ld[gene, gene]
    # TODO: prior weights
    # weight_matrix = np.diag(prior_weight[gene])
    # x = (ld_g * epsilon_effect) @ weight_matrix
    x = ld_g * epsilon_effect
    e_values = np.linalg.eigvals(x)

    betas_g = betas[gene]
    test_statsics = betas_g.T @ betas_g
    t_var = np.diag((ld_g * epsilon_effect) @ (ld_g * epsilon_effect)).sum()

    (q, _, _, _) = liu_sf(
        test_statsics, e_values, np.ones(len(e_values)), np.zeros(len(e_values))
    )
    p_value_g = q
    if p_value_g <= 0.0:
        p_value_g = 1e-20

    return test_statsics.squeeze().item(), t_var, p_value_g.squeeze().item()


def genee_ols_sg(ds, ld):
    betas = np.expand_dims(ds["beta"].values, 1)
    epsilon_effect = genee_EM(betas=betas)

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
            genee_test(
                slice(window_start, window_stop), ld, betas, epsilon_effect, None
            )
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


def to_sgkit(mydata):
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
    ds["beta"] = (["variants"], mydata.V4.to_numpy())
    return ds


if __name__ == "__main__":

    # pd.set_option("display.max_rows", 500)

    data_dir = Path(__file__).parent / "data" / "Simulated_Data_Example"

    mydata = pd.read_csv(data_dir / "mydata.csv", index_col=0)
    ld = pd.read_csv(data_dir / "ld.csv", index_col=0)

    # This was manually extracted using `write.csv(t(sapply(gene_list, unlist)), "gene_list.csv")` in R
    gene_list = "1:16,17:36,37:51,52:66,67:72,73:83,84:93,94:110,111:116,117:135,136:155,156:164,165:172,173:185,186:199,200:216,217:229,230:242,243:262,263:267,268:276,277:291,292:305,306:310,311:321,322:330,331:335,336:347,348:353,354:366,367:382,383:391,392:403,404:414,415:434,435:445,446:463,464:478,479:489,490:505,506:523,524:541,542:557,558:569,570:574,575:583,584:592,593:599,600:611,612:619,620:634,635:643,644:648,649:660,661:678,679:696,697:716,717:726,727:738,739:755,756:775,776:792,793:810,811:830,831:846,847:863,864:879,880:886,887:905,906:911,912:924,925:943,944:963,964:970,971:976,977:984,985:989,990:1007,1008:1016,1017:1035,1036:1052,1053:1068,1069:1082,1083:1087,1088:1107,1108:1113,1114:1130,1131:1140,1141:1157,1158:1166,1167:1185,1186:1197,1198:1210,1211:1226,1227:1235,1236:1243,1244:1260,1261:1276,1277:1295,1296:1300"
    gene_list = [[int(s) for s in ss.split(":")] for ss in gene_list.split(",")]
    gene_start, gene_stop = list(zip(*gene_list))
    gene_start = np.array(gene_start) - 1  # make 0-based
    gene_stop = np.array(gene_stop)

    ds = to_sgkit(mydata)
    print(ds)

    # turn ld into an array
    ld = ld.to_numpy()

    # genes are windows in this simple example
    ds["window_contig"] = (["windows"], np.full(len(gene_start), 0))
    ds["window_start"] = (["windows"], gene_start)
    ds["window_stop"] = (["windows"], gene_stop)
    print(ds)

    df = genee_ols_sg(ds, ld).compute()
    print(df)
