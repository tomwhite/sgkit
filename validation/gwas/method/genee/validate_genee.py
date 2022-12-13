# type: ignore
import sys
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

sys.path.append("/Users/tom/workspace/sgkit")

from sgkit import genee, window_by_interval
from sgkit.model import create_genotype_call_dataset
from sgkit.utils import encode_array


def to_sgkit(mydata):
    variant_contig, variant_contig_names = encode_array(mydata.V1.to_numpy())
    variant_contig = variant_contig.astype("int16")
    variant_contig_names = [str(contig) for contig in variant_contig_names]
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


def test_simulated_data():

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

    # turn ld into an array
    ld = ld.to_numpy()

    # genes are windows in this simple example
    ds["window_contig"] = (["windows"], np.full(len(gene_start), 0))
    ds["window_start"] = (["windows"], gene_start)
    ds["window_stop"] = (["windows"], gene_stop)

    df = genee(ds, ld).compute()

    expected = pd.read_csv(data_dir / "result.csv", index_col=0)
    expected = expected.reset_index()

    npt.assert_allclose(df["test_q"], expected["test_q"])
    npt.assert_allclose(df["q_var"], expected["q_var"], rtol=0.005)
    npt.assert_allclose(
        df[df["pval"] > 1e-6]["pval"],
        expected[expected["pval"] > 1e-6]["pval"],
        rtol=0.04,
    )


def test_real_data_gene_list():
    data_dir = Path(__file__).parent / "data" / "Real_Data_Example"

    mydata = pd.read_csv(data_dir / "mydata.csv", index_col=0)
    glist_hg19 = pd.read_csv(
        Path(__file__).parent / "data" / "glist.hg19.csv", index_col=0
    )

    with open(data_dir / "gene_list.txt") as f:
        gene_list = [
            [int(v) for v in line.rstrip().split(",")] for line in f.readlines()
        ]

    ds = to_sgkit(mydata)

    padding = 50000
    ds["gene_contig_name"] = (["genes"], glist_hg19.V1.to_numpy())
    ds["gene_start"] = (["genes"], glist_hg19.V2.to_numpy() - padding)
    ds["gene_stop"] = (["genes"], glist_hg19.V3.to_numpy() + padding)
    ds["gene_id"] = (["genes"], glist_hg19.V4.to_numpy())

    ds2 = window_by_interval(
        ds,
        interval_contig_name="gene_contig_name",
        interval_start="gene_start",
        interval_stop="gene_stop",
    )

    window_start = ds2["window_start"].values.tolist()
    window_stop = ds2["window_stop"].values.tolist()

    gene_list_starts = [g[0] - 1 for g in gene_list]
    gene_list_stops = [g[-1] for g in gene_list]

    assert window_start == gene_list_starts
    assert window_stop == gene_list_stops


@pytest.mark.filterwarnings("ignore::numpy.ComplexWarning")
def test_real_data():
    data_dir = Path(__file__).parent / "data" / "Real_Data_Example"

    mydata = pd.read_csv(data_dir / "mydata.csv", index_col=0)
    ld = pd.read_csv(data_dir / "ld.csv", index_col=0)
    glist_hg19_sorted = pd.read_csv(
        Path(__file__).parent / "data" / "glist.hg19.sorted.csv", index_col=0
    )

    ds = to_sgkit(mydata)

    # turn ld into an array
    ld = ld.to_numpy()

    padding = 50000
    ds["gene_contig_name"] = (["genes"], glist_hg19_sorted.V1.to_numpy())
    ds["gene_start"] = (["genes"], glist_hg19_sorted.V2.to_numpy() - padding)
    ds["gene_stop"] = (["genes"], glist_hg19_sorted.V3.to_numpy() + padding)
    ds["gene_id"] = (["genes"], glist_hg19_sorted.V4.to_numpy())

    ds2 = window_by_interval(
        ds,
        interval_contig_name="gene_contig_name",
        interval_start="gene_start",
        interval_stop="gene_stop",
    )

    df = genee(ds2, ld, reg_covar=0).compute()

    expected = pd.read_csv(data_dir / "result.csv", index_col=0)
    expected = expected.reset_index()

    npt.assert_allclose(df["test_q"], expected["test_q"])
    # The following only pass if reg_covar=0
    npt.assert_allclose(df["q_var"], expected["q_var"], rtol=0.07)
    npt.assert_allclose(
        -np.log10(df["pval"]), -np.log10(expected["pval"]), atol=1.0
    )  # 1 order of mag for p-val
