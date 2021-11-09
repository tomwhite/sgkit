# type: ignore
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pandas as pd
from genee_join import genee_ols
from scipy.sparse import coo_matrix


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

    # turn ld into an array
    ld = ld.to_numpy()
    ld = coo_matrix(ld)
    ld = pd.DataFrame(dict(i=ld.row, j=ld.col, val=ld.data))

    g = [list(range(start, stop)) for start, stop in zip(gene_start, gene_stop)]
    g = [
        list(zip(sublist, [i] * len(sublist), [len(sublist)] * len(sublist)))
        for i, sublist in enumerate(g)
    ]
    g = [item for sublist in g for item in sublist]  # flatten

    variant_groups = pd.DataFrame(
        g, columns=["variant_id", "variant_group", "variant_group_length"]
    )

    sumstats = pd.DataFrame(dict(beta=mydata.V4.to_numpy()))

    df = genee_ols(sumstats, ld, variant_groups)

    expected = pd.read_csv(data_dir / "result.csv", index_col=0)
    expected = expected.reset_index()

    npt.assert_allclose(df["test_q"], expected["test_q"])
    npt.assert_allclose(df["q_var"], expected["q_var"], rtol=0.005)
    npt.assert_allclose(
        df[df["pval"] > 1e-6]["pval"],
        expected[expected["pval"] > 1e-6]["pval"],
        rtol=0.04,
    )
