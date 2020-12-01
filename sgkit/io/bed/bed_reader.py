import dask.dataframe as dd
from dask.dataframe import DataFrame

from sgkit.typing import PathType


def read_bed(path: PathType) -> DataFrame:
    df = dd.read_csv(str(path), sep="\t", names=["chrom", "chromStart", "chromEnd"])
    return df
