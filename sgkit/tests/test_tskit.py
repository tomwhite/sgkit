import msprime  # type: ignore
import pytest
import tskit  # type: ignore
import xarray as xr
from xarray import Dataset

import sgkit as sg
from sgkit.io.vcf import vcf_to_zarr

from .tskit_zarr import read_ts, ts_to_zarr


def assert_identical(ds1: Dataset, ds2: Dataset) -> None:
    """Assert two Datasets are identical, including dtypes for all variables, except strings."""
    xr.testing.assert_identical(ds1, ds2)
    # check all types except strings (since they may differ e.g. "O" vs "U")
    assert all(
        [
            ds1[v].dtype == ds2[v].dtype
            for v in ds1.data_vars
            if ds1[v].dtype.kind not in ("O", "S", "U")
        ]
    )


def simulate_ts(
    sample_size: int,
    length: int = 100,
    mutation_rate: float = 0.05,
    random_seed: int = 42,
    ploidy: int = 1,
) -> tskit.TreeSequence:
    """
    Simulate some data using msprime with recombination and mutation and
    return the resulting tskit TreeSequence.
    """
    ancestry_ts = msprime.sim_ancestry(
        sample_size,
        ploidy=ploidy,
        recombination_rate=0.01,
        sequence_length=length,
        random_seed=random_seed,
    )
    # Make sure we generate some data that's not all from the same tree
    assert ancestry_ts.num_trees > 1
    return msprime.sim_mutations(
        ancestry_ts, rate=mutation_rate, random_seed=random_seed
    )


@pytest.mark.parametrize(
    "sample_size, ploidy",
    [(3, 1), (3, 2)],
)
@pytest.mark.parametrize(
    "chunk_length, chunk_width",
    [(3, 100)],
)
def test_roundtrip(tmp_path, sample_size, ploidy, chunk_length, chunk_width):
    output_vcf = tmp_path.joinpath("vcf").as_posix()
    output_zarr_from_vcf = tmp_path.joinpath("vcf.zarr").as_posix()
    output_zarr_from_ts = tmp_path.joinpath("ts.zarr").as_posix()

    ts = simulate_ts(sample_size, ploidy=ploidy)

    # write ts as vcf
    with open(output_vcf, "w") as vcf_file:
        ts.write_vcf(vcf_file)
    print(output_vcf)

    # convert vcf to zarr
    vcf_to_zarr(
        output_vcf,
        output_zarr_from_vcf,
        ploidy=ploidy,
        chunk_length=chunk_length,
        chunk_width=chunk_width,
    )

    # write ts as zarr directly
    ts_to_zarr(
        ts,
        output_zarr_from_ts,
        ploidy=ploidy,
        chunk_length=chunk_length,
        chunk_width=chunk_width,
    )

    # test they are the same
    vcf_ds = sg.load_dataset(str(output_zarr_from_vcf))
    ts_ds = sg.load_dataset(str(output_zarr_from_ts))

    vcf_ds = vcf_ds.drop_vars("call_genotype_phased")  # not included in ts_to_zarr

    assert_identical(vcf_ds, ts_ds)


@pytest.mark.parametrize(
    "sample_size, ploidy",
    [(3, 1), (3, 2)],
)
@pytest.mark.parametrize(
    "chunk_length, chunk_width",
    [(3, -1), (3, 2)],
)
# TODO: remove this and use special object dtype with varlen metadata
@pytest.mark.filterwarnings("ignore::xarray.coding.variables.SerializationWarning")
def test_read_ts(tmp_path, sample_size, ploidy, chunk_length, chunk_width):
    output_vcf = tmp_path.joinpath("vcf").as_posix()
    output_zarr_from_vcf = tmp_path.joinpath("vcf.zarr").as_posix()
    output_zarr_from_ts = tmp_path.joinpath("ts.zarr").as_posix()

    ts = simulate_ts(sample_size, ploidy=ploidy)

    # write ts as vcf
    with open(output_vcf, "w") as vcf_file:
        ts.write_vcf(vcf_file)
    print(output_vcf)

    # convert vcf to zarr
    vcf_to_zarr(
        output_vcf,
        output_zarr_from_vcf,
        ploidy=ploidy,
        chunk_length=chunk_length,
        chunk_width=chunk_width,
    )

    # convert ts to xarray, then as zarr
    ds = read_ts(
        ts,
        ploidy=ploidy,
        chunk_length=chunk_length,
        chunk_width=chunk_width,
    )
    # print(ds.load())
    sg.save_dataset(ds, output_zarr_from_ts)

    # test they are the same
    vcf_ds = sg.load_dataset(str(output_zarr_from_vcf))
    ts_ds = sg.load_dataset(str(output_zarr_from_ts))

    vcf_ds = vcf_ds.drop_vars("call_genotype_phased")  # not included in ts_to_zarr

    assert_identical(vcf_ds, ts_ds)
