from typing import Any, List, MutableMapping, Optional, Tuple, Union

import dask.array as da
import numpy as np
import tskit  # type: ignore
from numpy.typing import NDArray
from xarray.core.dataset import Dataset

from sgkit import DIM_VARIANT, create_genotype_call_dataset
from sgkit.io.vcf.utils import chunks as chunks_iterator
from sgkit.typing import PathType


def ts_to_zarr(
    ts: tskit.TreeSequence,
    output: Union[PathType, MutableMapping[str, bytes]],
    ploidy: Optional[int] = None,
    contig_id: str = "1",
    max_alt_alleles: int = 3,
    chunk_length: int = 10_000,
    chunk_width: int = 1_000,
    individual_names: Optional[List[str]] = None,
) -> None:
    """
    Save the specified tskit tree sequence to Zarr.
    """

    if ploidy is None:
        ploidy = 1

    if ts.num_samples % ploidy != 0:
        raise ValueError("Sample size must be divisible by ploidy")
    num_individuals = ts.num_samples // ploidy
    if individual_names is None:
        individual_names = [f"tsk_{j}" for j in range(num_individuals)]

    samples = ts.samples()
    tables = ts.dump_tables()

    # TODO: is there some way of finding max_alleles from ts?
    max_alleles = max_alt_alleles + 1

    offset = 0
    first_variants_chunk = True
    for variants_chunk in chunks_iterator(ts.variants(samples=samples), chunk_length):
        alleles = []
        genotypes = []
        for var in variants_chunk:
            alleles.append(var.alleles)
            genotypes.append(var.genotypes)
        padded_alleles = [
            list(site_alleles) + [""] * (max_alleles - len(site_alleles))
            for site_alleles in alleles
        ]
        alleles = np.array(padded_alleles).astype("O")
        genotypes = np.expand_dims(genotypes, axis=2)
        genotypes = genotypes.reshape(-1, num_individuals, ploidy)

        n_variants_in_chunk = len(genotypes)

        variant_id = np.full((n_variants_in_chunk), fill_value=".", dtype="O")
        variant_id_mask = variant_id == "."

        ds = create_genotype_call_dataset(
            variant_contig_names=[contig_id],
            variant_contig=np.zeros(n_variants_in_chunk, dtype="i1"),
            # TODO: should this be i8?
            variant_position=tables.sites.position[
                offset : offset + n_variants_in_chunk
            ].astype("i4"),
            variant_allele=alleles,
            sample_id=np.array(individual_names).astype("U"),
            call_genotype=genotypes,
            variant_id=variant_id,
        )
        ds["variant_id_mask"] = (
            [DIM_VARIANT],
            variant_id_mask,
        )

        if first_variants_chunk:
            # Enforce uniform chunks in the variants dimension
            # Also chunk in the samples direction
            chunk_sizes = dict(variants=chunk_length, samples=chunk_width)
            encoding = {}
            for var in ds.data_vars:
                var_chunks = tuple(
                    chunk_sizes.get(dim, size)
                    for (dim, size) in zip(ds[var].dims, ds[var].shape)
                )
                encoding[var] = dict(chunks=var_chunks)
            ds.to_zarr(output, mode="w", encoding=encoding)
            first_variants_chunk = False
        else:
            # Append along the variants dimension
            ds.to_zarr(output, append_dim=DIM_VARIANT)

        offset += n_variants_in_chunk


class TreeSequenceAlleleReader:
    # TODO: use varlen special dtype
    def __init__(
        self, ts: tskit.TreeSequence, shape: Tuple[int, int], dtype: Any = "O"
    ) -> None:
        self.ts = ts
        self.shape = shape
        self.dtype = dtype
        self.ndim = len(shape)

        _, self.max_alleles = shape
        self.tables = ts.dump_tables()

    def __getitem__(self, idx: Tuple[Any, ...]) -> NDArray[Any]:
        pos = self.tables.sites.position
        item_pos = pos[idx[0]]
        item_ts = self.ts.keep_intervals(np.array([[item_pos[0], item_pos[-1] + 1]]))

        alleles = []
        # we only care about alleles, so use empty set of samples
        for var in item_ts.variants(samples=[]):
            alleles.append(var.alleles)

        padded_alleles = [
            list(site_alleles) + [""] * (self.max_alleles - len(site_alleles))
            for site_alleles in alleles
        ]
        alleles = np.array(padded_alleles).astype(self.dtype)

        return alleles


class TreeSequenceGenotypeReader:
    def __init__(
        self, ts: tskit.TreeSequence, shape: Tuple[int, int, int], dtype: Any = np.int8
    ) -> None:
        self.ts = ts
        self.shape = shape
        self.dtype = dtype
        self.ndim = len(shape)

        _, _, self.ploidy = shape
        self.tables = ts.dump_tables()

    def __getitem__(self, idx: Tuple[Any, ...]) -> NDArray[Any]:
        ploidy = self.ploidy

        # subset ts by variant chunk
        pos = self.tables.sites.position
        item_pos = pos[idx[0]]
        item_ts = self.ts.keep_intervals(np.array([[item_pos[0], item_pos[-1] + 1]]))

        # find samples subset (need to adjust slicing taking ploidy into account)
        samples = self.ts.samples()
        samples_slice = slice(idx[1].start * ploidy, idx[1].stop * ploidy)
        item_samples = samples[samples_slice]

        # create an array of genotypes
        genotypes = np.empty((item_ts.num_sites, len(item_samples)), dtype=self.dtype)
        for i, var in enumerate(item_ts.variants(samples=item_samples)):
            genotypes[i] = var.genotypes

        # add ploidy dimension
        genotypes = genotypes.reshape(-1, genotypes.shape[1] // ploidy, ploidy)

        return genotypes


def read_ts(
    ts: tskit.TreeSequence,
    ploidy: Optional[int] = None,
    contig_id: str = "1",
    max_alt_alleles: int = 3,
    chunk_length: int = 10_000,
    chunk_width: int = 1_000,
    individual_names: Optional[List[str]] = None,
) -> Dataset:

    if ploidy is None:
        ploidy = 1

    if ts.num_samples % ploidy != 0:
        raise ValueError("Sample size must be divisible by ploidy")
    num_individuals = ts.num_samples // ploidy
    if individual_names is None:
        individual_names = [f"tsk_{j}" for j in range(num_individuals)]

    tables = ts.dump_tables()

    # TODO: is there some way of finding max_alleles from ts?
    max_alleles = max_alt_alleles + 1

    n_variants = ts.num_sites

    variant_allele = da.from_array(
        TreeSequenceAlleleReader(ts, (n_variants, max_alleles)),
        chunks=(chunk_length, -1),
        asarray=False,
    )

    call_genotype = da.from_array(
        TreeSequenceGenotypeReader(
            ts,
            (n_variants, num_individuals, ploidy),
        ),
        chunks=(chunk_length, chunk_width, -1),
        asarray=False,
    )

    variant_id = np.full((n_variants), fill_value=".", dtype="O")
    variant_id_mask = variant_id == "."

    ds = create_genotype_call_dataset(
        variant_contig_names=[contig_id],
        variant_contig=np.zeros(n_variants, dtype="i1"),
        # TODO: should this be i8?
        variant_position=tables.sites.position.astype("i4"),
        variant_allele=variant_allele,
        sample_id=np.array(individual_names).astype("U"),
        call_genotype=call_genotype,
        variant_id=variant_id,
    )
    ds["variant_id_mask"] = (
        [DIM_VARIANT],
        variant_id_mask,
    )

    return ds
