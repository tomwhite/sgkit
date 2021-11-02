# VCF Zarr specification

This document is a technical specification for VCF Zarr, an equivalent of VCF that uses Zarr storage.

This specification depends on definitions and terminology from [The Variant Call Format Specification, VCFv4.3 and BCFv2.2](https://samtools.github.io/hts-specs/VCFv4.3.pdf),
and [Zarr storage specification version 2](https://zarr.readthedocs.io/en/stable/spec/v2.html).

## Compatibility with VCF and BCF

Any VCF file that can be represented in BCF format can be represented in VCF Zarr.
Like BCF, VCF Zarr does not support the full set of VCF, since BCF is stricter than VCF around what is permitted in the header regarding field and contig information.

## Overall structure

A VCF Zarr store is composed of a top-level Zarr group that contains VCF header information stored as Zarr group attributes,
and VCF fields and sample information stored as Zarr arrays.

## VCF Zarr group attributes

The VCF Zarr store contains the following mandatory attributes:

| Key          | Value |
|--------------|-------|
| `fileformat` | `VCFZARRv0.1` |
| `header`     | The VCF header from `##fileformat` to `#CHROM` inclusive, stored as a single string. |
| `contigs`    | A list of strings of the contig IDs in the same order as specified in the header. |

TODO: would vcf_header or raw_header or raw_vcf_header be a better name?

The `contigs` attribute plays the same role as the dictionary of contigs in BCF, providing a way of encoding a contig (in the `CHROM` array)
with an integer offset into the `contigs` list.

## VCF Zarr arrays

Each VCF field is stored in a separate Zarr array. This specification only mandates the path, shape, dimension names, and dtype of each array. Other array metadata, including chunks, compression, layout order is not specified here.

### Zarr dtypes

This document uses the following shorthand notation to refer to Zarr data types (dtypes):

| Shorthand | Zarr dtypes    |
|-----------|----------------|
| `bool`    | `\|b1`         |
| `int`     | `<i1`, `<i2`, `<i4`, `<i8` or `>i1`, `>i2`, `>i4`, `>i8` |
| `int32`   | `<i4` or `>i4` |
| `int64`   | `<i8` or `>i8` |
| `float32` | `<f4` or `>f4` |
| `char`    | `\|S1`         |
| `str`     | `\|O`          |

This specification does not mandate a byte order for numeric types: little-endian (e.g. `<i4`) or big-endian (`>i4`) are both permitted.

The `str` dtype is used to represent [variable-length strings](https://zarr.readthedocs.io/en/stable/tutorial.html#string-arrays). In this case a Zarr array filter with and `id` of `vlen-utf8` must be specified for the array.

### Missing and fill values

Missing values indicate the value is absent, and fill values are used to pad variable length fields. The following are based on the values used in BCF. Note that the BCF specification refers to fill values as "END_OF_VECTOR" values.

| Dtype     | Missing    | Fill          |
|-----------|------------|---------------|
| `int32`   | 0x80000000 | 0x80000001    |
| `float32` | 0x7F800001 | 0x7F800002    |
| `char`    | "."        | ""            |
| `str`     | "."        | ""            |

There is no need for missing or fill values for the `bool` dtype, since Type=Flag fields can only appear in INFO fields, and they always have Number=0. Similarly, the only place where the `int` dtype is used that can have missing or fill values is in the `GT` field which defines custom values in this case. 

### Array dimension names

Following [Xarray conventions](http://xarray.pydata.org/en/stable/internals/zarr-encoding-spec.html), each Zarr array has an attribute `_ARRAY_DIMENSIONS`, which is a list of strings naming the dimensions.

The reserved dimension names and their sizes are listed in the following table, along with the corresponding VCF Number value, if applicable.

| Dimension name | Size                              | VCF Number |
|----------------|-----------------------------------|------------|
| `variants`     | The number of records in the VCF. | |
| `samples`      | The number of samples in the VCF. | |
| `ploidy`       | The maximum ploidy for any record in the VCF. | |
| `alleles`      | The maximum number of alleles for any record in the VCF. | R |
| `alt_alleles`  | The maximum number of alternate non-reference alleles for any record in the VCF. | A |
| `genotypes`    | The maximum number of genotypes for any record in the VCF. | G |

For fixed-size Number fields (e.g. Number=2) or unknown (Number=.), the dimension name can be any unique name that is not one of the reserved dimension names.

(*Implementation note: In general it is not possible to determine the size of some dimensions without a full pass through the VCF file being converted. Implementations may choose to fix the maximum size of some dimensions, although they should warn the user if this results in a lossy conversion.*)

### Fixed fields

The fixed VCF fields `CHROM`, `POS`, `ID`, `REF`, `QUAL`, and `FILTER` are stored as one-dimensional Zarr arrays at paths corresponding to the field name, each with shape `(variants)`, and dimension names `[variants]`. The `ALT` field is stored as a two-dimensional Zarr array at a path with name `ALT`, of shape `(variants, alt_alleles)`, and dimension names `[variants, alt_alleles]`. The dtype for each field is as follows:

| VCF field | Dtype     |
|-----------|-----------|
| `CHROM`   | `int`     |
| `POS`     | `int32` or `int64` |
| `ID`      | `str`     |
| `REF`     | `str`     |
| `ALT`     | `str`     |
| `QUAL`    | `float32` |
| `FILTER`  | `str`     |

Each value in the `CHROM` array is an integer offset into the `contigs` attribute list.

Usually `POS` uses `int32`, but `int64` can be used for cases when genome sizes exceed 32 bits.

Missing values are allowed for `ID`, `ALT`, and `FILTER` (all "."), and for `QUAL` (represented by 0x7F800001).

TODO: should we add a combined REF_ALT field that is of shape `(variants, alleles)`? 

### INFO fields

Each INFO field is stored as a two-dimensional Zarr array at a path with name `INFO_<field>`, of shape `(variants, <Number>)`, dimension names `[variants, <Number dimension name>]`, and with dtype determined by the following VCF Type field mapping:

| VCF Type  | Dtype     |
|-----------|-----------|
| Integer   | `int32`   |
| Float     | `float32` |
| Flag      | `bool`    |
| Character | `char`    |
| String    | `str`     |

Missing and fill values are encoded as described above.

### FORMAT fields

Each FORMAT field is stored as a three-dimensional Zarr array at a path with name `FORMAT_<field>`, of shape `(variants, samples, <Number>)`, dimension names `[variants, samples, <Number dimension name>]`, and with dtype determined by the same VCF Type field mapping for INFO fields, except for Flag, which is not permitted in FORMAT fields.

Missing and fill values are encoded as described above.

A **Genotype (GT) field** is stored as a three-dimensional Zarr array at a path with name `FORMAT_GT`, of shape `(variants, samples, ploidy)`, with an `int` dtype. Values encode the allele, with 0 for REF, 1 for the first alternate non-reference allele, and so on. A value of -1 indicates missing, and -2 indicates fill in mixed-ploidy datasets.

To indicate phasing, there is an accompanying Zarr array at a path with name `FORMAT_GT_phased`, of shape `(variants, samples)`, with dtype `bool`. Values are true if a call is phased, false if unphased (or not present).

### Sample information

Sample IDs are stored in a one-dimensional Zarr array at a path with name `sample_id`, of shape `(samples)`, dimension names `[samples]`, and with dtype `str`.
