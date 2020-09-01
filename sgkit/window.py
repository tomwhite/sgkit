from typing import Any, Callable

import numpy as np


def get_windows(start: int, stop: int, size: int, step: int) -> Any:
    # Find the indexes for the start positions of all windows
    # TODO: not sure the stop parameter is right here
    window_starts = np.arange(start, stop - size + 1, step)
    window_stops = np.clip(window_starts + size, start, stop)
    window_lengths = window_stops - window_starts
    return window_starts, window_stops, window_lengths


def sizes_to_start_offsets(sizes: Any) -> Any:
    """Convert an array of sizes, to cumulative offsets, starting with 0"""
    return np.cumsum(np.insert(sizes, 0, 0, axis=0))


def get_chunked_windows(
    chunks: Any, length: int, window_starts: Any, window_stops: Any
) -> Any:
    """Find the window start positions relative to the start of the chunk they are in,
    and the number of windows in each chunk."""

    # Find the indexes for the start positions of all chunks
    chunk_starts = sizes_to_start_offsets(chunks)

    # Find which chunk each window falls in
    chunk_numbers = np.searchsorted(chunk_starts, window_starts, side="right") - 1

    # Find the start positions for each window relative to each chunk start
    rel_window_starts = window_starts - chunk_starts[chunk_numbers]

    # Find the number of windows in each chunk
    _, windows_per_chunk = np.unique(chunk_numbers, return_counts=True)

    return rel_window_starts, windows_per_chunk


def moving_statistic(
    values: Any,
    statistic: Callable[..., Any],
    size: int,
    step: int,
    dtype: int,
    **kwargs: Any,
) -> Any:
    chunks = values.chunks[0]
    length = values.shape[0]
    min_chunksize = np.min(chunks[:-1])  # ignore last chunk
    if min_chunksize < size:
        raise ValueError(
            f"Minimum chunk size ({min_chunksize}) must not be smaller than size ({size})."
        )

    window_starts, window_stops, window_lengths = get_windows(0, length, size, step)
    rel_window_starts, windows_per_chunk = get_chunked_windows(
        chunks, length, window_starts, window_stops
    )

    # Add depth for map_overlap
    depth = np.max(window_lengths)
    rel_window_starts = rel_window_starts + depth
    rel_window_stops = rel_window_starts + window_lengths

    chunk_offsets = sizes_to_start_offsets(windows_per_chunk)

    def blockwise_moving_stat(x: Any, block_info: Any = None) -> Any:
        if block_info is None or len(block_info) == 0:
            return np.array([])
        chunk_number = block_info[0]["chunk-location"][0]
        chunk_offset_start = chunk_offsets[chunk_number]
        chunk_offset_stop = chunk_offsets[chunk_number + 1]
        chunk_window_starts = rel_window_starts[chunk_offset_start:chunk_offset_stop]
        chunk_window_stops = rel_window_stops[chunk_offset_start:chunk_offset_stop]
        out = np.array(
            [
                statistic(x[i:j], **kwargs)
                for i, j in zip(chunk_window_starts, chunk_window_stops)
            ]
        )
        return out

    new_chunks = (tuple(windows_per_chunk),)
    return values.map_overlap(
        blockwise_moving_stat,
        dtype=dtype,
        chunks=new_chunks,
        depth=depth,
        boundary=0,
        trim=False,
    )
