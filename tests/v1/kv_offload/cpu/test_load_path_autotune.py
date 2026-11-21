# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the init-time autotuning of the DMA/Triton load cutoff."""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.kv_offload.cpu import load_path_autotune
from vllm.v1.kv_offload.cpu.load_path_autotune import (
    LoadPathAutotuneError,
    autotune_load_min_n,
)
from vllm.v1.kv_offload.cpu.swap_blocks_triton import MIN_N, THRESHOLD_BYTES

BATCH_SIZES = load_path_autotune._BATCH_SIZES
HOST = torch.zeros(1, dtype=torch.int8)
CPU = torch.device("cpu")
requires_cuda = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Triton swap fast path requires CUDA"
)


def _fake_ratios(monkeypatch: pytest.MonkeyPatch, ratios_by_size: dict) -> None:
    monkeypatch.setattr(
        load_path_autotune,
        "_measure_copy_size",
        lambda copy_size, *args: ratios_by_size[copy_size],
    )


@pytest.mark.parametrize(
    "ratios, default, expected",
    [
        # Triton takes over at N=64.
        ([3.0, 1.5, 0.75, 0.5, 0.3], None, 64),
        # Triton is ahead everywhere: never go below the smallest probed N.
        ([0.75, 0.6, 0.5, 0.5, 0.3], None, 16),
        # Triton never catches up.
        ([3.0, 1.5, 1.4, 1.3, 1.2], 16, None),
        # A win at N=32 doesn't count when N=64 loses again.
        ([3.0, 0.75, 1.3, 0.5, 0.3], 16, 128),
        # Within the margin the default decides: DMA when it never uses Triton,
        # Triton from its own min_n otherwise.
        ([3.0, 1.5, 0.98, 0.5, 0.3], None, 128),
        ([1.02, 0.98, 0.75, 0.5, 0.3], 16, 16),
        ([1.02, 0.98, 0.75, 0.5, 0.3], 64, 64),
    ],
)
def test_pick_min_n(ratios, default, expected) -> None:
    assert load_path_autotune._pick_min_n(BATCH_SIZES, ratios, default) == expected


def test_autotune_uses_measured_min_n(monkeypatch: pytest.MonkeyPatch) -> None:
    """The measured cutoff replaces the default, also for a copy size above
    the default Triton threshold."""
    page = 32 * 1024
    assert page >= THRESHOLD_BYTES
    _fake_ratios(monkeypatch, {page: [3.0, 1.5, 0.75, 0.5, 0.3]})
    assert autotune_load_min_n([page], 8192, HOST, CPU) == 64

    # Triton never wins: stay on DMA.
    _fake_ratios(monkeypatch, {4096: [3.0, 1.5, 1.4, 1.3, 1.2]})
    assert autotune_load_min_n([4096], 4096, HOST, CPU) is None


def test_autotune_reduces_mixed_copy_sizes(monkeypatch: pytest.MonkeyPatch) -> None:
    """One call carries every copy size of the handler: the largest min_n
    wins, and a size that never takes Triton keeps the handler on DMA."""
    ratios = {4096: [3.0, 1.5, 0.75, 0.5, 0.3], 16384: [3.0, 0.75, 0.6, 0.5, 0.3]}
    _fake_ratios(monkeypatch, ratios)
    assert autotune_load_min_n([4096, 16384], 8192, HOST, CPU) == 64

    ratios[16384] = [3.0, 1.5, 1.4, 1.3, 1.2]
    assert autotune_load_min_n([4096, 16384], 8192, HOST, CPU) is None


def test_autotune_failure_keeps_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args):
        raise RuntimeError("no batch memcpy")

    for measure in (boom, lambda *args: None):
        monkeypatch.setattr(load_path_autotune, "_measure_copy_size", measure)
        assert autotune_load_min_n([4096], 4096, HOST, CPU) == MIN_N
        assert autotune_load_min_n([32 * 1024], 8192, HOST, CPU) is None


def test_autotune_does_not_hide_an_undrained_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*args):
        raise LoadPathAutotuneError("copies still in flight")

    monkeypatch.setattr(load_path_autotune, "_measure_copy_size", boom)
    with pytest.raises(LoadPathAutotuneError):
        autotune_load_min_n([4096], 4096, HOST, CPU)


def _pinned_host(copy_size: int = 4096) -> torch.Tensor:
    host = torch.zeros((2 * max(BATCH_SIZES), copy_size), dtype=torch.int8)
    return host.pin_memory()


@requires_cuda
def test_measure_times_every_batch_size() -> None:
    host, device = _pinned_host(), torch.device("cuda:0")
    measure = load_path_autotune._measure_copy_size

    ratios = measure(4096, 4096, host, device, BATCH_SIZES)

    assert ratios is not None
    assert len(ratios) == len(BATCH_SIZES)
    assert all(r > 0 for r in ratios)
    # Too small a host region: not measured.
    assert measure(4096, 4096, host[:16], device, BATCH_SIZES) is None


@requires_cuda
def test_measure_reads_strided_host_rows() -> None:
    # A worker's view of the shared offload region: one chunk per row, with
    # the other tensors' (and workers') bytes in between.
    copy_size, row_stride = 4096, 3 * 4096
    rows = 2 * max(BATCH_SIZES)
    region = torch.zeros(rows * row_stride, dtype=torch.int8).pin_memory()
    host = torch.as_strided(region, (rows, copy_size), (row_stride, 1))
    assert not host.is_contiguous()

    ratios = load_path_autotune._measure_copy_size(
        copy_size, 4096, host, torch.device("cuda:0"), BATCH_SIZES
    )

    assert ratios is not None
    assert len(ratios) == len(BATCH_SIZES)


@requires_cuda
def test_measure_avoids_default_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    # On the default stream the DMA arm would time one cudaMemcpyAsync per
    # copy instead of cuMemcpyBatchAsync, which real loads use.
    streams = []

    def record(*args, **kwargs):
        streams.append(torch.cuda.current_stream())

    monkeypatch.setattr(load_path_autotune.ops, "swap_blocks_batch", record)
    monkeypatch.setattr(load_path_autotune, "swap_blocks_batch", record)

    assert torch.cuda.current_stream() == torch.cuda.default_stream()
    load_path_autotune._measure_copy_size(
        4096, 4096, _pinned_host(), torch.device("cuda:0"), BATCH_SIZES
    )

    assert streams
    assert all(s != torch.cuda.default_stream() for s in streams)


@requires_cuda
def test_measure_drains_the_stream_when_a_path_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A path that raises may have submitted copies first; they are waited
    for before the scratch buffer is released."""
    events: list[str] = []
    synchronize = torch.accelerator.synchronize

    def boom(*args, **kwargs):
        events.append("raise")
        raise RuntimeError("failed at index 1")

    def record_sync(device=None):
        events.append("sync")
        synchronize(device)

    monkeypatch.setattr(load_path_autotune.ops, "swap_blocks_batch", boom)
    monkeypatch.setattr(torch.accelerator, "synchronize", record_sync)

    with pytest.raises(RuntimeError, match="failed at index 1"):
        load_path_autotune._measure_copy_size(
            4096, 4096, _pinned_host(), torch.device("cuda:0"), BATCH_SIZES
        )
    assert events[-2:] == ["raise", "sync"]

    # The stream can't be drained: don't fall back as if nothing was pending.
    def failing_sync(device=None):
        if events[-1] == "raise":
            raise RuntimeError("device lost")
        events.append("sync")

    monkeypatch.setattr(torch.accelerator, "synchronize", failing_sync)
    with pytest.raises(LoadPathAutotuneError):
        autotune_load_min_n([4096], 4096, _pinned_host(), torch.device("cuda:0"))
