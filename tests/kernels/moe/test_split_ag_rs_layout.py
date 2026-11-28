# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.split_ag_rs import (
    _piece_sizes,
    _reserve_unsplit_workspace,
    _supports_layout,
    try_split,
)


@pytest.mark.parametrize(
    "sizes,width,expected",
    [
        ([15360, 15360, 17408, 17408], 5120, True),
        ([15360, 15360, 17408, 17408], 1280, False),
        ([15360, 15360, 17408, 17408], 2560, True),
        ([30720, 30720, 34816, 34816], 2560, True),
        ([8192] * 4, 5120, True),
        ([8191, 8192, 8192, 8192], 5120, False),
        ([20480, 16384, 16384, 12288], 5120, True),
        ([20481, 16384, 16384, 12287], 5120, True),
        ([1, 32768, 32768, 32768], 5120, False),
        ([0, 32768, 32768, 32768], 5120, False),
        ([32768, 32769], 5120, False),
    ],
)
def test_layout(sizes, width, expected):
    assert _supports_layout(sizes, width, 2) is expected


def test_restore_sizes_on_nested_error():
    original = [5, 6, 7, 8]
    first = [2, 3, 3, 4]
    metadata = SimpleNamespace(local_sizes=original)
    with _piece_sizes(metadata, first):
        with pytest.raises(ValueError), _piece_sizes(metadata, [1, 1, 1, 2]):
            raise ValueError
        assert metadata.local_sizes is first
    assert metadata.local_sizes is original


def test_no_forward_context():
    assert try_split(*(None,) * 10) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU allocator")
def test_whole_fallback_fits_after_split_warmup():
    from vllm.v1.worker.workspace import WorkspaceManager

    manager = WorkspaceManager(torch.device("cuda"))
    width = 5120
    full = [8192] * 4
    smaller = [8191] * 4
    dtype = torch.bfloat16
    assert _supports_layout(full, width, dtype.itemsize)
    assert not _supports_layout(smaller, width, dtype.itemsize)
    _reserve_unsplit_workspace(manager, sum(full), width, dtype)
    manager.get_simultaneous(((sum(full) // 2, width), dtype))
    manager.lock()
    (fallback,) = manager.get_simultaneous(((sum(smaller), width), dtype))
    assert fallback.shape == (sum(smaller), width)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.version.hip, reason="requires ROCm"
)
def test_pynccl_and_device_see_same_stream():
    from vllm.utils.torch_utils import current_stream

    from vllm.model_executor.layers.fused_moe.split_ag_rs import _make_comm_stream

    previous = current_stream()
    side = _make_comm_stream(previous.device)
    with torch.cuda.stream(side):
        assert current_stream().cuda_stream == side.cuda_stream
        assert torch.cuda.current_stream().cuda_stream == side.cuda_stream
    assert current_stream().cuda_stream == previous.cuda_stream


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.version.hip, reason="requires ROCm"
)
def test_queue_survives_allocator_reuse():
    import gc

    from vllm.model_executor.layers.fused_moe.split_ag_rs import _make_comm_stream

    compute = torch.cuda.current_stream()
    device = compute.device
    for value in (3, 7):
        side = _make_comm_stream(device)
        handle = side.cuda_stream
        with torch.cuda.stream(side):
            buf = torch.full((32768,), value, device=device)
        compute.wait_stream(side)
        buf.record_stream(compute)
        actual = buf.sum()
        del buf, side
        gc.collect()
        torch.cuda.empty_cache()
        assert actual.item() == 32768 * value
        assert _make_comm_stream(device).cuda_stream == handle


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.version.hip, reason="requires ROCm"
)
@pytest.mark.parametrize(
    ("uses_alias", "fail_second"), [(True, False), (False, False), (True, True)]
)
def test_split_workspace_lifetime(monkeypatch, uses_alias, fail_second):
    from vllm.model_executor.layers.fused_moe import split_ag_rs

    sizes = [5, 3, 7, 6]
    metadata = SimpleNamespace(local_sizes=sizes)
    monkeypatch.setattr(
        split_ag_rs,
        "get_forward_context",
        lambda: SimpleNamespace(dp_metadata=metadata),
    )
    device = torch.device("cuda", torch.cuda.current_device())
    hidden = torch.arange(20, dtype=torch.float32, device=device).reshape(5, 4)
    scratch = hidden.new_empty((sum(sizes), 4))
    seen_alias = []
    failed = False

    def prepare(x, weights, ids, *_args):
        assert x.shape[0] == metadata.local_sizes[0]
        return x, None, None, ids, weights

    def compute(*, a1q, output_alias, **_kwargs):
        nonlocal failed
        seen_alias.append(None if output_alias is None else tuple(output_alias.shape))
        dest = output_alias if uses_alias and output_alias is not None else scratch
        dest = dest[: a1q.shape[0]]
        torch.add(a1q, 3, out=dest)
        if fail_second and output_alias is None:
            failed = True
            raise RuntimeError("injected second expert failure")
        return dest

    def finalize(output, result, *_args):
        output.copy_(result)

    kernel = SimpleNamespace(
        _prepare=prepare,
        _fused_experts=compute,
        _finalize=finalize,
        fused_experts=SimpleNamespace(output_hidden_dim=lambda width: width),
    )
    route = hidden[:, :1].contiguous()
    side = split_ag_rs._make_comm_stream(device)
    args = (kernel, hidden, hidden, hidden, route, route, None, 1, None, 0, sizes, side)
    if fail_second:
        cleanup = []
        original_wait = torch.cuda.Stream.wait_stream

        def observe_wait(stream, other):
            if failed:
                cleanup.append((stream.cuda_stream, other.cuda_stream))
            return original_wait(stream, other)

        monkeypatch.setattr(torch.cuda.Stream, "wait_stream", observe_wait)
        with pytest.raises(RuntimeError, match="injected second expert failure"):
            split_ag_rs._split(*args)
        main = torch.cuda.current_stream().cuda_stream
        assert cleanup == [(side.cuda_stream, main), (main, side.cuda_stream)]
    else:
        output = split_ag_rs._split(*args)
        scratch.fill_(-100)
        torch.testing.assert_close(output, hidden + 3)
    assert seen_alias == [(sum(n // 2 for n in sizes), 4), None]
    assert metadata.local_sizes is sizes
