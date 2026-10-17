# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV checksums on CUDA: the Triton kernel matches the definition, and the
copy to host is ordered after the checksums for any reader. The definition and
helpers are those of test_kv_checksum.py (CPU tests)."""

import threading

import numpy as np
import pytest
import torch

from vllm.distributed.kv_transfer.kv_checksum import worker as checksum_worker
from vllm.distributed.kv_transfer.kv_checksum.checksum import (
    MAX_KV_HEADS_PER_LAYER,
    KVChecksumBlocks,
    KVChecksumScheduled,
)
from vllm.triton_utils import HAS_TRITON
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
from vllm.v1.worker.gpu.async_utils import (
    AsyncKVChecksumOutput,
    AsyncOutput,
    KVChecksumCopy,
)
from vllm.v1.worker.gpu.sample.output import SamplerOutput

from .test_kv_checksum import (
    BLOCKS,
    MAMBA_BLOCKS,
    NUM_BLOCKS,
    _cache,
    _check_requests_and_groups_of_one_step,
    _config,
    _group,
    _logical_kv,
    _mamba_cache,
    _mamba_name,
    _mamba_reference,
    _mamba_spec,
    _name,
    _reference,
    _results,
    _send,
    _spec,
    _split_layer,
    _split_reference,
    _split_states,
    _worker,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

needs_triton = pytest.mark.skipif(not HAS_TRITON, reason="needs Triton")


@needs_triton
@pytest.mark.parametrize("token_major", [False, True])
@pytest.mark.parametrize("kernel_split", [1, 2])
def test_triton_kernel_matches_definition(token_major, kernel_split):
    """On CUDA the sums come from the Triton kernel, in one launch for layers
    of one layout, which must give the definition's values for any layout,
    kernel block split and partial last block, including a layer with a large
    slice factor."""
    layers = {0: _logical_kv(heads=2, seed=1), 3: _logical_kv(heads=2, seed=2)}
    caches = {
        _name(number): _cache(
            kv, token_major=token_major, kernel_split=kernel_split
        ).cuda()
        for number, kv in layers.items()
    }
    worker = _worker(_config(_group(_spec(2), *caches)), caches)
    [launch] = worker._recv_plans[0].launches
    assert launch.use_triton and len(launch.views) == 2
    [group] = _send(worker).values()
    np.testing.assert_array_equal(group.values[:, 0], _reference(layers))


@needs_triton
def test_triton_kernel_reads_rows_of_several_requests_and_groups():
    """The kernel reads each group's rows from one shared copy, after a group
    with an odd number of rows."""
    _check_requests_and_groups_of_one_step("cuda")


@needs_triton
def test_triton_kernel_tiles_large_single_head_blocks():
    """One head of 32 tokens x 36 pieces is more than one 1024-piece tile and
    not a power of two (as MLA-like caches are), so the kernel's tiling and
    tail mask matter."""
    head_size, block_size = 72, 32  # K + V of 72 bf16 = 288 bytes = 36 pieces.
    logical = _logical_kv(heads=1, tokens=block_size, cell_bytes=4 * head_size)
    caches = {_name(0): _cache(logical).cuda()}
    spec = _spec(1, block_size=block_size, head_size=head_size)
    worker = _worker(_config(_group(spec)), caches, uniform_kv_heads=1)
    # 40 tokens: the second block is partial, the third empty.
    blocks = KVChecksumBlocks(num_tokens=40, groups={0: (0, [3, 1, 4])})
    [group] = _send(worker, blocks).values()
    np.testing.assert_array_equal(group.values[:, 0], _reference({0: logical}, blocks))


@needs_triton
def test_triton_kernel_accumulates_into_each_layers_column():
    """With several replica keys a group's sums form a matrix, each layer of a
    launch adding into its own column: the kernel must honor each layer's
    column and leave the other columns alone."""
    layers = {0: _logical_kv(heads=2, seed=3), 3: _logical_kv(heads=2, seed=4)}
    views = [
        checksum_worker._LayerView(
            checksum_worker._as_pieces(_cache(kv).cuda(), NUM_BLOCKS),
            first_slice=number * MAX_KV_HEADS_PER_LAYER,
            replica_key=(1, 0),
        )
        for number, kv in layers.items()
    ]
    launch = checksum_worker._FusedLayers(views, columns=[1, 3])
    block_ids = torch.tensor([3, 1, 4], device="cuda")
    first_tokens = torch.tensor([0, 16, 32], device="cuda")
    valid_tokens = torch.tensor([16, 16, 8], device="cuda")
    out = torch.zeros(3, 4, dtype=torch.int64, device="cuda")
    launch.accumulate(block_ids, first_tokens, valid_tokens, out)
    values = out.cpu().numpy().view(np.uint64)
    np.testing.assert_array_equal(values[:, 1], _reference({0: layers[0]}))
    np.testing.assert_array_equal(values[:, 3], _reference({3: layers[3]}))
    assert not values[:, 0].any() and not values[:, 2].any()


@needs_triton
def test_triton_kernel_launches_once_per_layout():
    """Layers whose caches differ in layout take separate launches, which add
    up to the definition."""
    layers = {0: _logical_kv(heads=2, seed=5), 1: _logical_kv(heads=2, seed=6)}
    caches = {
        _name(0): _cache(layers[0], token_major=False).cuda(),
        _name(1): _cache(layers[1], token_major=True).cuda(),
    }
    worker = _worker(_config(_group(_spec(2), *caches)), caches)
    assert len(worker._recv_plans[0].launches) == 2
    [group] = _send(worker).values()
    np.testing.assert_array_equal(group.values[:, 0], _reference(layers))


@needs_triton
def test_triton_kernel_sums_multi_chunk_mamba_states():
    """A Mamba state of 24K pieces (192 KiB) spans two kernel chunks, the
    last partial, whose sums add up atomically into one value."""
    shapes, dtypes = ((256, 128), (128, 128)), (torch.float32, torch.float32)
    covered = 4 * (256 + 128) * 128
    cache, pages = _mamba_cache(covered, covered + 64)
    worker = _worker(
        _config(_group(_mamba_spec(shapes, dtypes), _mamba_name(2))),
        {_mamba_name(2): cache.cuda()},
    )
    expected = _mamba_reference({2 * MAX_KV_HEADS_PER_LAYER: pages}, 3, covered)
    assert _send(worker, MAMBA_BLOCKS)[0].values[0, 0] == expected


@needs_triton
@pytest.mark.parametrize("dim_first", [True, False])
def test_triton_kernel_sums_split_mamba_rows(dim_first):
    """Rows of TP-split Mamba state are slices of 2-byte pieces, which the
    kernel reads unsigned: TP2 shards with replicated B and C sum to the
    definition over the global state."""
    states = _split_states()
    total = np.uint64(0)
    for rank in range(2):
        spec, cache = _split_layer(states, 2, rank, one_group=True, dim_first=dim_first)
        worker = _worker(
            _config(_group(spec, _mamba_name(2))),
            {_mamba_name(2): cache.cuda()},
            tp_rank=rank,
            tp_size=2,
        )
        assert all(launch.use_triton for launch in worker._recv_plans[0].launches)
        for group in _send(worker, MAMBA_BLOCKS).values():
            with np.errstate(over="ignore"):
                total += group.values.sum(dtype=np.uint64)
    assert total == _split_reference(states)


def test_copy_is_ordered_after_checksums_for_any_reader():
    """The model runner's copy on its output copy stream is ordered after the
    checksums, so another thread getting the output (as the async output
    thread does) reads complete values even while the main stream is busy."""
    logical = _logical_kv(heads=2)
    worker = _worker(_config(_group(_spec(2))), {_name(0): _cache(logical).cuda()})
    main_stream = torch.cuda.Stream()
    copy_stream = torch.cuda.Stream()
    torch.cuda.set_stream(main_stream)
    try:
        torch.cuda._sleep(200_000_000)
        worker.bind(KVChecksumScheduled(reqs_to_send={"req": BLOCKS}))
        pending = worker.post_forward(_results(), set())
        assert pending is not None
        async_output = AsyncKVChecksumOutput(
            EMPTY_MODEL_RUNNER_OUTPUT,
            KVChecksumCopy(pending, main_stream, copy_stream),
        )
    finally:
        torch.cuda.set_stream(torch.cuda.default_stream())

    outputs = []
    reader = threading.Thread(target=lambda: outputs.append(async_output.get_output()))
    reader.start()
    reader.join()
    assert EMPTY_MODEL_RUNNER_OUTPUT.kv_connector_output is None
    [record] = outputs[0].kv_connector_output.kv_checksums
    np.testing.assert_array_equal(
        record.checksums["req"][0].values[:, 0], _reference({0: logical})
    )


def test_sampled_output_carries_kv_checksums():
    """On the last PP rank, the checksums of the step's sends and finished
    loads, queued after the sampled tokens' copy, reach the sampled output."""
    logical = _logical_kv(heads=2)
    worker = _worker(_config(_group(_spec(2))), {_name(0): _cache(logical).cuda()})
    main_stream = torch.cuda.Stream()
    copy_stream = torch.cuda.Stream()
    torch.cuda.set_stream(main_stream)
    try:
        torch.cuda._sleep(200_000_000)
        worker.bind(
            KVChecksumScheduled(
                reqs_to_send={"send": BLOCKS}, reqs_to_recv={"recv": BLOCKS}
            )
        )
        num_sampled = torch.ones(1, dtype=torch.int32, device="cuda")
        sampled_output = AsyncOutput(
            ModelRunnerOutput(req_ids=["send"], req_id_to_index={"send": 0}),
            SamplerOutput(
                sampled_token_ids=torch.full((1, 1), 7, device="cuda"),
                logprobs_tensors=None,
                num_nans=None,
                num_sampled=num_sampled,
                num_rejected=torch.zeros_like(num_sampled),
            ),
            num_sampled,
            main_stream=main_stream,
            copy_stream=copy_stream,
            check_ep_fault=False,
            pending_aux_output=None,
        )
        pending = worker.post_forward(_results("recv"), set())
        assert pending is not None
        async_output = AsyncKVChecksumOutput(
            sampled_output, KVChecksumCopy(pending, main_stream, copy_stream)
        )
    finally:
        torch.cuda.set_stream(torch.cuda.default_stream())

    output = async_output.get_output()
    assert output.sampled_token_ids == [[7]]
    [record] = output.kv_connector_output.kv_checksums
    expected = _reference({0: logical})
    for req_id in ("send", "recv"):
        np.testing.assert_array_equal(
            record.checksums[req_id][0].values[:, 0], expected
        )
