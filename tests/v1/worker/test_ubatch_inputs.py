# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.v1.worker.ubatch_inputs import (
    slice_lookback_token_ids,
    update_captured_lookback,
)


@pytest.mark.parametrize(
    "request_slice,token_slice,expected",
    [
        (slice(0, 1), slice(0, 2), [[9, 8, 7], [-1, -1, -1]]),
        (slice(0, 2), slice(2, 5), [[11, 10, 9], [29, 28, 27], [-1, -1, -1]]),
        (slice(1, 2), slice(3, 5), [[29, 28, 27], [-1, -1, -1]]),
        (slice(0, 1), slice(1, 3), [[10, 9, 8], [-1, -1, -1]]),
    ],
)
def test_lookbacks_follow_microbatch_chunk_start(request_slice, token_slice, expected):
    """A split request reads previous batch tokens before its older history."""
    actual = slice_lookback_token_ids(
        torch.tensor([[9, 8, 7], [29, 28, 27]], dtype=torch.int32),
        torch.tensor([10, 11, 12, 30, 31]),
        torch.tensor([0, 3, 5]),
        request_slice,
        token_slice,
    )
    assert actual.tolist() == expected


def test_lookback_padding_does_not_read_the_last_live_request():
    actual = slice_lookback_token_ids(
        torch.tensor([[5, -1, -1]]),
        torch.tensor([6, 0, 0]),
        torch.tensor([0, 1]),
        slice(0, 3),
        slice(0, 3),
    )
    assert actual.tolist() == [[5, -1, -1], [-1, -1, -1], [-1, -1, -1]]


def test_replay_refreshes_history_and_clears_unused_request_rows():
    captured = torch.tensor([[1, 2, 3], [4, 5, 6]])
    address = captured.data_ptr()
    current = slice_lookback_token_ids(
        torch.tensor([[7, 8, 9]]),
        torch.tensor([10, 11]),
        torch.tensor([0, 2]),
        slice(0, 1),
        slice(0, 2),
    )
    update_captured_lookback(captured, current)
    assert captured.data_ptr() == address
    assert captured.tolist() == [[7, 8, 9], [-1, -1, -1]]


def test_replay_rejects_a_changed_history_signature():
    with pytest.raises(ValueError, match="graph signature"):
        update_captured_lookback(torch.empty(2, 3), torch.empty(3, 3))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("capture", [False, True])
def test_wrapper_preserves_split_request_history_across_replay(capture):
    """Actual microbatch threads and graph replay must receive refreshed history."""
    from vllm.config import CUDAGraphMode, ParallelConfig, VllmConfig
    from vllm.forward_context import (
        BatchDescriptor,
        DPMetadata,
        create_forward_context,
        override_forward_context,
    )
    from vllm.v1.worker.gpu_ubatch_wrapper import UBatchWrapper
    from vllm.v1.worker.ubatch_utils import UBatchSlice

    config = VllmConfig(
        parallel_config=ParallelConfig(data_parallel_size=2, is_moe_model=True)
    )
    # The callable below has no attention or MoE layers to configure.
    config.parallel_config.ubatch_size = 2
    config.parallel_config.all2all_backend = "deepep_high_throughput"

    def model(
        *, input_ids, positions, intermediate_tensors, inputs_embeds, lookback_token_ids
    ):
        return lookback_token_ids.clone()

    mode = CUDAGraphMode.FULL if capture else CUDAGraphMode.NONE
    wrapper = UBatchWrapper(model, config, mode, torch.device("cuda:0"))
    ids = torch.arange(10, 18, device="cuda")
    positions = torch.arange(8, device="cuda")
    history = torch.tensor([[9, 8, 7], [29, 28, 27]], device="cuda")
    starts = torch.tensor([0, 6, 8], device="cuda", dtype=torch.int32)
    slices = [
        UBatchSlice(slice(0, 1), slice(0, 4)),
        UBatchSlice(slice(0, 2), slice(4, 8)),
    ]
    compute_stream = torch.cuda.Stream()

    def run(runtime_mode):
        context = create_forward_context(
            None,
            config,
            dp_metadata=DPMetadata(torch.tensor([8, 8])),
            cudagraph_runtime_mode=runtime_mode,
            batch_descriptor=BatchDescriptor(num_tokens=8),
            ubatch_slices=slices,
        )
        compute_stream.wait_stream(torch.cuda.current_stream())
        with override_forward_context(context), torch.cuda.stream(compute_stream):
            output = wrapper(
                input_ids=ids,
                positions=positions,
                intermediate_tensors=None,
                inputs_embeds=None,
                lookback_token_ids=history,
                lookback_query_start_loc=starts,
            )
        torch.cuda.current_stream().wait_stream(compute_stream)
        return output

    # Initialize thread-local CUDA handles before capture.
    run(CUDAGraphMode.NONE)
    if capture:
        run(mode)
    for change in (0, 100, 200):
        ids.copy_(torch.arange(10, 18, device="cuda") + change)
        history.copy_(torch.tensor([[9, 8, 7], [29, 28, 27]], device="cuda") + change)
        if change == 200:
            starts.copy_(torch.tensor([0, 8, 8], device="cuda"))
            slices[1] = UBatchSlice(slice(0, 1), slice(4, 8))
        actual = run(mode)
        expected = torch.full((8, 3), -1, device="cuda", dtype=torch.int64)
        expected[0] = history[0]
        expected[4] = ids[torch.tensor([3, 2, 1], device="cuda")]
        if change != 200:
            expected[5] = history[1]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    wrapper.clear_graphs()
