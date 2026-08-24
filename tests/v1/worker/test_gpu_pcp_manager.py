# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.pcp_manager import PCPManager

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Requires CUDA"
)


def _make_batch(
    device: torch.device,
    num_scheduled_tokens: np.ndarray,
    num_computed_tokens: np.ndarray,
    prefill_lens: np.ndarray,
    num_draft_tokens_per_req: np.ndarray,
) -> InputBatch:
    num_reqs = len(num_scheduled_tokens)
    query_start_loc_np = np.zeros(num_reqs + 1, dtype=np.int32)
    np.cumsum(num_scheduled_tokens, out=query_start_loc_np[1:])
    num_tokens = int(query_start_loc_np[-1])
    buffers = InputBuffers(num_reqs, num_tokens, device)
    batch = InputBatch.make_dummy(num_reqs, num_tokens, buffers)
    batch.num_scheduled_tokens = num_scheduled_tokens
    batch.num_draft_tokens = int(num_draft_tokens_per_req.sum())
    batch.num_draft_tokens_per_req = num_draft_tokens_per_req
    batch.query_start_loc_np = query_start_loc_np
    batch.query_start_loc.copy_(torch.from_numpy(query_start_loc_np).to(device))
    batch.num_computed_tokens_np = num_computed_tokens
    batch.prefill_len_np = prefill_lens
    batch.num_computed_prefill_tokens_np = np.minimum(num_computed_tokens, prefill_lens)
    batch.is_prefilling_np = num_computed_tokens < prefill_lens
    batch.has_prefill = bool(batch.is_prefilling_np.any())
    batch.seq_lens.copy_(
        torch.from_numpy(num_computed_tokens + num_scheduled_tokens).to(device)
    )
    positions = np.concatenate(
        [
            np.arange(computed, computed + scheduled, dtype=np.int64)
            for computed, scheduled in zip(num_computed_tokens, num_scheduled_tokens)
        ]
    )
    batch.positions.copy_(torch.from_numpy(positions).to(device))
    batch.input_ids.copy_(torch.arange(1000, 1000 + num_tokens, device=device))
    batch.is_padding.fill_(False)
    return batch


def _make_manager(device: torch.device) -> PCPManager:
    return PCPManager(
        pcp_world_size=2,
        pcp_rank=0,
        device=device,
        max_num_reqs=2,
        max_num_tokens=16,
    )


def test_local_batch_is_scoped_to_its_global_batch():
    manager = object.__new__(PCPManager)
    global_batch = object()
    local_batch = object()
    manager._global_batch = global_batch
    manager._local_batch = local_batch

    assert manager.local_batch_for(global_batch) is local_batch  # type: ignore[arg-type]
    assert manager.local_batch_for(object()) is None  # type: ignore[arg-type]


def test_draft_ids_reuse_consumed_local_input_buffer():
    manager = object.__new__(PCPManager)
    manager._local_gather_idx = torch.tensor([2, 0])
    local_batch = SimpleNamespace(input_ids=torch.full((2,), -1))
    manager._local_batch = local_batch

    localized = manager.localize_input_ids_for_draft(
        torch.tensor([10, 11, 12]),
        local_batch,  # type: ignore[arg-type]
    )

    assert localized.data_ptr() == local_batch.input_ids.data_ptr()
    assert local_batch.input_ids.tolist() == [12, 10]


@requires_cuda
def test_pcp_partitions_mtp_decode_batch():
    device = torch.device("cuda")
    global_batch = _make_batch(
        device,
        num_scheduled_tokens=np.array([4, 2], dtype=np.int32),
        num_computed_tokens=np.array([10, 20], dtype=np.int32),
        prefill_lens=np.array([5, 5], dtype=np.int32),
        num_draft_tokens_per_req=np.array([3, 1], dtype=np.int32),
    )
    # Simulate the GPU request-state cursor advancing beyond the async CPU copy.
    global_batch.positions.copy_(torch.tensor([30, 31, 32, 33, 40, 41]))

    local_batch = _make_manager(device).partition_batch(global_batch)

    assert local_batch.num_draft_tokens == 0
    assert local_batch.num_draft_tokens_per_req is None
    assert local_batch.cu_num_logits_np.tolist() == [0, 1, 2]
    assert local_batch.input_ids.tolist() == list(range(1000, 1006))
    assert local_batch.logits_indices.tolist() == [3, 5]
    assert local_batch.expanded_idx_mapping.tolist() == [0, 1]
    assert local_batch.expanded_local_pos.tolist() == [0, 0]
    assert local_batch.positions.tolist() == [30, 31, 32, 33, 40, 41]
    assert local_batch.seq_lens.tolist() == [34, 42]


@requires_cuda
def test_pcp_partitions_mixed_prefill_and_mtp_decode_batch():
    device = torch.device("cuda")
    global_batch = _make_batch(
        device,
        num_scheduled_tokens=np.array([4, 8], dtype=np.int32),
        num_computed_tokens=np.array([10, 0], dtype=np.int32),
        prefill_lens=np.array([5, 8], dtype=np.int32),
        num_draft_tokens_per_req=np.array([3, 0], dtype=np.int32),
    )

    local_batch = _make_manager(device).partition_batch(global_batch)

    assert local_batch.num_scheduled_tokens.tolist() == [4, 2, 2]
    assert local_batch.num_draft_tokens == 0
    assert local_batch.num_draft_tokens_per_req is None
    assert local_batch.cu_num_logits_np.tolist() == [0, 1, 2, 3]
    assert local_batch.input_ids.tolist() == [
        1000,
        1001,
        1002,
        1003,
        1010,
        1011,
        1004,
        1005,
    ]
    assert local_batch.logits_indices.tolist() == [3, 5, 7]
    assert local_batch.expanded_idx_mapping.tolist() == [0, 1, 1]
    assert local_batch.expanded_local_pos.tolist() == [0, 0, 0]


def _make_validation_config(
    *, multi_module_mtp: bool, cudagraph_mode: CUDAGraphMode
) -> SimpleNamespace:
    speculative_config = SimpleNamespace(
        method="mtp",
        use_multi_module_mtp=lambda: multi_module_mtp,
    )
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=2,
            pipeline_parallel_size=1,
        ),
        model_config=SimpleNamespace(
            use_mla=True,
            is_encoder_decoder=False,
            hf_text_config=SimpleNamespace(),
        ),
        lora_config=None,
        speculative_config=speculative_config,
        compilation_config=SimpleNamespace(cudagraph_mode=cudagraph_mode),
    )


@pytest.mark.parametrize(
    ("multi_module_mtp", "cudagraph_mode", "error"),
    [
        (True, CUDAGraphMode.NONE, "single-module MTP"),
        (False, CUDAGraphMode.PIECEWISE, "does not support CUDA graphs"),
    ],
)
def test_pcp_rejects_unsupported_mtp(
    multi_module_mtp: bool,
    cudagraph_mode: CUDAGraphMode,
    error: str,
):
    config = _make_validation_config(
        multi_module_mtp=multi_module_mtp,
        cudagraph_mode=cudagraph_mode,
    )

    with pytest.raises(NotImplementedError, match=error):
        PCPManager.validate_config(config, supports_mm_inputs=False)
