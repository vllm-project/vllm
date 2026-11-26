# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import numpy as np
import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.platforms import CpuArchEnum, current_platform
from vllm.triton_utils import HAS_TRITON
from vllm.utils.network_utils import get_open_port
from vllm.v1.worker.cpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.cpu.pp_utils import PPHandler, compute_need_sampled_mask

pytestmark = pytest.mark.skipif(
    not current_platform.is_cpu()
    or HAS_TRITON
    or current_platform.get_cpu_architecture() == CpuArchEnum.X86,
    reason="requires a non-x86 CPU platform without Triton",
)


def _batch(
    *,
    num_computed: list[int],
    prefill_len: list[int],
    num_scheduled: list[int],
    is_prefilling: list[bool],
    max_seq_len: list[int] | None,
    idx_mapping: list[int] | None = None,
) -> InputBatch:
    num_reqs = len(num_computed)
    idx_mapping = idx_mapping or list(range(num_reqs))
    idx_mapping_np = np.array(idx_mapping, dtype=np.int64)
    num_tokens = sum(num_scheduled)
    batch = InputBatch.make_dummy(
        num_reqs,
        num_tokens,
        InputBuffers(num_reqs, num_tokens, torch.device("cpu")),
    )
    batch.idx_mapping = torch.from_numpy(idx_mapping_np)
    batch.idx_mapping_np = idx_mapping_np
    batch.num_computed_tokens_np = np.array(num_computed, dtype=np.int32)
    batch.prefill_len_np = np.array(prefill_len, dtype=np.int32)
    batch.num_scheduled_tokens = np.array(num_scheduled, dtype=np.int32)
    batch.is_prefilling_np = np.array(is_prefilling, dtype=np.bool_)
    batch.max_seq_len_np = (
        None if max_seq_len is None else np.array(max_seq_len, dtype=np.int32)
    )
    return batch


@pytest.mark.parametrize(
    (
        "num_computed",
        "prefill_len",
        "num_scheduled",
        "is_prefilling",
        "max_seq_len",
        "expected",
    ),
    [
        ([512, 1000], [4096, 1004], [448, 4], [True, True], None, [False, True]),
        ([0, 512], [4096, 4096], [448, 448], [True, True], None, None),
        ([10, 512], [8, 4096], [1, 448], [False, True], None, [True, False]),
        ([1000], [1004], [4], [True], [1005], None),
        ([1000], [1004], [4], [True], [1260], [True]),
        (
            [1000, 10],
            [1004, 8],
            [4, 1],
            [True, False],
            [1005, 1024],
            [False, True],
        ),
    ],
)
def test_cpu_pp_sample_mask_matches_supported_gpu_scenarios(
    num_computed: list[int],
    prefill_len: list[int],
    num_scheduled: list[int],
    is_prefilling: list[bool],
    max_seq_len: list[int] | None,
    expected: list[bool] | None,
) -> None:
    batch = _batch(
        num_computed=num_computed,
        prefill_len=prefill_len,
        num_scheduled=num_scheduled,
        is_prefilling=is_prefilling,
        max_seq_len=max_seq_len,
    )

    mask = compute_need_sampled_mask(batch)

    assert (None if mask is None else mask.tolist()) == expected


def _run_cpu_pp_sample_sync(rank: int, port: int) -> None:
    try:
        os.environ["VLLM_DIST_IDENT"] = str(port)
        with set_current_vllm_config(VllmConfig()):
            init_distributed_environment(
                world_size=2,
                rank=rank,
                local_rank=rank,
                backend="gloo",
                distributed_init_method=f"tcp://127.0.0.1:{port}",
            )
            initialize_model_parallel(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=2,
                backend="gloo",
            )
            handler = PPHandler(max_num_reqs=4, device=torch.device("cpu"))
            batch = _batch(
                num_computed=[4, 10],
                prefill_len=[4, 8],
                num_scheduled=[1, 1],
                is_prefilling=[False, False],
                max_seq_len=None,
                idx_mapping=[2, 0],
            )

            assert handler.get_prev_sampled_outputs() is None
            if handler.is_last_rank:
                handler.broadcast(
                    sampled_token_ids=torch.tensor([[71], [82]], dtype=torch.int64),
                    num_sampled=torch.tensor([1, 1], dtype=torch.int32),
                    num_rejected=torch.tensor([0, 0], dtype=torch.int32),
                    input_batch=batch,
                )
            else:
                assert handler.receive(batch)
                assert handler.get_prev_sampled_outputs() is None
                handler.on_req_idx_freed(0)
                outputs = handler.get_prev_sampled_outputs()
                assert outputs is not None
                torch.testing.assert_close(
                    outputs["sampled_tokens"],
                    torch.tensor([[71], [82]], dtype=torch.int64),
                )
                torch.testing.assert_close(
                    outputs["num_sampled"], torch.tensor([1, 1], dtype=torch.int32)
                )
                torch.testing.assert_close(
                    outputs["num_rejected"], torch.tensor([0, 0], dtype=torch.int32)
                )
                torch.testing.assert_close(
                    outputs["idx_mapping"],
                    torch.tensor([2, -1], dtype=torch.int64),
                )

            torch.distributed.barrier()
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


def test_cpu_pp_synchronizes_samples_and_invalidates_reused_slots() -> None:
    torch.multiprocessing.spawn(
        _run_cpu_pp_sample_sync,
        args=(get_open_port(),),
        nprocs=2,
        join=True,
    )
