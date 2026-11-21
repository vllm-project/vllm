# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deferred MoE all-reduce contract for the naive all2all path.

The deepseek_v32 decoder builds every MoE layer with ``reduce_results=False``
and fuses the all-reduce of the returned tensor into the next layer's
``fused_allreduce_rms_norm``. ``resolve_skip_final_all_reduce`` keeps
``skip_final_all_reduce=True`` for the naive all-gather/reduce-scatter
container (its combine leaves the TP reduction to the consumer) and leaves
self-reducing combines and the sequence-parallel layout to the runner.
"""

import pytest
import torch

from tests.single_rank_group import SingleRankGroup, override_parallel_groups
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoEFactory,
    make_parallel_config,
    resolve_skip_final_all_reduce,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    make_moe_prepare_and_finalize_naive_dp_ep,
)
from vllm.v1.worker.workspace import init_workspace_manager

_AG_RS = "allgather_reducescatter"

# (tp, pcp, dp, ep, all2all_backend, sequence_parallel, expects_defer)
_GRID = [
    (4, 2, 1, True, _AG_RS, False, True),
    (2, 2, 1, True, _AG_RS, False, True),
    (2, 4, 1, True, _AG_RS, False, True),
    (1, 2, 1, True, _AG_RS, False, True),
    (4, 1, 1, True, _AG_RS, False, True),
    (4, 2, 1, False, _AG_RS, False, True),
    (1, 1, 2, True, _AG_RS, False, True),
    (2, 2, 1, True, "deepep_high_throughput", False, False),
    (2, 2, 1, True, "deepep_low_latency", False, False),
    (2, 2, 1, True, "flashinfer_nvlink_one_sided", False, False),
    (2, 2, 1, True, "nixl_ep", False, False),
    (2, 2, 1, True, "alltoall_batched", False, False),
    (2, 2, 1, True, "flashinfer_all2allv", False, False),
    (1, 1, 2, True, "deepep_high_throughput", False, False),
    (2, 2, 1, True, _AG_RS, True, False),
    (2, 2, 2, True, "deepep_high_throughput", True, False),
]


def _parallel_config(tp, pcp, dp, ep, backend):
    return ParallelConfig(
        tensor_parallel_size=tp,
        data_parallel_size=dp,
        prefill_context_parallel_size=pcp,
        enable_expert_parallel=ep,
        all2all_backend=backend,
    )


@override_parallel_groups(SingleRankGroup())
@pytest.mark.parametrize("tp,pcp,dp,ep,backend,sp,expects_defer", _GRID)
def test_resolve_skip_final_all_reduce_matches_backend(
    tp: int, pcp: int, dp: int, ep: bool, backend: str, sp: bool, expects_defer: bool
) -> None:
    pc = make_parallel_config(
        tp_size=tp,
        dp_size=dp,
        pcp_size=pcp,
        is_sequence_parallel=sp,
        parallel_config=_parallel_config(tp, pcp, dp, ep, backend),
    )
    assert resolve_skip_final_all_reduce(False, pc, None) is expects_defer


def test_ag_rs_prepare_finalize_reports_unreduced() -> None:
    """The AgRs container is the one the deferral exists for."""
    for use_monolithic in (False, True):
        pf = make_moe_prepare_and_finalize_naive_dp_ep(use_monolithic=use_monolithic)
        assert pf.output_is_reduced() is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@override_parallel_groups(SingleRankGroup())
@pytest.mark.parametrize(
    "tp,pcp,ep,expects_defer",
    [
        (4, 2, True, True),
        (1, 1, False, True),
    ],
)
def test_factory_wires_deferred_allreduce(
    tp: int, pcp: int, ep: bool, expects_defer: bool
) -> None:
    """The real FusedMoEFactory must carry the resolved flag on the config."""
    vllm_config = VllmConfig()
    vllm_config.compilation_config.static_forward_context = {}
    vllm_config.parallel_config.enable_expert_parallel = ep
    vllm_config.parallel_config.all2all_backend = _AG_RS
    with set_current_vllm_config(vllm_config), set_forward_context(None, vllm_config):
        init_workspace_manager(torch.accelerator.current_device_index())
        runner = FusedMoEFactory(
            num_experts=4,
            top_k=2,
            hidden_size=256,
            intermediate_size=512,
            params_dtype=torch.bfloat16,
            prefix="model.layers.0.mlp.experts",
            renormalize=False,
            reduce_results=False,
            tp_size=tp,
            pcp_size=pcp,
        ).cuda()
        assert runner.moe_config.skip_final_all_reduce is expects_defer
