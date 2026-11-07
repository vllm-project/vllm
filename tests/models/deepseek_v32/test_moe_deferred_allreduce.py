# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deferred MoE all-reduce contract between the DSv3.2 NVIDIA decoder and
FusedMoE.

The decoder builds every MoE layer with ``reduce_results=False`` and fuses
the all-reduce of the returned tensor into the next layer's
``fused_allreduce_rms_norm``; the runner must therefore not run its own
late all-reduce on that tensor. ``_defer_moe_allreduce`` in
``vllm.models.deepseek_v32.nvidia.model`` keeps the defer on the one
all2all container whose combine returns an un-reduced output (the naive
all-gather/reduce-scatter pair) and leaves self-reducing all2all combines
and the sequence-parallel layout to the default FusedMoE formula.
"""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.fused_moe.runner.moe_runner as moe_runner
from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.model_executor.layers.fused_moe.layer import make_parallel_config
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.models.deepseek_v32.nvidia.model import _defer_moe_allreduce

_AG_RS = "allgather_reducescatter"
# Combine output_is_reduced() per all2all backend, from source @main:
# naive (ag_rs) returns False; the self-reducing combines return True
# (deepep_ht.py:99-100, deepep_ll.py:138-139,
# flashinfer_nvlink_one_sided.py:69-70). The modules are mirrored, not
# imported: deep_ep / mori are not CPU-test dependencies.
_COMBINE_REDUCES = {
    _AG_RS: False,
    "deepep_high_throughput": True,
    "deepep_low_latency": True,
    "flashinfer_nvlink_one_sided": True,
}

# (tp, pcp, dp, ep, all2all_backend, sequence_parallel, expects_defer):
#  1-3: the naive non-SP combinations, incl. the TP4xPCP2 and TP2xPCP2
#       shapes and the TP=1 identity case;
#  4-5: no all2all engaged (PCP=1 / EP off): the build-time formula already
#       defers there, so the helper must not touch the flag;
#  6-8: self-reducing combines (non-ag_rs backends and SP layouts): the
#       helper must leave the flag alone — their combine is already
#       reduced and the runner skips on its own. Cell 6 (NVLink at TP>1,
#       non-SP) still double-reduces on the decoder side; that corner is
#       the known zero_mlp_output follow-up, so a green here means "no
#       runner double-reduce", not "fixed".
#  9: the naive combine under DP>1 at TP=1 (non-SP by definition);
#  10: the TP=1, DP=2 DeepEP cell from PR review 1 — the exact layout the
#      unconditional form would have crashed (coexist assert), while SP is
#      off only because tensor_parallel_size == 1 keeps it non-SP.
_GRID = [
    (4, 2, 1, True, _AG_RS, False, True),
    (2, 2, 1, True, _AG_RS, False, True),
    (1, 2, 1, True, _AG_RS, False, True),
    (4, 1, 1, True, _AG_RS, False, True),
    (4, 2, 1, False, _AG_RS, False, True),
    (2, 2, 1, True, "flashinfer_nvlink_one_sided", False, False),
    (2, 2, 2, True, "deepep_high_throughput", True, False),
    (2, 2, 2, True, "deepep_low_latency", True, False),
    (1, 1, 2, True, _AG_RS, False, True),
    (1, 1, 2, True, "deepep_high_throughput", False, False),
]


def _stub_group(world_size, rank_in_group=0):
    return SimpleNamespace(world_size=world_size, rank_in_group=rank_in_group)


def _build_moe_cfg(monkeypatch, tp, pcp, dp, ep, backend, sp):
    """A real FusedMoEConfig carrying a real per-cell parallel config,
    going through the real make_parallel_config path."""
    import vllm.model_executor.layers.fused_moe.config as fm_config
    import vllm.model_executor.layers.fused_moe.layer as fm_layer

    monkeypatch.setattr(fm_config, "get_dp_group", lambda: _stub_group(dp))
    monkeypatch.setattr(fm_config, "get_pcp_group", lambda: _stub_group(pcp))
    monkeypatch.setattr(
        fm_config, "get_tensor_model_parallel_rank", lambda: 0
    )
    monkeypatch.setattr(fm_layer, "get_pcp_group", lambda: _stub_group(pcp))

    parallel_config = SimpleNamespace(
        enable_expert_parallel=ep,
        all2all_backend=backend,  # config.py:1198-1240 reads only these
        enable_eplb=False,
    )
    pc = make_parallel_config(
        tp_size=tp,
        dp_size=dp,
        pcp_size=pcp,
        is_sequence_parallel=sp,
        parallel_config=parallel_config,
    )
    # intermediate_size must be divisible by the non-EP flattened
    # tp_size (up to dp*pcp*tp across the grid):
    # FusedMoEConfig.__post_init__ asserts the partitioning and
    # dataclasses.replace re-runs it (config.py @main).
    moe_cfg = make_dummy_moe_config(intermediate_size=64)
    moe_cfg = replace(moe_cfg, moe_parallel_config=pc)
    # The build-time value the FusedMoE factory would have produced
    # (layer.py:245-250): reduce_results=False, zero_expert_type=None.
    moe_cfg.skip_final_all_reduce = not pc.use_all2all_kernels and not sp
    _defer_moe_allreduce(moe_cfg)
    return moe_cfg


# _maybe_reduce_final_output reads these off the runner: moe_config and
# the _fused_output_is_reduced property's quant-method chain.
def _stub_runner(moe_cfg, combine_reduces: bool) -> MoERunner:
    runner = object.__new__(MoERunner)
    torch.nn.Module.__init__(runner)
    runner.moe_config = moe_cfg
    # _quant_method is a read-only property over routed_experts.quant_method.
    runner.routed_experts = SimpleNamespace(
        quant_method=SimpleNamespace(
            moe_kernel=SimpleNamespace(
                output_is_reduced=(lambda: combine_reduces)
            )
        )
    )
    return runner


@pytest.mark.parametrize(
    "tp,pcp,dp,ep,backend,sp,expects_defer", _GRID
)
def test_defer_moe_allreduce_matches_backend(
    monkeypatch: pytest.MonkeyPatch,
    tp: int,
    pcp: int,
    dp: int,
    ep: bool,
    backend: str,
    sp: bool,
    expects_defer: bool,
) -> None:
    """The defer only engages on the non-reducing naive RS combine."""
    moe_cfg = _build_moe_cfg(monkeypatch, tp, pcp, dp, ep, backend, sp)
    assert moe_cfg.skip_final_all_reduce is expects_defer


@pytest.mark.parametrize(
    "tp,pcp,dp,ep,backend,sp,expects_defer", _GRID
)
def test_runner_honors_deferred_allreduce(
    monkeypatch: pytest.MonkeyPatch,
    tp: int,
    pcp: int,
    dp: int,
    ep: bool,
    backend: str,
    sp: bool,
    expects_defer: bool,
) -> None:
    """The real late-all-reduce never fires behind the deferred norm, and
    the skip/coexist assert never trips for self-reducing combines."""
    moe_cfg = _build_moe_cfg(monkeypatch, tp, pcp, dp, ep, backend, sp)
    counter = SimpleNamespace(calls=0)

    def _counting_allreduce(states):
        counter.calls += 1
        return states

    monkeypatch.setattr(
        moe_runner, "tensor_model_parallel_all_reduce", _counting_allreduce
    )
    runner = _stub_runner(moe_cfg, _COMBINE_REDUCES[backend])
    states = torch.zeros(2, 4)
    out = runner._maybe_reduce_final_output(states, None)
    assert out is states
    assert counter.calls == 0, (
        f"tp={tp} pcp={pcp} dp={dp} ep={ep} backend={backend} sp={sp}: "
        "the decoder fuses this MoE's all-reduce into the next layer norm, "
        "so the runner's late all-reduce must not fire"
    )
