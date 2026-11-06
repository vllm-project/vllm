# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end correctness test for 2D MoE LoRA expert-parallel
load-time slicing
"""

from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vllm.lora.layers.fused_moe import FusedMoEWithLoRA
from vllm.lora.lora_model import LoRAModel, MoEEPLoadSpec
from vllm.lora.peft_helper import PEFTHelper
from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig
from vllm.model_executor.layers.fused_moe.expert_map_manager import ExpertMapManager

NUM_LAYERS = 48
GLOBAL_NUM_EXPERTS = 128
LOCAL_NUM_EXPERTS = 64  # ep_size = 2
EXPERT_PROJECTIONS = ("down_proj", "gate_proj", "up_proj")
NON_EXPERT_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate")


@pytest.mark.parametrize(
    "use_ep,requested_placement,num_expert_group,resolved_placement",
    [
        (False, "linear", 2, "linear"),
        (False, "round_robin", 2, "linear"),
        (True, "linear", 2, "linear"),
        (True, "round_robin", 1, "linear"),
        (True, "round_robin", 2, "round_robin"),
    ],
)
def test_moe_lora_ep_requires_contiguous_expert_placement(
    use_ep, requested_placement, num_expert_group, resolved_placement
):
    """Reject non-contiguous ownership, but allow requests resolved to linear."""
    parallel_config = replace(
        FusedMoEParallelConfig.make_no_parallel(),
        use_ep=use_ep,
        ep_size=2 if use_ep else 1,
    )
    expert_map = ExpertMapManager(
        max_num_batched_tokens=16,
        top_k=2,
        global_num_experts=4,
        num_redundant_experts=0,
        num_expert_group=num_expert_group,
        moe_parallel_config=parallel_config,
        placement_strategy=requested_placement,
        enable_eplb=False,
    )
    assert expert_map.placement_strategy == resolved_placement

    layer = FusedMoEWithLoRA.__new__(FusedMoEWithLoRA)
    torch.nn.Module.__init__(layer)
    layer.moe_config = SimpleNamespace(moe_parallel_config=parallel_config)
    layer.base_layer = SimpleNamespace(
        expert_placement_strategy=expert_map.placement_strategy
    )
    error = (
        pytest.raises(
            ValueError,
            match="Set expert_placement_strategy='linear' to use MoE LoRA "
            "with expert parallelism",
        )
        if use_ep and resolved_placement != "linear"
        else nullcontext()
    )
    with error:
        layer._ep_check()


def _expected_lora_modules() -> set[str]:
    """Replicate the set ``WorkerLoRAManager._load_adapter`` would build
    from this model's ``packed_modules_mapping``."""
    expected: set[str] = set(NON_EXPERT_MODULES)
    for expert in range(GLOBAL_NUM_EXPERTS):
        for proj in EXPERT_PROJECTIONS:
            expected.add(f"experts.{expert}.{proj}")
    return expected


def _load(lora_dir, peft_helper, *, moe_ep_spec, lora_id):
    return LoRAModel.from_local_checkpoint(
        lora_dir,
        _expected_lora_modules(),
        peft_helper=peft_helper,
        lora_model_id=lora_id,
        device="cpu",
        moe_ep_spec=moe_ep_spec,
    )


@pytest.mark.parametrize("ep_rank", [0, 1])
def test_moe_lora_ep2_real_qwen3moe(qwen3moe_lora_files, ep_rank):
    """ep_size=2 against the real Qwen3-MoE adapter: each rank's loaded
    LoRA has the right size, the right expert membership, and the
    right tensor values."""
    peft_helper = PEFTHelper.from_local_dir(
        qwen3moe_lora_files, max_position_embeddings=4096
    )

    # Baseline: no spec → loads every expert × projection × layer plus
    # all non-expert LoRA modules.
    ground_truth = _load(qwen3moe_lora_files, peft_helper, moe_ep_spec=None, lora_id=1)
    expected_baseline = (
        GLOBAL_NUM_EXPERTS * len(EXPERT_PROJECTIONS) * NUM_LAYERS
        + len(NON_EXPERT_MODULES) * NUM_LAYERS
    )
    assert len(ground_truth.loras) == expected_baseline

    # Sliced load: only this rank's experts; non-expert LoRA is untouched.
    spec = MoEEPLoadSpec(
        ep_rank=ep_rank,
        local_num_experts=LOCAL_NUM_EXPERTS,
        global_num_experts=GLOBAL_NUM_EXPERTS,
    )
    sliced = _load(
        qwen3moe_lora_files,
        peft_helper,
        moe_ep_spec=spec,
        lora_id=100 + ep_rank,
    )

    expected_sliced = (
        LOCAL_NUM_EXPERTS * len(EXPERT_PROJECTIONS) * NUM_LAYERS
        + len(NON_EXPERT_MODULES) * NUM_LAYERS
    )
    assert len(sliced.loras) == expected_sliced

    expert_start = ep_rank * LOCAL_NUM_EXPERTS
    expert_end = expert_start + LOCAL_NUM_EXPERTS

    for name, lora in sliced.loras.items():
        gt = ground_truth.loras[name]
        torch.testing.assert_close(lora.lora_a, gt.lora_a)
        torch.testing.assert_close(lora.lora_b, gt.lora_b)
        if ".experts." in name:
            expert_idx = int(name.split(".experts.")[-1].split(".")[0])
            assert expert_start <= expert_idx < expert_end, (
                f"non-local expert {expert_idx} leaked: {name}"
            )
