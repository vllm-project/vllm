# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SOTA-model E2E test for EP-sharded weight load.

Uses a real 80B-class MoE checkpoint (Qwen3-Next-80B-A3B-Instruct-FP8) on
TP=2, EP=2 and verifies:

1. ``get_moe_routed_ep_layout`` returns a valid per-layer EP layout with
   ``local_num_routed_experts = global_num_routed_experts / ep_size`` on each
   rollout rank.
2. ``FusedMoE.load_routed_expert_weights`` bit-exactly writes a fused 3D
   ``[local_routed_E, 2I_per_partition, H]`` tensor into live model params on
   each rank's local expert slots — on a real 80B-class MoE.

All assertions run inside the worker process via ``collective_rpc``.
"""

from __future__ import annotations

import os

import pytest
import torch

from vllm import LLM
from vllm.config import WeightTransferConfig

from ...utils import create_new_process_for_each_test

SOTA_MOE_MODEL_PATH = "/mnt/shared/models/Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"


def _prep_env():
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"


@pytest.mark.skipif(
    torch.accelerator.device_count() < 2,
    reason="Need >=2 GPUs for TP=2+EP=2 SOTA e2e.",
)
@pytest.mark.skipif(
    not os.path.isdir(SOTA_MOE_MODEL_PATH),
    reason=f"SOTA MoE checkpoint not found at {SOTA_MOE_MODEL_PATH}.",
)
@create_new_process_for_each_test()
def test_sota_moe_sharded_load_end_to_end():
    _prep_env()

    FUSED_MAPPING = [
        ("w13_weight", "gate_up_proj", 0, "w1"),
        ("w13_weight", "gate_up_proj", 1, "w3"),
        ("w2_weight", "down_proj", 0, "w2"),
    ]

    llm = LLM(
        model=SOTA_MOE_MODEL_PATH,
        enforce_eager=True,
        tensor_parallel_size=2,
        enable_expert_parallel=True,
        gpu_memory_utilization=0.85,
        max_model_len=2048,
        weight_transfer_config=WeightTransferConfig(backend="nccl"),
    )

    # (1) Layout RPC must return per-layer EP info consistent across ranks.
    def collect_layout(self):
        return self.get_moe_routed_ep_layout()

    layouts = llm.collective_rpc(collect_layout)
    assert len(layouts) == 2, "expected 2 ranks (TP=2, EP=2)"
    for rank_i, layout in enumerate(layouts):
        assert isinstance(layout, dict) and len(layout) > 0, (
            f"rank {rank_i}: empty layout"
        )
        for layer_name, info in layout.items():
            assert info["ep_size"] == 2
            assert info["ep_rank"] == rank_i
            assert (
                info["local_num_routed_experts"]
                == info["global_num_routed_experts"] // 2
            ), f"{layer_name}: local != global/ep"
            assert (
                len(info["local_to_global_routed"]) == info["local_num_routed_experts"]
            )
            for g in info["local_to_global_routed"]:
                assert 0 <= g < info["global_num_routed_experts"]

    # (2) Pick the first FusedMoE layer, inject fused mapping, bit-exactly
    # verify writes.
    def exercise_sharded(self):
        from vllm.model_executor.layers.fused_moe.layer import FusedMoE

        model = self.model_runner.model
        first = next((m for m in model.modules() if isinstance(m, FusedMoE)), None)
        if first is None:
            return {"ok": False, "reason": "no FusedMoE layer"}

        first.expert_mapping = FUSED_MAPPING

        assert first._expert_map is not None
        local_to_global = (first._expert_map != -1).nonzero().flatten().tolist()
        local_E = len(local_to_global)

        w13 = first.w13_weight.data
        w2 = first.w2_weight.data
        dev, dt = w13.device, w13.dtype

        # Build fused 3D source with per-expert unique pattern on each rank.
        w13_src = torch.empty(local_E, w13.shape[1], w13.shape[2], dtype=dt, device=dev)
        w2_src = torch.empty(local_E, w2.shape[1], w2.shape[2], dtype=dt, device=dev)
        for i, gid in enumerate(local_to_global):
            # Clamp within fp8 range; use small integers.
            w13_src[i].fill_(float((gid % 3) + 1))
            w2_src[i].fill_(-float((gid % 3) + 1))

        w13.zero_()
        w2.zero_()

        list(
            first.load_routed_expert_weights(
                [("gate_up_proj", w13_src)],
                {"gate_up_proj": local_to_global},
            )
        )
        list(
            first.load_routed_expert_weights(
                [("down_proj", w2_src)],
                {"down_proj": local_to_global},
            )
        )

        ok = True
        mismatches: list[str] = []
        for i, gid in enumerate(local_to_global):
            exp13 = float((gid % 3) + 1)
            exp2 = -float((gid % 3) + 1)
            if not torch.all(w13[i] == exp13).item():
                ok = False
                mismatches.append(
                    f"w13 local={i} gid={gid} uniq={w13[i].unique().tolist()[:3]}"
                )
            if not torch.all(w2[i] == exp2).item():
                ok = False
                mismatches.append(
                    f"w2  local={i} gid={gid} uniq={w2[i].unique().tolist()[:3]}"
                )
        return {
            "ok": ok,
            "layer": first.layer_name,
            "local_E": local_E,
            "first_global": local_to_global[:3],
            "last_global": local_to_global[-3:],
            "w13_shape": list(w13.shape),
            "w2_shape": list(w2.shape),
            "dtype": str(dt),
            "mismatches": mismatches[:3],
        }

    results = llm.collective_rpc(exercise_sharded)
    for r in results:
        assert r["ok"], f"sharded write failed on rank: {r}"
    # Summary for the log.
    local_Es = [r["local_E"] for r in results]
    print(
        "[SOTA e2e]",
        "layer=",
        results[0]["layer"],
        "w13_shape=",
        results[0]["w13_shape"],
        "w2_shape=",
        results[0]["w2_shape"],
        "dtype=",
        results[0]["dtype"],
        "ranks_local_E=",
        local_Es,
    )
