# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

import torch

from vllm.distributed import get_ep_group

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner


class EplbCheckExtension:
    model_runner: "GPUModelRunner"

    def get_weight_cache_eplb_placement(self) -> dict:
        """Return the committed placement after checking the runtime map."""
        eplb_state = self.model_runner.eplb_state
        assert eplb_state is not None
        state = next(iter(eplb_state.model_states.values()))
        binding = state.weight_cache_binding
        assert binding is not None
        placement = binding.placement
        runtime_map = tuple(
            tuple(row) for row in state.physical_to_logical_map.cpu().tolist()
        )
        assert runtime_map == placement.physical_to_logical_map
        return placement.to_json_dict()

    def rearrange_eplb_deterministic(self) -> dict:
        """Run a real synchronous rearrangement with a fixed logical load."""
        eplb_state = self.model_runner.eplb_state
        assert eplb_state is not None and not eplb_state.is_async
        state = next(iter(eplb_state.model_states.values()))
        state.last_expert_load = torch.arange(
            state.model.num_logical_experts,
            dtype=state.last_expert_load.dtype,
            device=state.last_expert_load.device,
        ).repeat(state.model.num_moe_layers, 1)
        eplb_state.rearrange(use_last_expert_load=True)
        return self.get_weight_cache_eplb_placement()

    def check_eplb_replica_slots(self) -> str | None:
        """Redundant expert slots must hold their source expert's weights."""
        eplb_state = self.model_runner.eplb_state
        assert eplb_state is not None
        state = next(iter(eplb_state.model_states.values()))
        sums = [
            sum(
                torch.stack([e.view(torch.uint8).sum(dtype=torch.int64) for e in w])
                for w in layer_weights
            )
            for layer_weights in state.model.expert_weights
        ]
        sums = get_ep_group().all_gather(torch.stack(sums), dim=1).tolist()
        bad = sum(
            s[slot] != s[p2l.index(logical)]
            for s, p2l in zip(sums, state.physical_to_logical_map.tolist())
            for slot, logical in enumerate(p2l)
        )
        return f"{bad} redundant expert slots differ from their source" if bad else None
