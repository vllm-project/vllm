# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

import torch

from vllm.distributed import get_ep_group

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner


class EplbCheckExtension:
    model_runner: "GPUModelRunner"

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
