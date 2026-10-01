# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.distributed import get_ep_group


class EplbCheckExtension:
    def check_eplb_replica_slots(self) -> str | None:
        """Redundant expert slots must hold their source expert's weights."""
        state = next(iter(self.model_runner.eplb_state.model_states.values()))
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
