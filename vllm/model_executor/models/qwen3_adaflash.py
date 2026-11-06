# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from .qwen3_dflash import DFlashQwen3ForCausalLM

_LENGTH_HEAD_PREFIX = "thresh_head_two_model."


class AdaFlashLengthHead(nn.Module):
    """Predict a verification-length ratio from a complete draft block."""

    def __init__(self, hidden_size: int, bottleneck_dim: int = 256):
        super().__init__()
        self.down = nn.Linear(2 * hidden_size, bottleneck_dim)
        self.res = nn.Sequential(nn.Linear(bottleneck_dim, bottleneck_dim), nn.SiLU())
        self.out = nn.Linear(bottleneck_dim, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        features = torch.cat((hidden_states.mean(dim=1), hidden_states[:, -1]), dim=-1)
        hidden = F.silu(self.down(features))
        return torch.sigmoid(self.out(hidden + self.res(hidden)))


class AdaFlashQwen3ForCausalLM(DFlashQwen3ForCausalLM):
    """AdaFlash drafter: a DFlash drafter plus a verification-length head."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        dflash_config = getattr(self.config, "dflash_config", None) or {}
        self.thresh_head_two_model = AdaFlashLengthHead(
            self.config.hidden_size,
            dflash_config.get("thresh_head_bottleneck_dim", 256),
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        head_weights: list[tuple[str, torch.Tensor]] = []

        def draft_weights():
            for name, weight in weights:
                if name.startswith(_LENGTH_HEAD_PREFIX):
                    head_weights.append((name, weight))
                else:
                    yield name, weight

        super().load_weights(draft_weights())
        params = dict(self.thresh_head_two_model.named_parameters())
        loaded = set()
        for name, weight in head_weights:
            key = name[len(_LENGTH_HEAD_PREFIX) :]
            if key not in params:
                raise ValueError(f"Unexpected AdaFlash length-head weight: {name}")
            default_weight_loader(params[key], weight)
            loaded.add(key)
        missing = set(params) - loaded
        if missing:
            raise ValueError(f"AdaFlash length-head weights missing: {sorted(missing)}")
