# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable

import torch

from .qwen3_dflash import DFlashQwen3ForCausalLM


class AdaFlashQwen3ForCausalLM(DFlashQwen3ForCausalLM):
    """AdaFlash drafter: a DFlash drafter plus a verification-length head."""

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        # The length head is not used yet.
        return super().load_weights(
            (name, weight)
            for name, weight in weights
            if not name.startswith("thresh_head")
        )
