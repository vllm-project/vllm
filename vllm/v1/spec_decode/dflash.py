# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.config import VllmConfig
from vllm.v1.spec_decode.eagle import EagleProposer


class DFlashProposer(EagleProposer):
    """Dedicated proposer for method='dflash' with DFlash-specific config."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        super().__init__(vllm_config=vllm_config, device=device, runner=runner)

    def _get_eagle3_use_aux_hidden_state_from_config(self) -> bool:
        """
        DFlash config precedence:
        1) dflash_config.use_aux_hidden_state
        2) eagle_config.use_aux_hidden_state
        3) default True
        """
        use_aux_hidden_state = True

        eagle_config = getattr(self.draft_model_config.hf_config, "eagle_config", None)
        if isinstance(eagle_config, dict):
            use_aux_hidden_state = eagle_config.get("use_aux_hidden_state", True)

        dflash_config = getattr(
            self.draft_model_config.hf_config, "dflash_config", None
        )
        if isinstance(dflash_config, dict):
            use_aux_hidden_state = dflash_config.get(
                "use_aux_hidden_state", use_aux_hidden_state
            )

        return use_aux_hidden_state

    def propose(self, *args, **kwargs) -> torch.Tensor:
        common_attn_metadata = kwargs.get("common_attn_metadata")
        if common_attn_metadata is not None and common_attn_metadata.batch_size() > 1:
            raise NotImplementedError(
                "DFlash speculative decoding currently supports batch size 1 only. "
                "Please reduce max_num_seqs to 1 for method='dflash'."
            )
        return super().propose(*args, **kwargs)
