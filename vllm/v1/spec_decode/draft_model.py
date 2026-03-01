# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn as nn
from typing_extensions import override

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.v1.spec_decode.eagle import SpecDecodeBaseProposer
from vllm.v1.spec_decode.utils import create_vllm_config_for_draft_model

logger = init_logger(__name__)


class DraftModelProposer(SpecDecodeBaseProposer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        super().__init__(
            vllm_config=vllm_config,
            device=device,
            pass_hidden_states_to_model=False,
            runner=runner,
        )
        self._raise_if_vocab_size_mismatch()
        self._raise_if_draft_tp_mismatch()

    def _raise_if_vocab_size_mismatch(self):
        self.speculative_config.verify_equal_vocab_size_if_draft_model()

    def _raise_if_draft_tp_mismatch(self):
        # Note(Tomas Ruiz) If we run the target model with TP > 1 and
        # the draft model with TP = 1, then the different TP ranks collide.
        # Specifically when all ranks compile the draft model on rank 0
        # (because TP=1), then the torch compile cache is overwritten and corrupted.
        # We need a mechanism like this: https://github.com/vllm-project/vllm/pull/5414
        # To prevent this error, we assert that both TP sizes must be the same.
        spec_cfg = self.speculative_config
        tgt_tp = spec_cfg.target_parallel_config.tensor_parallel_size
        draft_tp = spec_cfg.draft_parallel_config.tensor_parallel_size
        if draft_tp != tgt_tp:
            raise ValueError(
                f"Currently, 'draft_tensor_parallel_size' and 'tensor_parallel_size' "
                f"must be the same. Got {draft_tp} and {tgt_tp}. "
                "Please pass 'draft_tensor_parallel_size' in the speculative_config."
            )

    @override
    def _get_model(self) -> nn.Module:
        # Draft models may be quantized or on different parallelism,
        # so we load them with a modified vllm config
        from vllm.compilation.backends import set_model_tag

        temp_vllm_config = create_vllm_config_for_draft_model(self.vllm_config)
        with set_model_tag("draft_model"):
            model = get_model(
                vllm_config=temp_vllm_config,
                prefix="draft_model",
            )
        return model

    @override
    def _maybe_share_embeddings(self, target_language_model: nn.Module) -> None:
        # Draft models don't share embeddings with the target model
        pass

    @override
    def _maybe_share_lm_head(self, target_language_model: nn.Module) -> None:
        # Draft models don't share lm_head with the target model
        pass
