# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn as nn
from typing_extensions import override

from vllm.config import VllmConfig
from vllm.config.utils import replace
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
from vllm.v1.spec_decode.vocab_mapping import VocabMapping

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
        self._raise_if_draft_tp_mismatch()

        spec = self.speculative_config
        if spec.uses_universal_draft():
            # Heterogeneous vocabularies: build a VocabMapping to translate
            # token IDs between the two tokenizers and constrain draft logits
            # to the intersection so rejection sampling stays lossless.
            target_tokenizer = get_tokenizer(
                spec.target_model_config.tokenizer,
                trust_remote_code=spec.target_model_config.trust_remote_code,
            )
            draft_tokenizer = get_tokenizer(
                spec.draft_model_config.model,
                trust_remote_code=spec.draft_model_config.trust_remote_code,
            )
            self.vocab_mapping: VocabMapping | None = VocabMapping(
                target_tokenizer=target_tokenizer,
                draft_tokenizer=draft_tokenizer,
                target_vocab_size=spec.target_model_config.get_vocab_size(),
                draft_vocab_size=spec.draft_model_config.get_vocab_size(),
                device=device,
            )
        else:
            self._raise_if_vocab_size_mismatch()
            self.vocab_mapping = None

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
    def _create_draft_vllm_config(self) -> VllmConfig:
        base = super()._create_draft_vllm_config()
        spec = self.speculative_config

        return replace(
            base,
            quant_config=None,
            parallel_config=replace(
                spec.draft_parallel_config,
                rank=self.vllm_config.parallel_config.rank,
            ),
            model_config=spec.draft_model_config,
        )

    @override
    def _get_model(self) -> nn.Module:
        from vllm.compilation.backends import set_model_tag

        draft_vllm_config = self._create_draft_vllm_config()
        with set_model_tag("draft_model"):
            model = get_model(
                vllm_config=draft_vllm_config,
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

    @override
    def _prepare_draft_input_ids(self, token_ids: torch.Tensor) -> torch.Tensor:
        if self.vocab_mapping is not None:
            return self.vocab_mapping.map_target_to_draft_ids(token_ids)
        return token_ids

    @override
    def _greedy_sample(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.model.compute_logits(hidden_states)
        if self.vocab_mapping is not None:
            logits = self.vocab_mapping.constrain_draft_logits(logits)
        draft_token_ids = logits.argmax(dim=-1)
        if self.vocab_mapping is not None:
            return self.vocab_mapping.map_draft_to_target_ids(draft_token_ids)
        return draft_token_ids

    @override
    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        cad: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
    ) -> tuple[int, torch.Tensor, CommonAttentionMetadata]:
        if self.vocab_mapping is not None:
            target_token_ids = self.vocab_mapping.map_target_to_draft_ids(
                target_token_ids)
            next_token_ids = self.vocab_mapping.map_target_to_draft_ids(
                next_token_ids)
        return super().set_inputs_first_pass(
            target_token_ids=target_token_ids,
            next_token_ids=next_token_ids,
            target_positions=target_positions,
            target_hidden_states=target_hidden_states,
            token_indices_to_sample=token_indices_to_sample,
            cad=cad,
            num_rejected_tokens_gpu=num_rejected_tokens_gpu,
        )