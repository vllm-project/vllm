# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
from typing_extensions import override

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.v1.spec_decode.eagle import SpecDecodeBaseProposer
from vllm.v1.spec_decode.utils import create_vllm_config_for_draft_model
from vllm.v1.spec_decode.vocab_mapping import VocabMapping

logger = init_logger(__name__)


class UniversalDraftModelProposer(SpecDecodeBaseProposer):
    """Draft model proposer supporting heterogeneous vocabularies via TLI."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device, runner=None):
        super().__init__(vllm_config=vllm_config, device=device,
                         pass_hidden_states_to_model=False, runner=runner)
        self._raise_if_draft_tp_mismatch()

        spec_config = self.speculative_config
        assert spec_config.draft_model_config is not None
        draft_model_path = spec_config.draft_model_config.model
        self.draft_tokenizer = get_tokenizer(
            draft_model_path,
            trust_remote_code=spec_config.draft_model_config.trust_remote_code)

        assert spec_config.target_model_config is not None
        target_model_path = spec_config.target_model_config.tokenizer
        self.target_tokenizer = get_tokenizer(
            target_model_path,
            trust_remote_code=spec_config.target_model_config.trust_remote_code)

        target_vocab_size = spec_config.target_model_config.get_vocab_size()
        draft_vocab_size = spec_config.draft_model_config.get_vocab_size()
        self.vocab_mapping = VocabMapping(
            target_tokenizer=self.target_tokenizer,
            draft_tokenizer=self.draft_tokenizer,
            target_vocab_size=target_vocab_size,
            draft_vocab_size=draft_vocab_size,
            device=device)

    def _raise_if_draft_tp_mismatch(self):
        spec_cfg = self.speculative_config
        tgt_tp = spec_cfg.target_parallel_config.tensor_parallel_size
        draft_tp = spec_cfg.draft_parallel_config.tensor_parallel_size
        if draft_tp != tgt_tp:
            raise ValueError(
                f"draft_tensor_parallel_size ({draft_tp}) must match "
                f"tensor_parallel_size ({tgt_tp}).")

    @override
    def _get_model(self) -> nn.Module:
        from vllm.compilation.backends import set_model_tag
        temp_vllm_config = create_vllm_config_for_draft_model(self.vllm_config)
        with set_model_tag("draft_model"):
            model = get_model(vllm_config=temp_vllm_config, prefix="draft_model")
        return model

    @override
    def _maybe_share_embeddings(self, target_language_model: nn.Module) -> None:
        pass  # Don't share embeddings for heterogeneous vocab

    @override
    def _maybe_share_lm_head(self, target_language_model: nn.Module) -> None:
        pass  # Don't share lm_head for heterogeneous vocab

    @override
    def model_returns_tuple(self) -> bool:
        return False

    @override
    def _greedy_sample(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.model.compute_logits(hidden_states)
        constrained_logits = self.vocab_mapping.constrain_draft_logits(logits)
        draft_token_ids = constrained_logits.argmax(dim=-1)
        target_token_ids = self.vocab_mapping.map_draft_to_target_ids(draft_token_ids)
        return target_token_ids

    @override
    def _prepare_draft_input_ids(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.vocab_mapping.map_target_to_draft_ids(token_ids)

    @override
    def set_inputs_first_pass(self, target_token_ids, next_token_ids,
                              target_positions, target_hidden_states,
                              token_indices_to_sample, cad, num_rejected_tokens_gpu):
        draft_token_ids = self.vocab_mapping.map_target_to_draft_ids(target_token_ids)
        draft_next_token_ids = self.vocab_mapping.map_target_to_draft_ids(next_token_ids)
        return super().set_inputs_first_pass(
            target_token_ids=draft_token_ids,
            next_token_ids=draft_next_token_ids,
            target_positions=target_positions,
            target_hidden_states=target_hidden_states,
            token_indices_to_sample=token_indices_to_sample,
            cad=cad,
            num_rejected_tokens_gpu=num_rejected_tokens_gpu)
