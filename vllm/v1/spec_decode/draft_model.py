# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import replace
from typing import Any

import torch

from vllm.attention.layer import Attention
from vllm.config import ModelConfig, VllmConfig, get_layers_from_vllm_config
from vllm.forward_context import set_forward_context
from vllm.model_executor.model_loader import get_model
from vllm.v1.spec_decode.eagle import SpecDecodeBaseProposer


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
            pass_cudagraph_args_to_forward_ctx=True,
            # The draft model runs one forward pass to prefill
            # the target_token_ids, and another forward pass for decoding
            # based on the next_token_ids. I.e. it needs 1 more forward pass.
            one_extra_forward_pass=True,
            # the first draft_token_ids are replaced by next_token_ids, so
            # they don't need to be returned as proposed tokens
            drop_first_drafted_tokens=True,
            runner=runner,
        )
        self._raise_if_multimodal()
        self._raise_if_mrope()

    def _raise_if_multimodal(self):
        if self.supports_mm_inputs:
            raise NotImplementedError(
                "Speculative Decoding with draft models "
                "does not support multimodal models yet"
            )

    def _raise_if_mrope(self):
        if self.draft_model_config.uses_mrope:
            raise NotImplementedError(
                "Speculative Decoding with draft models does not support M-RoPE yet"
            )

    def _model_kwargs(self, num_tokens: int) -> dict[str, Any]:
        self._raise_if_multimodal()
        self._raise_if_mrope()
        return {
            "input_ids": self.input_ids[:num_tokens],
            "positions": self.positions[:num_tokens],
        }

    def dummy_run(self, num_tokens: int, forward_ctx_kwargs: dict):
        model_kwargs = self._model_kwargs(num_tokens)
        assert isinstance(self.model, torch.nn.Module)
        with set_forward_context(
            vllm_config=self.vllm_config,
            num_tokens=num_tokens,
            **forward_ctx_kwargs,
        ):
            self.model(**model_kwargs)

    def set_input_ids_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        num_tokens: int,
        last_token_indices: torch.Tensor,
    ) -> None:
        self.input_ids[:num_tokens] = target_token_ids

    def load_model(self, target_model: Any) -> None:
        """Takes target_model to satisfy the type checker."""
        draft_model_config: ModelConfig = (
            self.vllm_config.speculative_config.draft_model_config
        )
        vllm_config_draft: VllmConfig = replace(
            self.vllm_config, model_config=draft_model_config
        )

        # This must be computed before loading the draft model
        # because that mutates the forward_context of the vllm_config
        target_attn_layer_names = set(
            get_layers_from_vllm_config(self.vllm_config, Attention).keys()
        )

        from vllm.compilation.backends import set_model_tag

        with set_model_tag("draft_model"):
            self.model = get_model(
                vllm_config=vllm_config_draft,
                model_config=draft_model_config,
                prefix="draft_model",
            )

        # This must be computed after loading the draft model
        # because that mutates the forward_context of the vllm_config
        draft_attn_layer_names = (
            get_layers_from_vllm_config(self.vllm_config, Attention).keys()
            - target_attn_layer_names
        )
        self.attn_layer_names = list(draft_attn_layer_names)
