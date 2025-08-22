# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.config import VllmConfig


class PredictedProposer:

    def __init__(self, vllm_config: VllmConfig):
        assert vllm_config.speculative_config is not None

        # Maximum length of the model.
        self.max_model_len = vllm_config.model_config.max_model_len
        self.num_speculative_tokens = \
            vllm_config.speculative_config.num_speculative_tokens

    def propose(
        self,
        context_token_ids: list[list[int]],
        predicted_tokens: list[list[int]],
    ) -> list[list[int]]:
        """Effectively a passthrough function that simulates other speculative
        decoding proposers using user request inputs.
        """
        assert len(context_token_ids) == len(predicted_tokens) or len(
            predicted_tokens
        ) == 1, f"{len(context_token_ids)=} vs {len(predicted_tokens)=}"
        if len(context_token_ids) != len(predicted_tokens) and len(
                predicted_tokens) == 1:
            predicted_tokens = predicted_tokens * len(context_token_ids)

        # Clamp to max length (user may not know the bounds).
        # TODO(bwasti) optimize as needed
        clamped_predicted_tokens = []
        for ct, pt in zip(context_token_ids, predicted_tokens):
            total_tokens_len = len(ct)
            predicted_tokens_len = len(pt)
            k = min(predicted_tokens_len, self.num_speculative_tokens)
            k = min(k, max(self.max_model_len - total_tokens_len - 1, 0))
            pt = pt[:k]
            clamped_predicted_tokens.append(pt)

        return clamped_predicted_tokens

    def load_model(self, *args, **kwargs):
        # No model to load.
        pass
