# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gumiho proposer for V1 speculative decoding.

Reference:
    Li et al., "Gumiho: A Hybrid Architecture to Prioritize Early Tokens in
    Speculative Decoding", ICML 2025. https://arxiv.org/abs/2503.10135

Gumiho is a hybrid drafter: the first two speculative tokens are produced
autoregressively by a small transformer head (exactly like EAGLE), and every
additional speculative token is produced in parallel by per-step MLP heads.

This module only contains the proposer plumbing. The drafter model itself
lives in :mod:`vllm.model_executor.models.gumiho`. We reuse
:class:`~vllm.v1.spec_decode.eagle.EagleProposer` (which in turn extends
:class:`~vllm.v1.spec_decode.llm_base_proposer.SpecDecodeBaseProposer`)
to inherit the EAGLE-style input padding, KV slot bookkeeping, attention
metadata building and cudagraph dispatch logic, and only override the two
hooks that ``SpecDecodeBaseProposer.propose`` calls during the sequential
drafting loop.

The control flow is:

1. ``propose()`` runs the transformer draft head for the first speculative
   token (``sample_hidden_states``) and the second token (first iteration of
   the sequential loop).
2. After the second token, :meth:`_maybe_get_mlp_draft_token_ids` invokes
   :meth:`GumihoLlamaForCausalLM.generate_mlp_draft_token_ids` to produce the
   remaining ``num_speculative_tokens - 2`` tokens in parallel.
3. The loop breaks and the result is handed to the V1 verifier / rejection
   sampler unchanged.
"""

import torch

from vllm.v1.spec_decode.eagle import EagleProposer


class GumihoProposer(EagleProposer):
    """EAGLE-like proposer that calls Gumiho's MLP heads after step 2."""

    def _init_draft_hidden_states_list(
        self,
        sample_hidden_states: torch.Tensor,
    ) -> list[torch.Tensor] | None:
        # MLP heads are only used when at least one extra token has to be
        # generated after the transformer draft head's two autoregressive
        # tokens. Otherwise behave exactly like a plain EagleProposer.
        if self.num_speculative_tokens <= 2:
            return None
        return [sample_hidden_states]

    def _maybe_get_mlp_draft_token_ids(
        self,
        draft_token_ids_list: list[torch.Tensor],
        draft_hidden_states_list: list[torch.Tensor] | None,
    ) -> torch.Tensor | None:
        # Need the hidden states for both transformer-generated tokens to fuse.
        if draft_hidden_states_list is None or len(draft_hidden_states_list) < 2:
            return None
        remaining = self.num_speculative_tokens - len(draft_token_ids_list)
        if remaining <= 0:
            return None
        # Drafter checkpoints without MLP heads (or with a different API) just
        # fall through to the default sequential path.
        generate_mlp_draft_token_ids = getattr(
            self.model, "generate_mlp_draft_token_ids", None
        )
        if generate_mlp_draft_token_ids is None:
            return None
        draft_token_ids = generate_mlp_draft_token_ids(
            draft_token_ids=draft_token_ids_list[:2],
            draft_hidden_states=draft_hidden_states_list[:2],
            num_tokens=remaining,
        )
        if draft_token_ids is None or draft_token_ids.shape[1] != remaining:
            return None
        return draft_token_ids
