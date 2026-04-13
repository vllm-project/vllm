# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GradientRunner — core gradient computation for attribution.

Computes d/d(embeddings) of loss and/or per-token log-probs for a
prompt + target pair. Enables token-level attribution ("how much did
each input token influence each output token's log-probability?").

Key design decisions:
  - Uses @torch.enable_grad() to override the outer inference_mode
    context that wraps normal vLLM inference.
  - Embeddings are detached and cloned so they become leaf tensors
    that accumulate .grad without affecting the model's own parameters.
  - Per-token gradients require one backward pass per selected target
    token; retain_graph is used only when more passes follow.
  - Uses torch.autograd.grad instead of .backward() to avoid manual
    gradient zeroing and cloning overhead.
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.gradient_params import GradientParams
from vllm.model_executor.models.interfaces_base import (
    VllmModelForTextGeneration,
    is_text_generation_model,
)


@dataclass
class GradientRunnerOutput:
    """Raw output from the gradient runner (GPU tensors / Python scalars).

    This is converted to GradientOutput (CPU lists) by the caller before
    being sent over the wire.

    Attributes:
        token_log_probs: Per-target-token log p(y_t | x, y_{<t}).
        token_attributions: Attribution matrix.
            If aggregated: [num_selected_targets, num_grad_tokens]
            If full: [num_selected_targets, num_grad_tokens, hidden_dim]
        loss: Scalar loss value.
        loss_gradients: dict mapping gradient target name to gradient tensor.
    """

    token_log_probs: list[float] | None = None
    token_attributions: torch.Tensor | None = None
    loss: float | None = None
    loss_gradients: dict[str, torch.Tensor] = field(default_factory=dict)


class GradientRunner:
    """Computes gradients for a causal LM given prompt + target tokens.

    Runs a forward pass with gradient tracking enabled (overriding
    the outer torch.inference_mode context), computes a loss or per-token
    log-probs, calls backward, and extracts gradients w.r.t. the
    requested targets.
    """

    def __init__(self, model: nn.Module):
        if not is_text_generation_model(model):
            raise TypeError(
                "GradientRunner requires a text generation model that "
                "implements embed_input_ids() and compute_logits(). "
                f"Got: {type(model).__name__}"
            )
        self.model: VllmModelForTextGeneration = model  # type: ignore[assignment]

    @torch.enable_grad()
    def compute_gradients(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        gradient_params: GradientParams,
    ) -> GradientRunnerOutput:
        """Compute gradients for the given input/target pair.

        Args:
            input_ids: Prompt token IDs, shape [num_input_tokens].
            target_ids: Target continuation token IDs, shape [num_target_tokens].
            gradient_params: Configuration for what/how to compute.

        Returns:
            GradientRunnerOutput with requested gradient information.
        """
        device = input_ids.device
        num_input = input_ids.shape[0]
        num_target = target_ids.shape[0]

        if num_input < 1:
            raise ValueError(
                "input_ids must have at least 1 token for gradient computation"
            )

        num_total = num_input + num_target

        want_input_grad = "input_embeddings" in gradient_params.gradient_targets
        want_output_grad = "output_embeddings" in gradient_params.gradient_targets

        # --- Build the full sequence: [input | target] ---
        full_ids = torch.cat([input_ids, target_ids])
        positions = torch.arange(num_total, device=device)

        # --- Embed with gradient tracking ---
        # We detach and re-enable grad so the embedding lookup itself
        # is not part of the graph, but the embeddings tensor IS a leaf
        # that accumulates gradients.
        with torch.no_grad():
            all_embeds = self.model.embed_input_ids(full_ids)

        input_embeds = (
            all_embeds[:num_input].detach().clone().requires_grad_(want_input_grad)
        )
        output_embeds = (
            all_embeds[num_input:].detach().clone().requires_grad_(want_output_grad)
        )
        del all_embeds  # Free original embeddings

        inputs_embeds = torch.cat([input_embeds, output_embeds], dim=0)

        # Build list of tensors we need gradients for.
        grad_targets = []
        grad_target_names = []
        if want_input_grad:
            grad_targets.append(input_embeds)
            grad_target_names.append("input_embeddings")
        if want_output_grad:
            grad_targets.append(output_embeds)
            grad_target_names.append("output_embeddings")

        # --- Forward pass (builds computation graph) ---
        # Concrete model classes accept intermediate_tensors and inputs_embeds
        # but the VllmModel Protocol only declares (input_ids, positions).
        hidden_states = self.model.forward(  # type: ignore[call-arg]
            input_ids=None,
            positions=positions,
            intermediate_tensors=None,
            inputs_embeds=inputs_embeds,
        )
        logits = self.model.compute_logits(hidden_states)
        del hidden_states  # Free intermediate activations

        assert logits is not None, (
            "compute_logits returned None (likely not last PP rank)"
        )
        # logits: [num_total, vocab_size]

        # --- Compute per-token log-probs for target positions ---
        # Position i predicts token at position i+1.
        # So target token 0 (at position num_input) is predicted by
        # logits at position num_input - 1.
        predict_start = num_input - 1
        predict_end = num_input + num_target - 1
        target_logits = logits[predict_start:predict_end]  # [num_target, V]
        del logits  # Free full logits

        log_probs = target_logits.log_softmax(dim=-1)

        # Gather log-prob of each actual target token
        token_log_probs = log_probs.gather(
            dim=-1, index=target_ids.unsqueeze(-1)
        ).squeeze(-1)  # [num_target]
        del log_probs

        result = GradientRunnerOutput()

        if gradient_params.return_log_probs:
            result.token_log_probs = token_log_probs.detach().tolist()

        # --- Per-token log-prob gradients (token-level attribution) ---
        if gradient_params.gradient_of in ("token_log_probs", "both"):
            indices = gradient_params.target_token_indices
            if indices is None:
                indices = list(range(num_target))

            per_token_grads = []
            for i, t in enumerate(indices):
                is_last = i == len(indices) - 1
                # Only retain the graph if we need more backward passes
                # (more token indices, or a loss backward to follow).
                need_graph = not is_last or (gradient_params.gradient_of == "both")

                # torch.autograd.grad returns gradients directly, avoiding
                # the overhead of .backward() + manual .grad zeroing/cloning.
                grads = torch.autograd.grad(
                    token_log_probs[t],
                    grad_targets,
                    retain_graph=need_graph,
                )

                # Collect and aggregate gradients from all targets.
                token_grad = self._aggregate_grad_targets(
                    grads,
                    gradient_params.aggregation,
                )
                per_token_grads.append(token_grad)

            # Stack: [num_selected_targets, num_grad_tokens, ...]
            result.token_attributions = torch.stack(per_token_grads, dim=0)

        # --- Loss gradients (single backward) ---
        if gradient_params.gradient_of in ("loss", "both"):
            if gradient_params.loss_function == "cross_entropy":
                loss = F.cross_entropy(target_logits, target_ids)
            else:
                # log_prob_sum: negative log-likelihood
                loss = -token_log_probs.sum()

            grads = torch.autograd.grad(loss, grad_targets)
            result.loss = loss.item()

            for name, grad in zip(grad_target_names, grads):
                result.loss_gradients[name] = self._aggregate(
                    grad, gradient_params.aggregation
                )

        return result

    @staticmethod
    def _aggregate_grad_targets(
        grads: tuple[torch.Tensor, ...],
        aggregation: str,
    ) -> torch.Tensor:
        """Concatenate gradients from multiple targets and aggregate."""
        grad = torch.cat(grads, dim=0) if len(grads) > 1 else grads[0]
        return GradientRunner._aggregate(grad, aggregation)

    @staticmethod
    def _aggregate(
        grad: torch.Tensor,
        aggregation: str,
    ) -> torch.Tensor:
        """Reduce [num_tokens, hidden_dim] → [num_tokens] or keep full.

        Args:
            grad: Gradient tensor of shape [num_tokens, hidden_dim].
            aggregation: "none", "l2_norm", or "abs_sum".

        Returns:
            If aggregation is "none": [num_tokens, hidden_dim]
            Otherwise: [num_tokens]
        """
        if aggregation == "l2_norm":
            return grad.norm(dim=-1)
        elif aggregation == "abs_sum":
            return grad.abs().sum(dim=-1)
        elif aggregation == "none":
            return grad
        else:
            raise ValueError(f"Unknown aggregation mode: {aggregation!r}")
