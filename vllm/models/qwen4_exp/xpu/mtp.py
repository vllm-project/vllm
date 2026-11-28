# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Qwen4-Exp MTP (multi-token-prediction) draft model.

Qwen4-Exp's MTP head is a *custom* hyper-connection design (not the stock
Qwen3Next MTP):

  * the fc stage takes the target's **wide** hyper-connection residual
    ``[T, hc, n_embd]`` (``hc = hc_count = 4``) plus the token embedding:
    ``pre_fc_norm_hidden`` RMS-normalizes the full ``hc * n_embd`` vector
    (the streams stay distinct), ``fc_hidden`` projects each stream, and the
    normalized, ``fc_embedding``-projected token embedding is added to every
    stream;
  * the resulting wide state is the residual of the single MTP decoder layer,
    a standard ``Qwen4ExpDecoderLayer`` (full attention + hyper-connections +
    MoE);
  * the final output norm is the top-level ``hyper_connection_mixer``
    (a ``HyperConnection`` with ``has_inject=False``).

All MTP weights live under the ``mtp.`` prefix. They follow the checkpoint's
quantization config like the target: in the FP8 checkpoint only the MTP routed
experts are FP8, the rest stays BF16. The MTP layer's QSA indexer is ignored
(attention runs dense on XPU). The speculative decoding framework replaces the
draft's token embedding and LM head with the target's after loading.

The target model exposes the wide residual via ``get_mtp_target_hidden_states``
(see ``Qwen4ExpModel``), mirroring the DeepSeek-V4 XPU MTP hook.
"""

from collections.abc import Iterable

import torch
from torch import nn
from transformers import Qwen4ExpTextConfig

from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    make_empty_intermediate_tensors_factory,
    maybe_prefix,
)
from vllm.models.qwen4_exp.xpu.model import (
    HyperConnection,
    Qwen4ExpDecoderLayer,
    Qwen4ExpMixtureOfExperts,
    Qwen4ExpModel,
)
from vllm.sequence import IntermediateTensors

logger = init_logger(__name__)


class _WideRMSNorm(nn.Module):
    """RMSNorm over the full ``hc * n_embd`` wide state, keeping the streams.

    Equivalent to the amd/nvidia
    ``pre_fc_norm_hidden(hidden.flatten(-2)).view(T, hc, H)`` (GemmaRMSNorm,
    gamma = 1 + w), matching the checkpoint's ``pre_fc_norm_hidden`` (shape
    ``[hc * n_embd]``).
    """

    def __init__(self, hc: int, n_embd: int, eps: float):
        super().__init__()
        self.hc = hc
        self.n_embd = n_embd
        self.eps = eps
        # gamma is stored flat as [hc * n_embd]; loaded from the checkpoint.
        self.weight = nn.Parameter(torch.empty(hc * n_embd), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [T, hc, n_embd]
        t = x.shape[0]
        flat = x.reshape(t, self.hc * self.n_embd)
        var = flat.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        xn = flat * torch.rsqrt(var + self.eps).to(flat.dtype)
        gamma = (1.0 + self.weight.float()).to(xn.dtype)
        return (xn * gamma).view(t, self.hc, self.n_embd)


class Qwen4ExpMultiTokenPredictor(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        model_config = vllm_config.model_config

        config: Qwen4ExpTextConfig = model_config.hf_text_config
        self.config = config

        self.vocab_size = config.vocab_size
        self.hc = config.hc_count
        self.n_embd = config.hidden_size
        self.num_mtp_layers = getattr(config, "mtp_num_hidden_layers", 1)

        # Shared with the target model (loaded from the same checkpoint).
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )

        # fc stage: two separate projections (BF16, unquantized).
        self.pre_fc_norm_embedding = HyperConnectionRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_fc_norm_hidden = _WideRMSNorm(
            self.hc, config.hidden_size, config.rms_norm_eps
        )
        self.fc_embedding = ReplicatedLinear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=None,  # MTP fc is BF16
            prefix=f"{prefix}.fc_embedding",
        )
        self.fc_hidden = ReplicatedLinear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
            return_bias=False,
            quant_config=None,  # MTP fc is BF16
            prefix=f"{prefix}.fc_hidden",
        )

        # The single MTP decoder layer (full attention + HC + MoE).
        self.layers = torch.nn.ModuleList(
            Qwen4ExpDecoderLayer(
                vllm_config,
                layer_type="full_attention",
                prefix=f"{prefix}.layers.{idx}",
            )
            for idx in range(self.num_mtp_layers)
        )

        # Top-level hyper-connection mixer acts as the final output norm.
        self.hyper_connection_mixer = HyperConnection(config, has_inject=False)

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def _fc_stage(
        self,
        inputs_embeds: torch.Tensor,
        wide_hidden: torch.Tensor,
    ) -> torch.Tensor:
        # inputs_embeds: [T, n_embd]; wide_hidden: [T, hc, n_embd]
        # residual_linear_shared fusion (matches amd/nvidia): normalize over
        # the full hc*n_embd vector, project each hc stream with the shared
        # fc_hidden, then add the projected embedding as a residual to every
        # stream. The hc streams stay DISTINCT -- mean-collapsing them here
        # discards the inter-stream information that hyper-connections exist
        # to carry, and measurably lowers MTP acceptance.
        e = self.fc_embedding(self.pre_fc_norm_embedding(inputs_embeds))
        h = self.pre_fc_norm_hidden(wide_hidden)
        h = self.fc_hidden(h)
        return e.unsqueeze(-2) + h  # [T, hc, n_embd]

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is None:
                inputs_embeds = self.embed_input_ids(input_ids)
            # hidden_states arrives flat [T, hc * n_embd]; reshape to wide.
            # Use the existing dim-0 symbol (shape[0]) rather than -1 so the
            # reshape reuses the input_ids/hidden_states batch symbol instead
            # of introducing a new one (which trips a torch dynamic-shape
            # guard: "Constraints violated (hidden_states.size()[0],
            # input_ids.size()[0])").
            wide_hidden = hidden_states.view(
                hidden_states.shape[0], self.hc, self.n_embd
            )
            # The fc stage already produces the wide state, so hand it to the
            # decoder layer as ``residual``; passing it as ``hidden_states``
            # with ``residual=None`` would make the layer broadcast one
            # stream into hc identical copies.
            residual = self._fc_stage(inputs_embeds, wide_hidden)
            # Not consumed by the layer when ``residual`` is not None.
            x = residual.mean(dim=1)
        else:
            assert intermediate_tensors is not None
            x = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        current_step_idx = spec_step_idx % self.num_mtp_layers
        mtp_layer = self.layers[current_step_idx]
        # The layer re-broadcasts x [T, n_embd] to the wide state when
        # residual is None, runs attention + MoE, and returns
        # (collapsed [T, n_embd], wide [T, hc, n_embd]).
        _collapsed, wide_out = mtp_layer(
            hidden_states=x,
            residual=residual,
            positions=positions,
        )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": _collapsed, "residual": wide_out}
            )

        # Final output norm = the top-level hyper-connection mixer.
        out, _ = self.hyper_connection_mixer.mix(wide_out)
        # Return (collapsed, wide-flat). The speculator sizes its
        # multi-step feedback buffer for the HC-widened residual, so the
        # feedback tensor must be the wide state; ``out`` stays the
        # single stream consumed by the LM head. Matches the two-stream
        # contract used by the amd/nvidia Qwen4Exp MTP.
        wide_flat = wide_out.reshape(wide_out.shape[0], self.hc * self.n_embd)
        return out, wide_flat

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # The MTP decoder layer is a stock Qwen4ExpDecoderLayer (fused
        # qkv_proj / gate_up_proj), but the checkpoint stores the separate
        # q/k/v and gate/up projections. Apply the main model's stacked
        # mapping so they fuse on load.
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=Qwen4ExpModel.hf_to_vllm_mapper)


class HyperConnectionRMSNorm(nn.Module):
    """RMSNorm (deviation gamma) for the MTP embedding pre-norm.

    The checkpoint stores ``pre_fc_norm_embedding`` as a deviation ``w``
    (negative mean); the true RMSNorm gamma is ``(1 + w)``, matching the
    model's ``HyperConnection`` convention and the GGUF converter.
    """

    def __init__(self, n_embd: int, eps: float):
        super().__init__()
        self.n_embd = n_embd
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(n_embd), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        xn = x * torch.rsqrt(var + self.eps).to(x.dtype)
        # Add 1 in fp32: in bf16, 1 + w loses most of a small w's precision.
        gamma = (1.0 + self.weight.float()).to(xn.dtype)
        return xn * gamma


class Qwen4ExpMTP(nn.Module, Qwen4ExpMixtureOfExperts):
    """MTP draft wrapper: predictor + LM head.

    MTP-specific params live under the ``mtp.`` prefix and follow the
    checkpoint's quantization config (FP8 routed experts in the FP8
    checkpoint). The LM head and token embedding are replaced by the target's
    after loading.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        self.quant_config = vllm_config.quant_config
        super().__init__()
        self.config = config
        self.model = Qwen4ExpMultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "mtp")
        )

        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.set_moe_parameters()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ):
        hidden_states = self.model(
            input_ids, positions, hidden_states, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # The drafter receives the *full* checkpoint stream (all 48 main
        # layers + the vision tower). Keep only what the draft model owns:
        #   * ``mtp.*`` -> remapped to ``model.*`` (the predictor)
        #   * shared ``embed_tokens`` (strip the VL-wrapper prefix)
        #   * shared ``lm_head``
        # Everything else (main layers, visual tower, PLE) is skipped, as is
        # the MTP layer's QSA indexer: the XPU backend runs dense attention.
        shared_weight_names = ("embed_tokens", "lm_head")

        def remap_weight_names(weights):
            for name, weight in weights:
                if ".indexer." in name:
                    continue
                if name.startswith("mtp."):
                    name = name.replace("mtp.", "model.")
                elif name.startswith("model.language_model.embed_tokens."):
                    name = name.replace("model.language_model.", "model.")
                elif not any(key in name for key in shared_weight_names):
                    continue
                yield name, weight

        loader = AutoWeightsLoader(self)
        return loader.load_weights(remap_weight_names(weights))


__all__ = [
    "Qwen4ExpMTP",
    "Qwen4ExpMultiTokenPredictor",
]
