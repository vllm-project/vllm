# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Qwen4-Exp MTP (multi-token-prediction) draft model.

Qwen4-Exp's MTP head is a *custom* hyper-connection design (not the stock
Qwen3Next MTP):

  * the fc stage takes the target's **wide** hyper-connection residual
    ``[T, hc, n_embd]`` (``hc = hc_count = 4``) plus the token embedding,
    applies two separate projections (``fc_embedding`` / ``fc_hidden``) and
    sums them;
  * ``pre_fc_norm_hidden`` is a grouped RMSNorm over the wide state
    (per-stream, gamma of size ``hc * n_embd``) that collapses the streams by
    their mean back to ``[T, n_embd]``;
  * the single MTP decoder layer is a standard ``Qwen4ExpDecoderLayer``
    (full attention + hyper-connections + MoE) and re-broadcasts the fc output
    to the wide state;
  * the final output norm is the top-level ``hyper_connection_mixer``
    (a ``HyperConnection`` with ``has_inject=False``).

All MTP weights live under the ``mtp.`` prefix in the checkpoint and are kept
BF16 (the quant ``ignore`` list matches ``re:.*mtp\\..*``), so the draft model
is entirely unquantized. The token embedding and LM head are shared with the
target model.

The target model exposes the wide residual via ``get_mtp_target_hidden_states``
(see ``Qwen4ExpModel``), mirroring the DeepSeek-V4 XPU MTP hook.
"""

from collections.abc import Iterable

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.models.qwen4_exp.xpu.model import (
    HyperConnection,
    Qwen4ExpDecoderLayer,
    Qwen4ExpModel,
    Qwen4ExpMixtureOfExperts,
)
from vllm.sequence import IntermediateTensors
from transformers import Qwen4ExpTextConfig

from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    make_empty_intermediate_tensors_factory,
    maybe_prefix,
)

logger = init_logger(__name__)


class _GroupedRMSNormCollapse(nn.Module):
    """Grouped RMSNorm over the wide hyper-connection state, then mean-collapse.

    Normalizes each of the ``hc`` streams independently (reducing over the
    ``n_embd`` dim), scales by a per-element gamma of size ``hc * n_embd``,
    and collapses the streams by their mean -> ``[T, n_embd]``. This mirrors
    ``HyperConnection._grouped_rms_norm`` + gamma + mean (without the low-rank
    gating), matching the checkpoint's ``pre_fc_norm_hidden`` (shape
    ``[hc * n_embd]``).
    """

    def __init__(self, hc: int, n_embd: int, eps: float):
        super().__init__()
        self.hc = hc
        self.n_embd = n_embd
        self.eps = eps
        # gamma is stored flat as [hc * n_embd]; loaded from the checkpoint.
        self.weight = nn.Parameter(
            torch.empty(hc * n_embd), requires_grad=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [T, hc, n_embd]
        var = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        xn = x * torch.rsqrt(var + self.eps).to(x.dtype)
        # The checkpoint stores a DEVIATION w (negative mean); the true RMSNorm
        # gamma is (1 + w), matching HyperConnection.mix / the GGUF converter.
        gamma = (1.0 + self.weight.float()).view(self.hc, self.n_embd).to(xn.dtype)
        xn = xn * gamma
        return xn.mean(dim=1)  # [T, n_embd]

    def forward_wide(self, x: torch.Tensor) -> torch.Tensor:
        """RMSNorm over the full ``hc * n_embd`` vector, keeping the wide state.

        Equivalent to the amd/nvidia
        ``pre_fc_norm_hidden(hidden.flatten(-2)).view(T, hc, H)``
        (GemmaRMSNorm, gamma = 1 + w). Unlike :meth:`forward`, the hc
        streams are NOT mean-collapsed.
        """
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
        quant_config = vllm_config.quant_config

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
        self.pre_fc_norm_hidden = _GroupedRMSNormCollapse(
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

        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size
            )
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
        h = self.pre_fc_norm_hidden.forward_wide(wide_hidden)
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

    _DEVIATION = True  # gamma = (1 + w)

    def __init__(self, n_embd: int, eps: float):
        super().__init__()
        self.n_embd = n_embd
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(n_embd), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        xn = x * torch.rsqrt(var + self.eps).to(x.dtype)
        gamma = self.weight if not self._DEVIATION else (1.0 + self.weight)
        return xn * gamma


class Qwen4ExpMTP(nn.Module, Qwen4ExpMixtureOfExperts):
    """MTP draft wrapper: predictor + shared LM head.

    All MTP-specific params live under the ``mtp.`` prefix so the quant
    ``ignore`` list (``re:.*mtp\\..*``) keeps them unquantized (BF16).
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        cache_config = vllm_config.cache_config
        if cache_config.mamba_cache_mode == "all":
            raise NotImplementedError(
                "Qwen4ExpMTP does not support 'all' prefix caching; use "
                "'--mamba-cache-mode=align'."
            )

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
        # Everything else (main layers, visual tower, PLE) is skipped.
        shared_weight_names = ("embed_tokens", "lm_head")

        def remap_weight_names(weights):
            for name, weight in weights:
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