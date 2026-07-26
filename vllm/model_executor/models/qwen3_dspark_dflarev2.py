# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlareV2 backbone + DSpark Markov head for speculative decoding.

Combines:
  * DFlareV2 context path (shared FC + per-layer fusion residual)
  * DSpark sequential Markov bias (optional position-adaptive alpha)

Runtime dispatch uses ``method="dspark"``. Checkpoint config should set
``architectures=["Qwen3DSparkDFlareV2Model"]`` and ``model_arch="dflarev2"``.
"""

import os
from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.triton_utils import tl, triton

from .qwen3_dflarev2 import DFlareV2Qwen3Model
from .qwen3_dspark import (
    DSparkMarkovHead,
    HiddenStatesCorrection,
    HpcHiddenCorrectionMixin,
    Qwen3DSparkForCausalLM,
)
from .utils import AutoWeightsLoader, maybe_prefix, process_eagle_weight

logger = init_logger(__name__)


@triton.jit
def _fused_dual_rmsnorm_concat_kernel(
    hidden_ptr,
    embed_ptr,
    hidden_weight_ptr,
    embed_weight_ptr,
    output_ptr,
    hidden_row_stride,
    embed_row_stride,
    hidden_size: tl.constexpr,
    eps: tl.constexpr,
    block_size: tl.constexpr,
):
    """Normalize the two correction inputs and concatenate in one kernel."""
    row = tl.program_id(0)
    offsets = tl.arange(0, block_size)
    mask = offsets < hidden_size
    hidden_offset = row * hidden_row_stride + offsets
    embed_offset = row * embed_row_stride + offsets
    hidden = tl.load(hidden_ptr + hidden_offset, mask=mask, other=0.0).to(tl.float32)
    embed = tl.load(embed_ptr + embed_offset, mask=mask, other=0.0).to(tl.float32)
    hidden_scale = tl.rsqrt(tl.sum(hidden * hidden, axis=0) / hidden_size + eps)
    embed_scale = tl.rsqrt(tl.sum(embed * embed, axis=0) / hidden_size + eps)
    hidden_weight = tl.load(hidden_weight_ptr + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    embed_weight = tl.load(embed_weight_ptr + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    output_offset = row * (2 * hidden_size) + offsets
    tl.store(
        output_ptr + output_offset,
        hidden * hidden_scale * hidden_weight,
        mask=mask,
    )
    tl.store(
        output_ptr + output_offset + hidden_size,
        embed * embed_scale * embed_weight,
        mask=mask,
    )


class NoOpMarkovHead(nn.Module):
    """DSpark-compatible zero transition bias for correction-only checkpoints."""

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return token_ids.new_empty((*token_ids.shape, 0), dtype=torch.float32)

    def bias(
        self,
        markov_embed: torch.Tensor,
        logits_processor,
        step: int | None = None,
    ) -> torch.Tensor:
        del logits_processor, step
        return markov_embed.new_zeros(())


class LowRankHiddenStatesCorrection(HpcHiddenCorrectionMixin, nn.Module):
    """Low-rank residual correction used by ``hidden_correction_type=low_rank``.

    The exported checkpoint contains ``down_proj[rank, 2 * hidden_size]`` and
    ``up_proj[hidden_size, rank]`` rather than the SwiGLU correction's
    gate/up/down triplet.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        rms_norm_eps: float,
        prefix: str,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.use_fused_kernel = os.getenv("VLLM_DSPARK_FUSED_CORRECTION", "1") != "0"
        self.hidden_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.embed_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.down_proj = ColumnParallelLinear(
            2 * hidden_size,
            intermediate_size,
            bias=False,
            prefix=maybe_prefix(prefix, "down_proj"),
        )
        self.up_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            prefix=maybe_prefix(prefix, "up_proj"),
        )
        # This low-rank path already has a stride-aware fused dual-RMSNorm +
        # concat kernel, which is faster than two HPC calls. Keep it as the
        # default while allowing explicit HPC A/B tests via the environment.
        self._init_hpc_correction(default_enabled=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        prev_token_embeds: torch.Tensor,
    ) -> torch.Tensor:
        prev_token_embeds = prev_token_embeds.to(hidden_states.dtype)
        if self._use_hpc_correction:
            normalized = torch.cat(
                [
                    self._norm_with_hpc(self.hidden_norm, hidden_states),
                    self._norm_with_hpc(self.embed_norm, prev_token_embeds),
                ],
                dim=-1,
            )
        elif hidden_states.is_cuda and self.use_fused_kernel:
            rows = hidden_states.numel() // self.hidden_size
            hidden = hidden_states.view(rows, self.hidden_size)
            embeds = prev_token_embeds.view(rows, self.hidden_size)
            normalized = torch.empty(
                rows,
                2 * self.hidden_size,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            _fused_dual_rmsnorm_concat_kernel[(rows,)](
                hidden,
                embeds,
                self.hidden_norm.weight,
                self.embed_norm.weight,
                normalized,
                hidden.stride(0),
                embeds.stride(0),
                hidden_size=self.hidden_size,
                eps=self.hidden_norm.variance_epsilon,
                block_size=triton.next_power_of_2(self.hidden_size),
            )
        else:
            normalized = torch.cat(
                [
                    self.hidden_norm(hidden_states),
                    self.embed_norm(prev_token_embeds),
                ],
                dim=-1,
            )
        low_rank, _ = self.down_proj(normalized)
        correction, _ = self.up_proj(F.silu(low_rank))
        return self._residual_add_with_hpc(hidden_states, correction)


class Qwen3DSparkDFlareV2Model(DFlareV2Qwen3Model):
    """DFlareV2 backbone with DSpark Markov head."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            start_layer_id=start_layer_id,
            prefix=prefix,
        )
        config = self.config
        draft_vocab_size = (
            getattr(config, "draft_vocab_size", None) or config.vocab_size
        )
        markov_rank = int(getattr(config, "markov_rank", 0))
        if markov_rank > 0:
            self.markov_head = DSparkMarkovHead(
                config.vocab_size,
                draft_vocab_size,
                markov_rank,
                prefix=maybe_prefix(prefix, "markov_head"),
                pos_adaptive=bool(getattr(config, "markov_pos_adaptive", False)),
                block_size=getattr(config, "block_size", None),
                alpha_max=float(getattr(config, "markov_alpha_max", 1.0)),
                replicated=bool(getattr(config, "markov_head_replicated", False)),
            )
        else:
            self.markov_head = NoOpMarkovHead()

        self.enable_hidden_correction = bool(
            getattr(config, "enable_hidden_correction", False)
        )
        self.hidden_correction = None
        if self.enable_hidden_correction:
            correction_type = getattr(config, "hidden_correction_type", "swiglu")
            intermediate_config = getattr(
                config, "hidden_correction_intermediate_size", None
            )
            if correction_type == "low_rank":
                intermediate = int(intermediate_config or 0)
                if intermediate <= 0:
                    raise ValueError(
                        "low-rank hidden correction requires a positive "
                        "hidden_correction_intermediate_size."
                    )
                self.hidden_correction = LowRankHiddenStatesCorrection(
                    hidden_size=int(config.hidden_size),
                    intermediate_size=intermediate,
                    rms_norm_eps=float(getattr(config, "rms_norm_eps", 1e-6)),
                    prefix=maybe_prefix(prefix, "hidden_correction"),
                )
            elif correction_type == "swiglu":
                intermediate = int(intermediate_config or config.hidden_size)
                self.hidden_correction = HiddenStatesCorrection(
                    hidden_size=int(config.hidden_size),
                    embed_size=int(config.hidden_size),
                    intermediate_size=intermediate,
                    rms_norm_eps=float(getattr(config, "rms_norm_eps", 1e-6)),
                    prefix=maybe_prefix(prefix, "hidden_correction"),
                )
            else:
                raise NotImplementedError(
                    "Unsupported hidden correction type "
                    f"{correction_type!r}; expected 'low_rank' or 'swiglu'."
                )


class Qwen3DSparkDFlareV2ForCausalLM(Qwen3DSparkForCausalLM):
    """Top-level DFlareV2 + DSpark draft model."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        self.draft_model_config = vllm_config.speculative_config.draft_model_config
        self.config = self.draft_model_config.hf_config
        if getattr(self.config, "draft_vocab_size", None) is None:
            self.config.draft_vocab_size = getattr(self.config, "vocab_size", None)
        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.model = Qwen3DSparkDFlareV2Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            start_layer_id=target_layer_num,
        )

        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.lm_head = ParallelLMHead(
            self.config.draft_vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(
            self.config.draft_vocab_size, scale=logit_scale
        )
        target_vocab_size = vllm_config.model_config.get_vocab_size()
        if self.config.draft_vocab_size != target_vocab_size:
            self.draft_id_to_target_id = nn.Parameter(
                torch.zeros(self.config.draft_vocab_size, dtype=torch.long),
                requires_grad=False,
            )
        else:
            self.draft_id_to_target_id = None

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Keep the uncollapsed T*D aux hidden states for DFlareV2 fusion.
        return self.model.combine_hidden_states(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """Load DFlareV2 + DSpark training export weights.

        Remaps:
          context_proj -> fc
          context_norm -> hidden_norm
          final_norm   -> norm
        """
        model_weights: dict[str, torch.Tensor] = {}
        includes_draft_id_mapping = False
        includes_embed_tokens = False
        includes_lm_head = False

        for name, loaded_weight in weights:
            if "t2d" in name or "confidence_head" in name:
                continue
            if "d2t" in name:
                name = name.replace("d2t", "draft_id_to_target_id")
                includes_draft_id_mapping = True
            elif "lm_head" not in name:
                name = "model." + name

            if name == "model.context_proj.weight":
                name = "model.fc.weight"
            elif name == "model.context_norm.weight":
                name = "model.hidden_norm.weight"
            elif name == "model.final_norm.weight":
                name = "model.norm.weight"

            if "embed_tokens" in name:
                includes_embed_tokens = True
            if "lm_head" in name and "markov" not in name:
                includes_lm_head = True

            model_weights[name] = loaded_weight
            process_eagle_weight(self, name)

        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_param_names: set[str] = set()
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        remaining_weights: list[tuple[str, torch.Tensor]] = []

        for name, loaded_weight in model_weights.items():
            if "scale" in name:
                remapped = maybe_remap_kv_scale_name(name, params_dict)
                if remapped is None:
                    continue
                name = remapped

            if "attention_sink_bias" in name:
                if name not in params_dict:
                    continue
                param = params_dict[name]
                heads_per_rank = loaded_weight.shape[0] // tp_size
                head_start = tp_rank * heads_per_rank
                narrow = loaded_weight.narrow(0, head_start, heads_per_rank)
                param.data.copy_(narrow)
                loaded_param_names.add(name)
                continue

            matched = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                mapped = name.replace(weight_name, param_name)
                if mapped not in params_dict:
                    continue
                param = params_dict[mapped]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_param_names.add(mapped)
                matched = True
                break
            if matched:
                continue

            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_param_names.add(name)
            else:
                remaining_weights.append((name, loaded_weight))

        skip_substrs = ["mask_embedding", "confidence_head"]
        if not includes_embed_tokens:
            skip_substrs.append("embed_tokens")
        if not includes_lm_head:
            skip_substrs.append("lm_head")
        if not includes_draft_id_mapping:
            skip_substrs.append("draft_id_to_target_id")

        if remaining_weights:
            loader = AutoWeightsLoader(self, skip_substrs=skip_substrs)
            loader.load_weights(iter(remaining_weights))

        # Ensure required params were actually populated.
        missing = [
            name
            for name, param in params_dict.items()
            if name not in loaded_param_names
            and not any(s in name for s in skip_substrs)
            and param.numel() > 0
        ]
        # Buffers like _fusion_probs are filled after load.
        missing = [n for n in missing if "_fusion_probs" not in n]
        if missing:
            logger.warning(
                "DFlareV2+DSpark load: %d params not filled from checkpoint "
                "(may be shared later): %s",
                len(missing),
                missing[:12],
            )

        self.model._build_fused_kv_buffers()


__all__ = [
    "LowRankHiddenStatesCorrection",
    "Qwen3DSparkDFlareV2Model",
    "Qwen3DSparkDFlareV2ForCausalLM",
]
