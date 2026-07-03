# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-Token Prediction (MTP) draft head for IquestMoeV13.

First-version, low-cost integration: only the FIRST MTP layer
(``mtp_layers.0``) of the checkpoint is loaded and used. The second MTP
layer (``mtp_layers.1``) is intentionally discarded. This lets the head
plug directly into vLLM v1's EagleProposer single-layer autoregressive
drafting loop, with no changes to the spec-decode core loop.

Note: this reuses the same vLLM MTP proposer path, but is NOT equivalent to
DeepSeek-V3's multi-layer MTP semantics. A multi-layer MTP would rotate to a
different layer per step (spec_step_idx % num_mtp_layers); here every draft
step reuses mtp_layers.0. In particular, K=2 does NOT use the training-time
second layer that predicts the +2 position — the 2nd draft token is produced
by autoregressively reusing the first layer.

Structurally the head mirrors ``deepseek_mtp.py``:
    emb  = enorm(embed(input_ids))
    h    = hnorm(previous_hidden_states)
    x    = eh_proj(cat([emb, h]))
    out  = mtp_model_layer(x)              # one decoder block
    out  = final_layernorm(out)            # returned + fed to shared lm_head

Differences from DeepSeek MTP:
  * the inner decoder block uses Iquest sink attention and a
    sandwich-norm + MoE structure (``attention_norm`` / ``attn_out_norm`` /
    ``feed_forward_norm`` / ``ffn_out_norm``), matching the megatron
    MTP layer spec (``layer_number == 1``, so ``first_layer_*_scale``);
  * logits use the **shared main ``lm_head``** — the checkpoint has no
    per-layer ``shared_head`` — and ``final_layernorm`` is already applied
    to the returned hidden, so ``compute_logits`` adds no extra norm.
"""

from collections.abc import Iterable

import torch
import torch.nn as nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors

from .iquest_moe_v13 import (
    IquestMoEBlock,
    IquestMoeAttention,
    IquestMoeRMSNorm,
)
from .utils import is_pp_missing_parameter, maybe_prefix

logger = init_logger(__name__)


class IquestMoeV13MTPInnerLayer(nn.Module):
    """The MTP transformer block (``mtp_model_layer`` in the checkpoint).

    A hybrid matching NEITHER main-stack shape exactly: sandwich-norm structure
    like main-stack layer 0 (all four norms attention_norm/attn_out_norm/
    feed_forward_norm/ffn_out_norm plus first_layer_* out-scales), but an MoE
    MLP like the deeper layers -- whereas main-stack layer 0 is a DENSE MLP
    (mlp_only_layers == [0]). This sandwich-norm + MoE combination is dictated
    by the megatron MTP layer spec (a non-dense TransformerLayer at
    layer_number == 1). Verified against iter_0024000: mtp_layers.0
    .mtp_model_layer has all four norms and an mlp.router + mlp.experts MoE.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.speculative_config.draft_model_config.hf_config
        quant_config = vllm_config.quant_config
        self.hidden_size = config.hidden_size

        self.self_attn = IquestMoeAttention(
            vllm_config=vllm_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = IquestMoEBlock(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.attention_norm = IquestMoeRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.attn_out_norm = IquestMoeRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.feed_forward_norm = IquestMoeRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.ffn_out_norm = IquestMoeRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        # megatron MTP inner layer runs as layer_number == 1 -> first_layer scales
        self.attn_out_scale = getattr(config, "first_layer_attn_out_scale", 1.0)
        self.ffn_out_scale = getattr(config, "first_layer_ffn_out_scale", 1.0)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # sandwich-norm forward (same norm/residual/scale wiring as
        # IquestMoeDecoderLayer's layer-0 path, but with an MoE MLP here)
        norm_hidden_states = self.attention_norm(hidden_states)
        attn_output = self.self_attn(
            positions=positions, hidden_states=norm_hidden_states
        )
        h = hidden_states + self.attn_out_norm(attn_output) * self.attn_out_scale
        ffn_out = self.mlp(self.feed_forward_norm(h))
        output = h + self.ffn_out_norm(ffn_out) * self.ffn_out_scale
        return output


class IquestMoeV13MTPLayer(nn.Module):
    """One MTP module: enorm/hnorm + eh_proj + inner decoder block + final LN."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.speculative_config.draft_model_config.hf_config

        self.enorm = IquestMoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = IquestMoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(
            config.hidden_size * 2, config.hidden_size, bias=False
        )
        self.mtp_model_layer = IquestMoeV13MTPInnerLayer(
            vllm_config=vllm_config, prefix=f"{prefix}.mtp_model_layer"
        )
        self.final_layernorm = IquestMoeRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor,
    ) -> torch.Tensor:
        # Mask position-0 embedding: the roll-by-one boundary has no valid
        # target token there (mirrors deepseek_mtp).
        inputs_embeds = torch.where(
            positions.unsqueeze(-1) == 0, 0, inputs_embeds
        )
        inputs_embeds = self.enorm(inputs_embeds)
        previous_hidden_states = self.hnorm(previous_hidden_states)
        hidden_states = self.eh_proj(
            torch.cat([inputs_embeds, previous_hidden_states], dim=-1)
        )
        hidden_states = self.mtp_model_layer(positions, hidden_states)
        return self.final_layernorm(hidden_states)


class IquestMoeV13MultiTokenPredictor(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.speculative_config.draft_model_config.hf_config
        # Only the first MTP layer is materialised (low-cost v1). Use a layer
        # index of num_hidden_layers so the inner Attention registers a KV-cache
        # name distinct from the target model's layers.
        #
        # KNOWN CONSTRAINT: IquestMoeAttention derives its layer index from this
        # prefix and uses it ONLY for hybrid/sliding-window and shared-KV
        # strategy selection. Sink attention is independent of the layer index
        # and IS supported here: enable_sink_attention=true on the current
        # checkpoint, and mtp_layers.0...self_attn.sink_k is loaded into this
        # layer normally. The index (== num_hidden_layers) sits one past the
        # main stack, so it is only safe while the backbone has hybrid layers
        # and shared-KV disabled (current checkpoint: use_hybrid_layers unset,
        # shared_kv_num_layers=0). If a future same-architecture checkpoint
        # enables either of those two, the MTP attention could mis-select a
        # strategy or index out of range; generalizing then needs an explicit
        # is_mtp_layer flag (or layer_idx_override/cache_prefix) on
        # IquestMoeAttention so the MTP layer skips backbone layer-index
        # policies while keeping a unique KV-cache name.
        self.mtp_layer_idx = config.num_hidden_layers
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.layers = nn.ModuleDict(
            {
                str(self.mtp_layer_idx): IquestMoeV13MTPLayer(
                    vllm_config=vllm_config,
                    prefix=f"{prefix}.layers.{self.mtp_layer_idx}",
                )
            }
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            assert input_ids is not None, (
                "IquestMoeV13 MTP requires input_ids when inputs_embeds is None"
            )
            inputs_embeds = self.embed_tokens(input_ids)
        # Single-layer (low-cost v1): every spec step reuses the same layer.
        return self.layers[str(self.mtp_layer_idx)](
            positions, previous_hidden_states, inputs_embeds
        )


@support_torch_compile
class IquestMoeV13MTP(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        quant_config = vllm_config.quant_config
        self.model = IquestMoeV13MultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        # Shared main lm_head (the checkpoint has no per-layer shared_head).
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

        # Make the single-layer behavior explicit at runtime: unlike a true
        # multi-layer MTP, K>1 here reuses mtp_layers.0 autoregressively and
        # mtp_layers.1 is NOT loaded, so later draft positions accept less.
        spec_config = vllm_config.speculative_config
        num_spec = getattr(spec_config, "num_speculative_tokens", 1) or 1
        if num_spec > 1:
            logger.warning_once(
                "IquestMoeV13 MTP uses only the first MTP layer (mtp_layers.0). "
                "num_speculative_tokens=%s reuses this single layer "
                "autoregressively; mtp_layers.1 is NOT loaded, so this is not "
                "the training-time multi-layer MTP and acceptance drops for "
                "draft positions beyond the first.",
                num_spec,
            )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        return self.model(
            input_ids, positions, hidden_states, inputs_embeds, spec_step_idx
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        # final_layernorm is already applied inside the MTP layer, so feed the
        # hidden straight into the shared lm_head.
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load only the shared embedding, the shared lm_head, and the first
        MTP layer (``mtp_layers.0.*``). Everything else in the checkpoint —
        the main transformer stack, ``model.norm``, and ``mtp_layers.1.*`` —
        is skipped.
        """
        mtp_prefix = "mtp_layers.0."
        dst_layer_prefix = f"model.layers.{self.model.mtp_layer_idx}."

        # (fused_param, ckpt_component, shard_id). Matched against the trailing
        # ".<component>.weight" suffix — NOT an arbitrary substring — so a path
        # that merely contains e.g. "q_proj" elsewhere cannot be mis-rewritten.
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        enable_sink_attention = getattr(self.config, "enable_sink_attention", False)

        def _load_into(param_name: str, weight: torch.Tensor) -> None:
            param = params_dict[param_name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, weight)
            loaded_params.add(param_name)

        for name, loaded_weight in weights:
            # Shared weights loaded into the draft head.
            if name in ("model.embed_tokens.weight", "lm_head.weight"):
                _load_into(name, loaded_weight)
                continue

            # Only the first MTP layer; drop mtp_layers.1.* and all main weights.
            if not name.startswith(mtp_prefix):
                continue

            # Rewrite mtp_layers.0.<rest> -> model.layers.<idx>.<rest>
            name = dst_layer_prefix + name[len(mtp_prefix):]

            # Fused QKV: match on the exact ".<component>.weight" suffix.
            for param_name, weight_name, shard_id in stacked_params_mapping:
                suffix = f".{weight_name}.weight"
                if not name.endswith(suffix):
                    continue
                mapped = name[: -len(suffix)] + f".{param_name}.weight"
                if mapped not in params_dict:
                    continue
                param = params_dict[mapped]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(mapped)
                break
            else:
                # Sonic-MoE fused expert tensors: match exact ".experts.fc" /
                # ".experts.proj" component suffixes.
                if name.endswith(".mlp.experts.fc"):
                    mapped = name[: -len(".fc")] + ".w13_weight"
                    param = params_dict[mapped]
                    weight_loader = param.weight_loader
                    for expert_id in range(self.config.num_experts):
                        weight_loader(
                            param,
                            loaded_weight[expert_id][: self.config.intermediate_size],
                            mapped,
                            shard_id="w1",
                            expert_id=expert_id,
                        )
                        weight_loader(
                            param,
                            loaded_weight[expert_id][self.config.intermediate_size:],
                            mapped,
                            shard_id="w3",
                            expert_id=expert_id,
                        )
                    loaded_params.add(mapped)
                    continue
                if name.endswith(".mlp.experts.proj"):
                    mapped = name[: -len(".proj")] + ".w2_weight"
                    param = params_dict[mapped]
                    weight_loader = param.weight_loader
                    for expert_id in range(self.config.num_experts):
                        weight_loader(
                            param,
                            loaded_weight[expert_id],
                            mapped,
                            shard_id="w2",
                            expert_id=expert_id,
                        )
                    loaded_params.add(mapped)
                    continue

                if not enable_sink_attention and name.endswith(".sink_k"):
                    logger.warning_once("sink attention feature is disabled")
                    continue

                # Sonic-MoE router naming: ".mlp.router.weight" -> ".mlp.gate.weight".
                if name.endswith(".mlp.router.weight"):
                    name = name[: -len(".router.weight")] + ".gate.weight"

                if is_pp_missing_parameter(name, self):
                    continue
                if name not in params_dict:
                    logger.warning_once("Unexpected MTP weight skipped: %s", name)
                    continue
                _load_into(name, loaded_weight)

        # Defensive check: EVERY MTP-layer parameter must have been loaded. A
        # missing mtp_layers.0.* prefix, or an individual weight whose name
        # changed and got silently skipped ("Unexpected MTP weight"), would
        # leave the draft partly random-initialized with poor acceptance and no
        # hard error. All MTP params live under this prefix; the shared
        # embed_tokens/lm_head are excluded (different prefix). Attention scale
        # buffers and sinks_k/sinks_v are buffers, not parameters, so they never
        # appear in params_dict. Be strict: this draft layer is small.
        lp = f"model.layers.{self.model.mtp_layer_idx}."
        missing = sorted(
            n
            for n in params_dict
            if n.startswith(lp)
            and n not in loaded_params
            and not is_pp_missing_parameter(n, self)
        )
        if missing:
            raise ValueError(
                "IquestMoeV13 MTP draft layer is not fully loaded: "
                + str(len(missing))
                + " parameter(s) under '"
                + lp
                + "' were not populated from the checkpoint (e.g. "
                + str(missing[:5])
                + "). This usually means the checkpoint lacks 'mtp_layers.0.*' "
                "or a weight name changed. Ensure the checkpoint contains the "
                "first MTP layer before enabling the iquest_mtp speculative "
                "method."
            )
        return loaded_params
