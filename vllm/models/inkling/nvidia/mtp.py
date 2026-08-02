# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling MTP (Multi-Token Prediction) draft model (NVIDIA).

Implements the first MTP depth from the reference ``mtp_model.py`` shipped with
the checkpoint. It owns ``hidden_norm`` / ``embed_norm`` RMSNorms, a ``2H -> H``
input projection, and a full Inkling transformer block with a dense bf16 MLP.

The draft shares the target's token embedding table and LM head
(``load_eagle_model`` wires those references) and applies the backbone
``embed_norm`` on top: the depth layers were trained on the same normed
embeddings the backbone consumes (their own ``embed_norm`` weights are
near-identity trims, unlike the backbone's whitening ``embed_norm``).
"""

from __future__ import annotations

from collections.abc import Iterable

import regex as re
import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import maybe_prefix
from vllm.sequence import IntermediateTensors

from ..configs import InklingModelConfig
from .layernorm import InklingRMSNorm
from .model import InklingDecoderLayer, InklingReplicatedEmbedding
from .ops.norm import embed_dual_rmsnorm_cat, embed_rmsnorm

# Checkpoint attention projections (wq_du/wk_dv/wv_dv/wr_du) -> fused qkvr.
# Mirrors the backbone's hf_to_vllm_mapper.orig_to_new_stacked; kept as a
# local (pname, wname, shard) list since the MTP loader remaps by hand.
_ATTENTION_PARAMS_MAPPING = [
    ("qkvr", "wq_du", 0),
    ("qkvr", "wk_dv", 1),
    ("qkvr", "wv_dv", 2),
    ("qkvr", "wr_du", 3),
]


def _mtp_depth_from_name(name: str) -> int | None:
    m = re.search(r"\.mtp\.layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


class InklingMTPDepthLayer(nn.Module):
    """One MTP depth: norm both inputs, fuse (2H->H), run a Inkling block."""

    def __init__(self, config: InklingModelConfig, prefix: str, is_local: bool) -> None:
        super().__init__()
        self.hidden_norm = InklingRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.embed_norm = InklingRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_proj = ReplicatedLinear(
            config.hidden_size * 2,
            config.hidden_size,
            bias=False,
            return_bias=False,
            prefix=f"{prefix}.input_proj",
        )
        # A force-dense-MLP bf16 block; ``is_local`` selects sliding-window vs
        # full attention (the swa_* head config and sliding_window window) to
        # match this depth's checkpoint transformer_block weights.
        self.transformer_block = InklingDecoderLayer(
            config,
            layer_id=0,
            is_local=is_local,
            quant_config=None,
            prefix=f"{prefix}.transformer_block",
            force_dense_mlp=True,
        )

    def forward(self, combined: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # ``combined`` is the fused-normed [rmsnorm(hidden) | embed_norm(emb)]
        # input, built by InklingMultiTokenPredictor.fused_input_cat in one launch.
        hidden = self.input_proj(combined)
        # The short conv self-fetches its paged SWA-cache metadata from the
        # forward context (via its conv_owner prefix); no conv_meta to thread.
        return self.transformer_block(positions, hidden)


class InklingMultiTokenPredictor(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        config: InklingModelConfig = (
            vllm_config.speculative_config.draft_model_config.hf_config
        )
        self.config = config
        if vllm_config.speculative_config.num_speculative_tokens != 1:
            raise ValueError(
                "Inkling MTP currently supports exactly one speculative token"
            )
        self.chain_hidden_post_norm = config.chain_hidden_post_norm
        local_ids = set(config.local_layer_ids)
        self.layers = nn.ModuleDict(
            {"0": InklingMTPDepthLayer(config, f"{prefix}.layers.0", 0 in local_ids)}
        )
        self.chain_norm = (
            InklingRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if self.chain_hidden_post_norm
            else None
        )
        # The target's raw token embedding (pre embed_norm), attached by
        # load_eagle_model. Never materialized here: building our own
        # replicated copy would transiently double the 2.3 GiB table.
        self.embed_tokens: InklingReplicatedEmbedding = None  # type: ignore[assignment]
        # The depth layers consume the *backbone-normed* embedding
        # (embed_norm(embed(ids))), not the raw one: mtp embed_norm weights
        # are near-identity (trained on already-normalized inputs), and
        # feeding raw embeddings drops MTP1 acceptance from ~0.85 to ~0.70.
        # Weight loaded from the target's embed_norm.weight; gated like the
        # target's InklingModel.embed_norm.
        self.backbone_embed_norm = (
            InklingRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.use_embed_norm
            else None
        )

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: object | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Draft-prefill embedding: fused gather + backbone embed_norm, then
        the target's tower embeddings scattered in unnormed (the backbone
        convention — MM embeds are merged after embed_norm)."""
        norm = self.backbone_embed_norm
        embeds = embed_rmsnorm(
            input_ids,
            self.embed_tokens.weight,
            norm.weight if norm is not None else None,
            norm.variance_epsilon if norm is not None else 0.0,
        )
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:  # type: ignore[arg-type]
            return embeds
        from vllm.model_executor.models.utils import _merge_multimodal_embeddings

        assert is_multimodal is not None
        return _merge_multimodal_embeddings(
            inputs_embeds=embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    def fused_input_cat(
        self,
        layer: InklingMTPDepthLayer,
        previous_hidden: torch.Tensor,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor | None,
    ) -> torch.Tensor:
        """The depth layer's [rmsnorm(hidden) | embed_norm(embed)] input in one
        launch: embedding row gather + the backbone embed_norm + the depth
        embed_norm chain on one side, hidden_norm on the other, written
        straight into the cat buffer."""
        hidden_w = layer.hidden_norm.weight
        embed_w = layer.embed_norm.weight
        eps = layer.hidden_norm.variance_epsilon
        if inputs_embeds is not None:
            # Draft prefill with target-merged MM embeddings (already
            # backbone-normed via embed_input_ids); only the depth embed_norm
            # remains.
            return embed_dual_rmsnorm_cat(
                previous_hidden, hidden_w, embed_w, eps, embeds=inputs_embeds
            )
        return embed_dual_rmsnorm_cat(
            previous_hidden,
            hidden_w,
            embed_w,
            eps,
            input_ids=input_ids,
            embed_table=self.embed_tokens.weight,
            pre_norm_weight=(
                self.backbone_embed_norm.weight
                if self.backbone_embed_norm is not None
                else None
            ),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        # The draft's short conv is a paged SWA-cache layer (its conv_owner is
        # auto-enumerated as a draft attention layer); its per-token metadata is
        # built by the speculator's build_attn_metadata and read from the
        # forward context, so nothing extra is threaded here.
        if spec_step_idx != 0:
            raise ValueError("Inkling MTP only supports spec_step_idx=0")
        layer = self.layers["0"]
        combined = self.fused_input_cat(
            layer, previous_hidden_states, input_ids, inputs_embeds
        )
        hidden = layer(combined, positions)
        if self.chain_norm is not None:
            hidden = self.chain_norm(hidden)
        return hidden


class InklingMTP(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        config: InklingModelConfig = (
            vllm_config.speculative_config.draft_model_config.hf_config
        )
        self.config = config
        self.model = InklingMultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        # The target's (vocab-sharded) LM head, attached by load_eagle_model;
        # never materialized here (same reasoning as model.embed_tokens).
        self.lm_head: ParallelLMHead = None  # type: ignore[assignment]
        self.logits_processor = LogitsProcessor(
            config.padded_vocab_size,
            org_vocab_size=config.vocab_size,
            soft_cap=config.final_logit_softcapping,
        )
        self._logits_zero: torch.Tensor | None = None

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: object | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model.embed_input_ids(
            input_ids, multimodal_embeddings, is_multimodal=is_multimodal
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        return self.model(
            input_ids,
            positions,
            hidden_states,
            inputs_embeds,
            spec_step_idx,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        # The MTP shares the base model's LM head, which is trained on
        # ``hidden / mup``-scaled inputs, so apply the same mup scaling here
        # for a matching logit scale — folded into the lm_head GEMM alpha
        # (fp32 epilogue) like the target's compute_logits. (Argmax-invariant
        # for greedy draft sampling, but it matters for the gumbel sampling
        # distribution at temperature > 0.)
        mup = self.config.logits_mup_width_multiplier
        if not mup:
            return self.logits_processor(self.lm_head, hidden_states)
        assert self.logits_processor.soft_cap is None
        assert self.logits_processor.scale == 1.0
        w = self.lm_head.weight
        if self._logits_zero is None:
            self._logits_zero = w.new_zeros(1)
        logits = torch.addmm(
            self._logits_zero,
            hidden_states,
            w.t(),
            beta=0.0,
            alpha=1.0 / mup,
        )
        logits = self.logits_processor._gather_logits(logits)
        if logits is not None:
            logits = logits[..., : self.logits_processor.org_vocab_size]
        return logits

    def get_top_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Greedy draft tokens via rank-local argmax + tiny (value, index)
        reduction — no full-vocab logits all-gather. The muP divisor is a
        positive scalar, so the argmax is invariant and the scaling is
        skipped entirely."""
        return self.logits_processor.get_top_tokens(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return _load_inkling_mtp_weights(self, weights)


def _load_inkling_mtp_weights(
    module: InklingMTP,
    weights: Iterable[tuple[str, torch.Tensor]],
) -> set[str]:
    """Load ``model.mtp.*`` weights into the MTP module.

    Checkpoint keys look like ``model.mtp.chain_norm.weight`` and
    ``model.mtp.layers.{i}.{...}``. The transformer block reuses the backbone
    layer's fused-projection layout, so we apply the same qkvr / gate_up / down
    remapping as ``_load_inkling_weights``. Token embedding and LM head are shared
    (provided by ``load_eagle_model``) and are not present in mtp.safetensors.
    """
    # Per-depth attention is full or sliding-window (config.local_layer_ids);
    # each depth's qkvr MergedColumnParallelLinear is built with the matching
    # (swa_)num_key_value_heads, and its weight_loader handles the TP sharding.
    # The sconv SWA cache pins tp_size <= num_key_value_heads, so tp never
    # exceeds a layer's kv-head count and no GQA K/V replication is needed here.
    params = dict(module.named_parameters())
    loaded: set[str] = set()

    def _load(name: str, weight: torch.Tensor, shard_id: object = None) -> bool:
        param = params.get(name)
        if param is None:
            return False
        loader = getattr(param, "weight_loader", default_weight_loader)
        if shard_id is None:
            if loader is default_weight_loader or param.shape == weight.shape:
                default_weight_loader(param, weight)
            else:
                loader(param, weight)
        else:
            loader(param, weight, shard_id)  # type: ignore[call-arg]
        loaded.add(name)
        return True

    for name, weight in weights:
        depth = _mtp_depth_from_name(name)
        # Token embedding and LM head are never materialized on the draft
        # (no params to load into); load_eagle_model attaches the target's.
        if name in ("model.llm.embed.weight", "model.llm.unembed.weight"):
            continue
        # The backbone embed_norm, applied to the shared embedding before the
        # depth layers (see InklingMultiTokenPredictor.embed_input_ids). The
        # per-depth mtp.layers.{i}.embed_norm keys carry ".mtp." and are loaded
        # below. Only the shared backbone key routes here.
        if name == "model.llm.embed_norm.weight":
            _load("model.backbone_embed_norm.weight", weight)
            continue
        # Only consume the MTP weights; everything else belongs to the target.
        if ".mtp." not in name:
            continue
        # Only the first checkpoint depth is used for MTP=1.
        if depth is not None and depth != 0:
            continue
        # model.mtp.chain_norm.weight -> model.chain_norm.weight
        # model.mtp.layers.{i}.X      -> model.layers.{i}.X
        original_name = name
        name = name.replace(".mtp.layers.", ".layers.").replace(
            ".mtp.chain_norm.", ".chain_norm."
        )

        if ".chain_norm." in name and module.model.chain_norm is None:
            raise ValueError(
                "Inkling checkpoint contains chain_norm weights but "
                "chain_hidden_post_norm is disabled."
            )

        # Fused attention qkvr (wq_du/wk_dv/wv_dv/wr_du -> qkvr).
        matched = False
        for pname, wname, shard in _ATTENTION_PARAMS_MAPPING:
            if f".attn.{wname}." in name:
                mapped_name = name.replace(f".{wname}.", f".{pname}.")
                if not _load(mapped_name, weight, shard):
                    raise ValueError(f"Unexpected Inkling MTP weight: {original_name}")
                matched = True
                break
        if matched:
            continue

        # Dense MLP fused gate/up + down.
        if ".mlp.w13_dn.weight" in name:
            loaded_weight = _load(name.replace(".w13_dn.", ".gate_up_proj."), weight)
        elif ".mlp.w2_md.weight" in name:
            loaded_weight = _load(name.replace(".w2_md.", ".down_proj."), weight)
        else:
            if name.endswith(".bias") and name not in params:
                continue
            loaded_weight = _load(name, weight)
        if not loaded_weight:
            raise ValueError(f"Unexpected Inkling MTP weight: {original_name}")
    required = {
        name
        for name in params
        if name.startswith("model.layers.") or name.startswith("model.chain_norm.")
    }
    if missing := sorted(required - loaded):
        raise ValueError(
            "Inkling MTP checkpoint is missing required parameters: "
            + ", ".join(missing)
        )
    return loaded


EntryClass = [InklingMTP]
