# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling model implementation for NVIDIA GPUs."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import regex as re
import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_reduce_scatter,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsLoRA,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.models.inkling.common.mm_preprocess import (
    InklingDummyInputsBuilder,
    InklingMultiModalProcessor,
    InklingProcessingInfo,
    inkling_audio_enabled,
    inkling_vision_enabled,
)
from vllm.models.inkling.common.towers import InklingAudio, InklingVision
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors

from ..configs import InklingMMConfig, InklingModelConfig
from .attention import InklingAttention, compute_log_scaling_tau
from .layernorm import InklingRMSNorm
from .logits_processor import InklingLogitsProcessor
from .mlp import InklingDenseMLP
from .moe import InklingMoE
from .ops.lamport import get_lamport_rs_conv, initialize_lamport_rs_conv
from .ops.norm import add_rmsnorm, embed_rmsnorm
from .sconv_swa_attn import _ATTN, _MLP, InklingConvState, InklingSconvMetadata
from .short_conv import InklingShortConv


def _layer_id(name: str) -> int | None:
    m = re.search(r"\.layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def _sconv_add_norm(
    delta: torch.Tensor,
    hidden: torch.Tensor,
    sconv: InklingShortConv,
    norm: InklingRMSNorm | None,
    positions: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """``h = hidden + sconv(TP-sum(delta)); y = rmsnorm(h)``.

    The Lamport path performs reduce-scatter + shard sconv + all-gather +
    residual add + norm. The NCCL path handles unsupported configurations."""
    attn_metadata = get_forward_context().attn_metadata
    m = (
        attn_metadata.get(sconv.owner.prefix)
        if isinstance(attn_metadata, dict)
        else None
    )
    cache = sconv.owner.kv_cache
    off_s, ws = sconv.owner.stream_ranges[sconv.stream_idx]
    norm_w = norm.weight if norm is not None else None
    eps = norm.variance_epsilon if norm is not None else 0.0

    mm = get_lamport_rs_conv(hidden.shape[-1], sconv.kernel_size)
    if mm is not None and mm.usable(delta.shape[0]) and m is not None:
        assert cache.numel() > 0
        assert isinstance(m, InklingSconvMetadata)
        return mm.rs_sconv_ag_add_norm(
            delta,
            hidden,
            sconv.weight.squeeze(1),
            norm_w,
            eps,
            cache,
            positions,
            m.block_table,
            m.seq_idx,
            m.slot_mapping,
            off_s,
            ws,
            sconv.owner.block_size,
        )

    # Fallback: NCCL RS -> shard sconv -> AG -> fused add(+rmsnorm).
    shard = tensor_model_parallel_reduce_scatter(delta, dim=-1)
    shard = sconv(shard.contiguous(), positions)
    full = tensor_model_parallel_all_gather(shard, dim=-1)
    if norm is None:
        return None, hidden + full
    return add_rmsnorm(hidden, full, norm_w, eps)


class InklingDecoderLayer(nn.Module):
    def __init__(
        self,
        config: InklingModelConfig,
        layer_id: int,
        is_local: bool,
        quant_config: QuantizationConfig | None,
        prefix: str,
        force_dense_mlp: bool = False,
    ) -> None:
        super().__init__()
        # Per-layer owner of the conv state as a paged SWA cache. The 4 sconv
        # streams (K/V/attn/mlp) are packed head-major into one block and share
        # it. Built first so the attention layer can wire its K/V sconv to it.
        self.conv_state = InklingConvState(
            num_kv_heads=(
                config.swa_num_key_value_heads
                if is_local
                else config.num_key_value_heads
            ),
            head_dim=config.swa_head_dim if is_local else config.head_dim,
            hidden_size=config.hidden_size,
            kernel_size=config.sconv_kernel_size,
            prefix=f"{prefix}.conv_state",
        )
        self.attn_norm = InklingRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = InklingAttention(
            config,
            num_heads=(
                config.swa_num_attention_heads
                if is_local
                else config.num_attention_heads
            ),
            num_kv_heads=(
                config.swa_num_key_value_heads
                if is_local
                else config.num_key_value_heads
            ),
            head_dim=config.swa_head_dim if is_local else config.head_dim,
            rel_extent=config.rel_extent,
            local_extent=config.sliding_window_size,
            is_local=is_local,
            prefix=f"{prefix}.attn",
            quant_config=quant_config,
            conv_owner=self.conv_state,
        )
        self.mlp_norm = InklingRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if force_dense_mlp or layer_id < config.dense_mlp_idx:
            self.mlp: nn.Module = InklingDenseMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.dense_intermediate_size,
                use_global_scale=config.use_global_scale,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = InklingMoE(
                config,
                prefix=f"{prefix}.mlp",
                quant_config=quant_config,
            )

        # Short convolution on the attention-output and MLP-output residual
        # streams, hidden-sharded: the sublayer outputs are reduce-scattered
        # to [T, H/tp], the sconv runs on the shard, and an all-gather
        # restores the full residual — all fused with the residual add + next
        # rmsnorm via the Lamport P2P kernels for decode-sized batches.
        tp_size = get_tensor_model_parallel_world_size()
        sconv_dim = config.hidden_size // tp_size
        self.attn_sconv = InklingShortConv(
            sconv_dim, config.sconv_kernel_size, owner=self.conv_state, stream_idx=_ATTN
        )
        self.mlp_sconv = InklingShortConv(
            sconv_dim, config.sconv_kernel_size, owner=self.conv_state, stream_idx=_MLP
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        pending: tuple[torch.Tensor | None, InklingShortConv] | None = None,
        defer_mlp_add: bool = False,
        attn_in: torch.Tensor | None = None,
        log_scaling: torch.Tensor | None = None,
    ) -> (
        torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor | None, InklingShortConv]]
    ):
        # The previous sublayer's (pre-reduce, pre-sconv) delta is folded in
        # fused with its RS/sconv/AG and this layer's pre-attention rmsnorm.
        # A None delta means the partials sit in the NVLS symm buffer.
        if pending is None:
            if attn_in is None:
                # First layer; on the text path attn_norm comes fused with
                # the embedding gather (chain_weight in embed_rmsnorm).
                attn_in = self.attn_norm(hidden_states)
        else:
            attn_in, hidden_states = _sconv_add_norm(
                pending[0], hidden_states, pending[1], self.attn_norm, positions
            )
        attn_output = self.attn(positions, attn_in, log_scaling)
        mlp_in, hidden_states = _sconv_add_norm(
            attn_output, hidden_states, self.attn_sconv, self.mlp_norm, positions
        )
        mlp_output = self.mlp(mlp_in)
        if defer_mlp_add:
            # Caller folds mlp_output (pre-reduce, pre-sconv) into the next
            # fused sconv+add+rmsnorm.
            return hidden_states, (mlp_output, self.mlp_sconv)
        return _sconv_add_norm(
            mlp_output, hidden_states, self.mlp_sconv, None, positions
        )[1]


class InklingReplicatedEmbedding(nn.Module):
    """Full-vocab embedding table replicated on every TP rank.

    Trades the full table per rank (~2.3 GiB at V=201k / H=6144 bf16, vs a
    1/tp shard) for no masked lookup or per-lookup TP all-reduce, and keeps the
    full table on-rank for the fused gather+norm kernel. Bit-exact vs
    vocab-parallel: the all-reduce there only ever summed one real row against
    exact zeros. The LM head stays vocab-sharded.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, dtype=torch.get_default_dtype()),
            requires_grad=False,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return embed_rmsnorm(input_ids, self.weight, None, 0.0)


class InklingModel(nn.Module):
    def __init__(
        self,
        *,
        config: InklingModelConfig,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = InklingReplicatedEmbedding(
            config.padded_vocab_size, config.hidden_size
        )
        self.embed_norm = (
            InklingRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.use_embed_norm
            else None
        )
        local_ids = set(config.local_layer_ids)

        def get_layer(prefix: str) -> InklingDecoderLayer:
            idx = _layer_id(prefix + ".") or int(prefix.split(".")[-1])
            return InklingDecoderLayer(
                config, idx, idx in local_ids, quant_config, prefix
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
        self.norm = InklingRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Row gather + embed_norm in one launch.
        norm = self.embed_norm
        return embed_rmsnorm(
            input_ids,
            self.embed_tokens.weight,
            norm.weight if norm is not None else None,
            norm.variance_epsilon if norm is not None else 0.0,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        attn_in0: torch.Tensor | None = None
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                # embed_norm was already applied when producing inputs_embeds.
                hidden_states = inputs_embeds
            else:
                # Gather + embed_norm + the first layer's attn_norm, one launch.
                norm = self.embed_norm
                hidden_states, attn_in0 = embed_rmsnorm(
                    input_ids,
                    self.embed_tokens.weight,
                    norm.weight if norm is not None else None,
                    self.config.rms_norm_eps,
                    chain_weight=self.layers[self.start_layer].attn_norm.weight,
                )
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        log_scaling = None
        if self.config.log_scaling_n_floor is not None:
            log_scaling = compute_log_scaling_tau(
                positions,
                self.config.log_scaling_n_floor,
                self.config.log_scaling_alpha,
            )

        pending: tuple[torch.Tensor | None, InklingShortConv] | None = None
        for layer in self.layers[self.start_layer : self.end_layer]:
            hidden_states, pending = layer(
                positions,
                hidden_states,
                pending=pending,
                defer_mlp_add=True,
                attn_in=attn_in0,
                log_scaling=log_scaling,
            )
            attn_in0 = None

        if not get_pp_group().is_last_rank:
            if pending is not None:
                hidden_states = _sconv_add_norm(
                    pending[0], hidden_states, pending[1], None, positions
                )[1]
            return IntermediateTensors({"hidden_states": hidden_states})
        if pending is not None:
            # Final RS/sconv/AG + residual add fused with the final rmsnorm.
            norm_out = _sconv_add_norm(
                pending[0], hidden_states, pending[1], self.norm, positions
            )[0]
            assert norm_out is not None
            return norm_out
        return self.norm(hidden_states)


class _TmlForCausalLMBase(nn.Module, SupportsPP, SupportsLoRA):
    """Shared text-backbone causal-LM scaffolding for both entry classes."""

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={
            ".w13_dn": ".gate_up_proj",
            ".w2_md": ".down_proj",
        },
        orig_to_new_stacked={
            ".attn.wq_du.": (".attn.qkvr.", 0),
            ".attn.wk_dv.": (".attn.qkvr.", 1),
            ".attn.wv_dv.": (".attn.qkvr.", 2),
            ".attn.wr_du.": (".attn.qkvr.", 3),
        },
        orig_to_new_prefix={
            "model.llm.layers.": "model.layers.",
            "model.llm.embed_norm": "model.embed_norm",
            "model.llm.embed": "model.embed_tokens",
            "model.llm.norm": "model.norm",
            "model.llm.unembed": "lm_head",
            "language_model.layers.": "model.layers.",
            "language_model.lm_head.": "lm_head.",
        },
        orig_to_new_suffix={
            # NVFP4 scale
            ".w13_weight.scale": ".w13_weight_scale",
            ".w13_weight.scale2": ".w13_weight_scale_2",
            ".w2_weight.scale": ".w2_weight_scale",
            ".w2_weight.scale2": ".w2_weight_scale_2",
        },
    )
    packed_modules_mapping = {
        "qkvr": ["wq_du", "wk_dv", "wv_dv", "wr_du"],
        "w13": ["w1", "w3"],
    }
    embedding_modules = {
        "lm_head": "output_embeddings",
    }

    def _build(
        self,
        vllm_config: VllmConfig,
        text_config: InklingModelConfig,
        prefix: str,
    ) -> None:
        quant_config = vllm_config.quant_config
        self.config = text_config
        # Read by the MRV2 runner to publish per-request short-conv metadata.
        # Short convolution is intrinsic to Inkling, so this is always set.
        self.uses_sconv = True
        self.model = InklingModel(
            config=text_config,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        initialize_lamport_rs_conv(
            text_config.hidden_size,
            text_config.sconv_kernel_size,
            vllm_config.scheduler_config.max_num_batched_tokens,
        )
        self.lm_head = ParallelLMHead(
            text_config.padded_vocab_size,
            text_config.hidden_size,
            org_num_embeddings=text_config.padded_vocab_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = InklingLogitsProcessor(
            text_config.padded_vocab_size,
            org_vocab_size=text_config.vocab_size,
            soft_cap=text_config.final_logit_softcapping,
            logits_mup_width_multiplier=text_config.logits_mup_width_multiplier,
        )
        self.make_empty_intermediate_tensors = (  # type: ignore[method-assign]
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return _load_inkling_weights(self, weights, self.config)


class InklingForCausalLM(_TmlForCausalLMBase):
    """Text-only entry point (``inkling_model`` checkpoints)."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self._build(vllm_config, vllm_config.model_config.hf_config, prefix)


@MULTIMODAL_REGISTRY.register_processor(
    InklingMultiModalProcessor,
    info=InklingProcessingInfo,
    dummy_inputs=InklingDummyInputsBuilder,
)
class InklingForConditionalGeneration(_TmlForCausalLMBase, SupportsMultiModal):
    """Top-level (multimodal) entry point.

    Builds the vision + audio towers on top of the shared text backbone. Inkling has
    NO cross-modal fusion (the vision tower emits one token per patch, the audio
    tower one token per frame), so generation reuses the inherited backbone
    ``forward`` / ``compute_logits`` (the latter already applies muP) and this
    class only adds multimodal embedding + merge.
    """

    hf_to_vllm_mapper = _TmlForCausalLMBase.hf_to_vllm_mapper | WeightsMapper(
        orig_to_new_prefix={
            "model.audio.": "audio.",
            "model.visual.": "visual.vision_encoder.",
        },
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<|content_image|>"
        if modality.startswith("audio"):
            return "<|content_audio_input|>"
        raise ValueError("Only image or audio modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: InklingMMConfig = vllm_config.model_config.hf_config

        self.visual = (
            InklingVision(config.vision_config, prefix=maybe_prefix(prefix, "visual"))
            if inkling_vision_enabled(config)
            else None
        )
        self.audio = (
            InklingAudio(config.audio_config, prefix=maybe_prefix(prefix, "audio"))
            if inkling_audio_enabled(config)
            else None
        )

        self._build(vllm_config, config.text_config, prefix)

    # -- multimodal embedding -------------------------------------------

    def _process_image_input(
        self, pixel_values: Any, num_patches: Any
    ) -> tuple[torch.Tensor, ...]:
        assert self.visual is not None
        # pixel_values is a list (per item) of [P_i, 2, P, P, 3] tensors,
        # or a single concatenated tensor. Normalize to a flat batch, run the
        # tower once, then split back per item.
        if isinstance(pixel_values, (list, tuple)):
            if not pixel_values:
                return ()
            sizes = [int(p.shape[0]) for p in pixel_values]
            patches = torch.cat(list(pixel_values), dim=0)
        else:
            patches = pixel_values
            sizes = self._sizes_from(num_patches, patches.shape[0])

        patches = patches.to(device=self.visual.device, dtype=self.visual.dtype)
        embeds = self.visual(patches)  # [total_patches, D]
        return tuple(embeds.split(sizes))

    def _process_audio_input(
        self, input_audio_features: Any, num_audio_tokens: Any
    ) -> tuple[torch.Tensor, ...]:
        assert self.audio is not None
        if isinstance(input_audio_features, (list, tuple)):
            if not input_audio_features:
                return ()
            sizes = [int(d.shape[0]) for d in input_audio_features]
            dmel = torch.cat(list(input_audio_features), dim=0)
        else:
            dmel = input_audio_features
            sizes = self._sizes_from(num_audio_tokens, dmel.shape[0])

        dmel = dmel.to(device=self.audio.device)
        embeds = self.audio(dmel)  # [total_frames, D]
        return tuple(embeds.split(sizes))

    @staticmethod
    def _sizes_from(counts: Any, total: int) -> list[int]:
        if counts is None:
            return [total]
        if isinstance(counts, torch.Tensor):
            return [int(c) for c in counts.flatten().tolist()]
        if isinstance(counts, (list, tuple)):
            flat: list[int] = []
            for c in counts:
                flat.append(int(c.item()) if isinstance(c, torch.Tensor) else int(c))
            return flat
        return [int(counts)]

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        # Iterate modalities in a stable order so the returned per-item tensors
        # line up with their appearance order; the positional merge in
        # embed_input_ids handles actual placement.
        pixel_values = kwargs.get("pixel_values")
        num_patches = kwargs.get("num_patches")
        input_audio_features = kwargs.get("input_audio_features")
        num_audio_tokens = kwargs.get("num_audio_tokens")

        embeddings: tuple[torch.Tensor, ...] = ()
        if pixel_values is not None and self.visual is not None:
            embeddings += self._process_image_input(pixel_values, num_patches)
        if input_audio_features is not None and self.audio is not None:
            embeddings += self._process_audio_input(
                input_audio_features, num_audio_tokens
            )
        return embeddings

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Override the base's 1-arg embed_input_ids: the runner calls this 3-arg
        # signature for multimodal models. Text embeddings come from the shared
        # backbone (which applies embed_norm); MM embeddings are scattered in.
        from vllm.model_executor.models.utils import _merge_multimodal_embeddings

        # Placeholder ids use unused vocabulary slots and these positions are
        # overwritten by MM embeds below.
        inputs_embeds = self.model.embed_input_ids(input_ids)
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds
        assert is_multimodal is not None
        return _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    def get_language_model(self) -> nn.Module:
        # This class IS the causal LM (the towers are side branches), so the
        # language model is self — callers expect a module exposing ``.model``
        # and ``.lm_head``.
        return self


# ===========================================================================
# Weight loading
# ===========================================================================


_MOE_EXPERT_WEIGHT_RE = re.compile(
    r"^(?P<mlp>.*\.mlp)\.(?P<rest>(?:shared_)?experts\..+)$"
)


def _load_inkling_weights(
    module: nn.Module,
    weights: Iterable[tuple[str, torch.Tensor]],
    config: InklingModelConfig,
) -> set[str]:
    moe_modules = {
        name: mod for name, mod in module.named_modules() if isinstance(mod, InklingMoE)
    }
    loaded: set[str] = set()
    tp_size = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()
    local_ids = set(config.local_layer_ids)

    def _iter_loadable_weights() -> Iterable[tuple[str, torch.Tensor]]:
        for name, weight in module.hf_to_vllm_mapper.apply(weights):
            shard_id = getattr(weight, "shard_id", None)
            # Replicate K/V conv-free GQA heads when tp_size > num_kv_heads.
            if (
                shard_id in (1, 2)
                and name.endswith(".attn.qkvr.weight")
                and weight.shape[0] > 0
            ):
                lid = _layer_id(name)
                if lid is not None:
                    is_local = lid in local_ids
                    n_kv = (
                        config.swa_num_key_value_heads
                        if is_local
                        else config.num_key_value_heads
                    )
                    head_dim = config.swa_head_dim if is_local else config.head_dim
                    if tp_size > n_kv and weight.shape[0] == n_kv * head_dim:
                        kv_idx = (tp_rank * n_kv) // tp_size
                        weight = weight.narrow(0, kv_idx * head_dim, head_dim)
                        weight.shard_id = shard_id

            # MoE expert tensors (fused stacked, routed + shared sink): translate
            # the checkpoint layout to per-expert FusedMoE loads.
            moe_match = _MOE_EXPERT_WEIGHT_RE.match(name)
            if moe_match is not None and moe_match.group("mlp") in moe_modules:
                moe = moe_modules[moe_match.group("mlp")]
                for rel in moe.load_expert_weight(moe_match.group("rest"), weight):
                    loaded.add(f"{moe_match.group('mlp')}.{rel}")
                continue

            yield name, weight

    # The release checkpoint also carries auxiliary prediction-head weights;
    # they are not part of the causal LM served by this implementation.
    loader = AutoWeightsLoader(module, skip_prefixes=["model.mtp."])
    loaded |= loader.load_weights(_iter_loadable_weights())

    # Post-load MoE fixups (default input scales, zeroed EP-padding experts).
    for moe_name, moe in moe_modules.items():
        for rel in moe.finalize_load():
            loaded.add(f"{moe_name}.{rel}")
    return loaded


EntryClass = [InklingForCausalLM, InklingForConditionalGeneration]
