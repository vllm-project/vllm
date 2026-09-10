# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import typing
from collections.abc import Callable, Iterable
from itertools import islice

import regex as re
import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.kernels.mhc.torch import (
    mhc_post_torch,
    mhc_pre_delayed_torch,
)
from vllm.model_executor.layers.fused_moe import (
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import (
    EagleModelMixin,
    MixtureOfExperts,
    SupportsEagle3,
    SupportsLoRA,
    SupportsPP,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    is_pp_missing_parameter,
    make_layers,
    maybe_prefix,
)
from vllm.models.common.ops.sequence_parallel import (
    sp_all_gather,
    sp_padding_mask,
    sp_reduce_scatter,
    sp_shard,
)
from vllm.models.deepseek_v4.amd.model import (
    DeepseekV4MoE as DeepseekV4MoEBase,
)
from vllm.models.deepseek_v4_1.amd.rocm import DeepseekV41ROCMAiterMLAAttention
from vllm.models.deepseek_v4_1.attention import DeepseekV4Attention
from vllm.sequence import IntermediateTensors
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.worker.ubatching import dbo_current_ubatch_id

from ..common.engram import Engram, EngramLayout, NgramHashState
from ..common.mm_preprocess import IMAGE_SENTINEL_BASE_ID, image_sentinel_mask

if typing.TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata

logger = init_logger(__name__)


class DeepseekV4MoE(DeepseekV4MoEBase):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        use_sequence_parallel: bool = False,
    ):
        config = vllm_config.model_config.hf_config
        n_routed_experts = config.n_routed_experts
        n_activated_experts = config.num_experts_per_tok
        if extract_layer_index(prefix) >= config.num_hidden_layers:
            n_routed_experts = (
                getattr(config, "dspark_n_routed_experts", 0) or n_routed_experts
            )
            n_activated_experts = (
                getattr(config, "dspark_num_experts_per_tok", 0) or n_activated_experts
            )
        # The reused ROCm V4 MoE reads expert counts from the HF config. Draft
        # layers use V4.1's smaller DSpark expert set, so temporarily expose
        # the per-layer counts while constructing the module, then restore the
        # shared config for the remaining backbone layers.
        old_routed = config.n_routed_experts
        old_activated = config.num_experts_per_tok
        old_hash_layers = getattr(config, "num_hash_layers", None)
        config.n_routed_experts = n_routed_experts
        config.num_experts_per_tok = n_activated_experts
        config.num_hash_layers = 0
        try:
            super().__init__(vllm_config, prefix=prefix)
        finally:
            config.n_routed_experts = old_routed
            config.num_experts_per_tok = old_activated
            if old_hash_layers is None:
                delattr(config, "num_hash_layers")
            else:
                config.num_hash_layers = old_hash_layers

        # The V4.1 checkpoint contains a second per-expert routing bias for
        # image sentinel tokens.  The reused ROCm V4 MoE predates this field,
        # so register it here and attach the same Parameter to the router,
        # matching the CUDA V4.1 implementation.  Text tokens continue to use
        # e_score_correction_bias.
        self.gate.bias_vl = None
        self.image_sentinel_lo = 0
        if getattr(config, "vision_n_layers", 0) > 0:
            self.gate.bias_vl = nn.Parameter(
                torch.empty(self.n_routed_experts, dtype=torch.float32),
                requires_grad=False,
            )
            self.image_sentinel_lo = IMAGE_SENTINEL_BASE_ID
            self.experts.router.bias_vl = self.gate.bias_vl
            self.experts.router.image_sentinel_lo = self.image_sentinel_lo
        self.use_mega_moe = False
        # Expose the MixtureOfExperts protocol fields expected by the V4.1
        # outer model.  The current ROCm runner has no redundant EPLB copies,
        # so logical and physical expert counts are identical.
        self.n_logical_experts = self.n_routed_experts
        self.n_physical_experts = self.n_routed_experts
        self.n_local_physical_experts = self.n_local_experts
        self.n_redundant_experts = 0


def _select_dsv4_attn_cls(vllm_config: VllmConfig) -> type[DeepseekV4Attention]:
    """Pick the ROCm sparse-MLA implementation for DeepSeek V4.1."""
    backend = vllm_config.attention_config.backend
    if backend is not None and (
        backend is not AttentionBackendEnum.ROCM_FLASHMLA_SPARSE_DSV4
    ):
        raise ValueError(
            f"{backend.name} is not supported for DeepSeek V4.1 on ROCm; "
            "use ROCM_FLASHMLA_SPARSE_DSV4."
        )
    return DeepseekV41ROCMAiterMLAAttention


def _use_sequence_parallel(vllm_config: VllmConfig) -> bool:
    # The ROCm fused-MoE path currently uses ordinary tensor/expert parallel.
    return False


class DeepseekV4DecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config,
        prefix,
        topk_indices_buffer: torch.Tensor | None = None,
        aux_stream_list: list[torch.cuda.Stream] | None = None,
        candidate_block_buffer: torch.Tensor | None = None,
        engram_layout: EngramLayout | None = None,
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.hidden_size = config.hidden_size
        self.use_sequence_parallel = _use_sequence_parallel(vllm_config)

        self.engram: Engram | None = None
        if engram_layout is not None:
            layer_id = extract_layer_index(prefix)
            if layer_id in engram_layout.layer_ids:
                self.engram = Engram(
                    config,
                    vllm_config.quant_config,
                    engram_layout,
                    engram_layout.layer_ids.index(layer_id),
                    use_sequence_parallel=self.use_sequence_parallel,
                    prefix=f"{prefix}.engram",
                )

        self.rms_norm_eps = config.rms_norm_eps
        self.attn = _select_dsv4_attn_cls(vllm_config)(
            vllm_config,
            prefix=f"{prefix}.attn",
            topk_indices_buffer=topk_indices_buffer,
            aux_stream_list=aux_stream_list,
            candidate_block_buffer=candidate_block_buffer,
        )
        if self.use_sequence_parallel:
            self.attn.wo_b.reduce_results = False
        self.ffn = DeepseekV4MoE(
            vllm_config,
            prefix=f"{prefix}.ffn",
            use_sequence_parallel=self.use_sequence_parallel,
        )

        self.attn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)
        self.ffn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.hc_post_alpha = 2.0
        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * self.hidden_size
        self.hc_attn_fn = nn.Parameter(
            torch.empty(
                (mix_hc, hc_dim),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_attn_fn_broadcast: torch.Tensor | None = None
        self.hc_ffn_fn = nn.Parameter(
            torch.empty(
                (mix_hc, hc_dim),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_attn_base = nn.Parameter(
            torch.empty(
                mix_hc,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_ffn_base = nn.Parameter(
            torch.empty(
                mix_hc,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_attn_scale = nn.Parameter(
            torch.empty(
                3,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        self.hc_ffn_scale = nn.Parameter(
            torch.empty(
                3,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )

    @staticmethod
    def _hc_collapse(x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
        """Reference hyper-connection collapse used by DSpark on ROCm."""
        return (pre_mix.unsqueeze(-1) * x.float()).sum(dim=1).to(x.dtype)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        pre_mix: torch.Tensor | None = None,
        post_mix: torch.Tensor | None = None,
        res_mix: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
        engram_hashes: torch.Tensor | None = None,
        engram_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # The reference collapses each sublayer's input with the *previous*
        # sublayer's pre-mix: attention uses the pre-mix carried in (identity
        # for the first layer), the FFN uses this layer's attention pre-mix.
        if residual is None:
            if x.dim() == 2:
                # First layer: the stream is the embedding broadcast to hc
                # copies and the identity pre-mix selects copy 0.
                assert self.hc_attn_fn_broadcast is not None
                residual = x.unsqueeze(1).expand(-1, self.hc_mult, -1).contiguous()
                post_mix, res_mix, x, attn_pre = mhc_pre_delayed_torch(
                    residual,
                    self.hc_attn_fn_broadcast,
                    self.hc_attn_scale,
                    self.hc_attn_base,
                    self.rms_norm_eps,
                    self.hc_eps,
                    self.hc_eps,
                    self.hc_post_alpha,
                    self.hc_sinkhorn_iters,
                    x=x,
                )
            else:
                residual = x
                post_mix, res_mix, x, attn_pre = mhc_pre_delayed_torch(
                    residual,
                    self.hc_attn_fn,
                    self.hc_attn_scale,
                    self.hc_attn_base,
                    self.rms_norm_eps,
                    self.hc_eps,
                    self.hc_eps,
                    self.hc_post_alpha,
                    self.hc_sinkhorn_iters,
                    pre_mix=pre_mix,
                )
        else:
            residual = mhc_post_torch(x, residual, post_mix, res_mix)
            if self.engram is not None and engram_hashes is not None:
                # Engram injection happens between the previous sublayer's
                # post and this block's pre, on the full hc stream, so the
                # mix coefficients see the injected stream.
                residual = self.engram(
                    residual,
                    engram_hashes[:, self.engram.layer_hash_index],
                    engram_mask,
                )
            post_mix, res_mix, x, attn_pre = mhc_pre_delayed_torch(
                residual,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                self.rms_norm_eps,
                self.hc_eps,
                self.hc_eps,
                self.hc_post_alpha,
                self.hc_sinkhorn_iters,
                pre_mix=pre_mix,
            )
        x = self.attn_norm(x)

        if self.use_sequence_parallel:
            x = sp_all_gather(x)[: positions.shape[0]]

        x = self.attn(positions, x, None)
        if self.use_sequence_parallel:
            x = sp_reduce_scatter(x)

        residual = mhc_post_torch(x, residual, post_mix, res_mix)
        post_mix, res_mix, x, ffn_pre = mhc_pre_delayed_torch(
            residual,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            self.rms_norm_eps,
            self.hc_eps,
            self.hc_eps,
            self.hc_post_alpha,
            self.hc_sinkhorn_iters,
            pre_mix=attn_pre,
        )
        x = self.ffn_norm(x)
        x = self.ffn(x, input_ids)
        return x, residual, post_mix, res_mix, ffn_pre


class DeepseekV4Model(nn.Module, EagleModelMixin):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        self.parallel_config = vllm_config.parallel_config
        self.use_mega_moe = False
        self.use_sequence_parallel = _use_sequence_parallel(vllm_config)
        if self.use_mega_moe and not vllm_config.parallel_config.enable_expert_parallel:
            raise NotImplementedError(
                "DeepSeek V4 MegaMoE currently requires expert parallel. "
                "Enable it with --enable-expert-parallel, or pick a different "
                "moe backend."
            )
        self.vocab_size = config.vocab_size
        self.hc_eps = config.hc_eps
        self.hc_mult = config.hc_mult
        self.hc_dim = self.hc_mult * config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps

        # Three aux streams: one per non-default input GEMM in
        # DeepseekV4Attention._run_parallel_input_projections
        # (compressor kv_score, indexer.weights_proj). fused_wqa_wkv stays on
        # the default stream.
        aux_stream_list = [torch.cuda.Stream() for _ in range(3)]

        # Reserved topk indices buffer for all Indexer layers to reuse.
        self.topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            config.index_topk,
            dtype=torch.int32,
        )

        # Two-level candidate filtering: the indexer at
        # candidate_source_layer_id publishes the top candidate blocks of
        # compressed positions here; later ratio-1 indexers (24/28/32/36)
        # mask their scores with it.
        candidate_source_layer = getattr(config, "candidate_source_layer_id", -1)
        candidate_topk_blocks = getattr(config, "candidate_topk_blocks", 0)
        if candidate_source_layer >= 0 and candidate_topk_blocks > 0:
            self.candidate_block_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                candidate_topk_blocks,
                dtype=torch.int32,
            )
        else:
            self.candidate_block_buffer = None

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.engram_layout = EngramLayout.from_config(config)

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: DeepseekV4DecoderLayer(
                vllm_config,
                prefix=prefix,
                topk_indices_buffer=self.topk_indices_buffer,
                aux_stream_list=aux_stream_list,
                candidate_block_buffer=self.candidate_block_buffer,
                engram_layout=self.engram_layout,
            ),
            prefix=f"{prefix}.layers",
        )

        # The n-gram hash needs a slot-keyed rolling store of compressed ids
        # (chunked prefill / decode lookback); key it off the first local
        # layer's sliding-window KV cache. Only PP ranks owning an engram
        # layer need it.
        self.engram_hash: NgramHashState | None = None
        self.engram_swa_prefix: str | None = None
        if self.engram_layout is not None:
            local_engram = any(
                isinstance(layer, DeepseekV4DecoderLayer) and layer.engram is not None
                for layer in islice(self.layers, self.start_layer, self.end_layer)
            )
            if local_engram:
                first_layer = next(
                    iter(islice(self.layers, self.start_layer, self.end_layer))
                )
                swa_cache_module = first_layer.attn.swa_cache_layer
                self.engram_hash = NgramHashState(
                    vllm_config, self.engram_layout, swa_cache_module
                )
                self.engram_swa_prefix = swa_cache_module.prefix

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, self.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        spec_config = vllm_config.speculative_config
        needs_mtp_hidden_states = spec_config is not None and (
            spec_config.use_eagle() or spec_config.uses_draft_model()
        )
        if get_pp_group().is_last_rank and needs_mtp_hidden_states:
            self._mtp_hidden_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                self.hc_dim,
                dtype=vllm_config.model_config.dtype,
            )
        else:
            self._mtp_hidden_buffer = None

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def make_empty_intermediate_tensors(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        # PP intermediate tensors carry the multi-stream hidden_states
        # of shape (num_tokens, hc_mult, hidden_size) — V4 expands the
        # token embedding to hc_mult streams before the first decoder
        # layer and keeps that shape until the final hc collapse — plus the
        # (num_tokens, hc_mult) pre-mix the next rank's first layer needs
        # for its attention collapse.
        return IntermediateTensors(
            {
                "hidden_states": torch.zeros(
                    (batch_size, self.hc_mult, self.config.hidden_size),
                    dtype=dtype,
                    device=device,
                ),
                "pre_mix": torch.zeros(
                    (batch_size, self.hc_mult),
                    dtype=torch.float32,
                    device=device,
                ),
            }
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
        lookback_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        if self.use_mega_moe:
            input_ids = input_ids.to(torch.int64)

        # Engram n-gram hashes for the whole (flattened) batch, computed once
        # on the full token stream — before any sequence-parallel sharding —
        # and consumed by the engram layers (1 and 14) below. Skipped on
        # profile runs (KV cache unbound -> no slot space to key the rolling
        # compressed-id cache into).
        engram_hashes: torch.Tensor | None = None
        engram_mask: torch.Tensor | None = None
        if (
            self.engram_hash is not None
            and input_ids is not None
            and is_forward_context_available()
        ):
            attn_metadata = get_forward_context().attn_metadata
            if isinstance(attn_metadata, list):
                attn_metadata = attn_metadata[dbo_current_ubatch_id()]
            if isinstance(attn_metadata, dict) and self.engram_hash.ensure_cache():
                assert self.engram_swa_prefix is not None
                swa_metadata = typing.cast(
                    "DeepseekSparseSWAMetadata", attn_metadata[self.engram_swa_prefix]
                )
                # Image-span tokens are dead: they break n-grams (hash op
                # takes True=dead) and their gate is zeroed (Engram.forward
                # takes True=keep).
                image_mask = image_sentinel_mask(input_ids)
                engram_mask = ~image_mask
                if lookback_token_ids is None:
                    if not self.engram_hash.use_slot_cache:
                        raise NotImplementedError(
                            "engram needs `lookback_token_ids` from the model "
                            "runner (the DBO/ubatch wrapper drops model kwargs)"
                        )
                    num_reqs = swa_metadata.num_decodes + swa_metadata.num_prefills
                    lookback_token_ids = input_ids.new_full(
                        (num_reqs, self.engram_hash.lookback_depth), -1
                    )
                engram_hashes = self.engram_hash(
                    input_ids,
                    positions,
                    swa_metadata.query_start_loc,
                    image_mask,
                    lookback_token_ids,
                    image_sentinel_mask(lookback_token_ids),
                    swa_metadata.slot_mapping,
                    swa_metadata.block_table,
                )
                # Gather all Engram rows before entering the decoder layers.
                for layer in islice(self.layers, self.start_layer, self.end_layer):
                    engram = getattr(layer, "engram", None)
                    if engram is not None:
                        engram.prepare_embeddings(
                            engram_hashes[:, engram.layer_hash_index]
                        )

        full_num_tokens = positions.shape[0]
        if self.use_sequence_parallel:
            if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
                forward_context = get_forward_context()
                forward_context.is_padding = sp_padding_mask(
                    forward_context.is_padding, hidden_states
                )
            hidden_states = sp_shard(hidden_states)
            input_ids = sp_shard(input_ids)

        residual, post_mix, res_mix = None, None, None
        pre_mix: torch.Tensor | None = None
        if not get_pp_group().is_first_rank:
            assert intermediate_tensors is not None
            pre_mix = intermediate_tensors["pre_mix"]
        aux_hidden_states: list[torch.Tensor] = []
        final_aux_recon: torch.Tensor | None = None  # avoid duplicate mhc_post call
        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            hidden_states, residual, post_mix, res_mix, pre_mix = layer(
                hidden_states,
                positions,
                input_ids,
                pre_mix,
                post_mix,
                res_mix,
                residual,
                engram_hashes,
                engram_mask,
            )
            if idx + 1 in self.aux_hidden_state_layers:
                # Reconstruct the aux hidden state for draft models
                aux_recon = mhc_post_torch(hidden_states, residual, post_mix, res_mix)
                aux_hidden_state = aux_recon.mean(dim=1)
                if self.use_sequence_parallel:
                    aux_hidden_state = sp_all_gather(aux_hidden_state)[:full_num_tokens]
                aux_hidden_states.append(aux_hidden_state)
                final_aux_recon = aux_recon
        if layer is not None:
            # Reuse if the last layer was captured as an aux hidden state
            if self.end_layer in self.aux_hidden_state_layers:
                hidden_states = final_aux_recon
            else:
                hidden_states = mhc_post_torch(
                    hidden_states, residual, post_mix, res_mix
                )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "pre_mix": pre_mix}
            )

        if self.use_sequence_parallel:
            hidden_states = sp_all_gather(hidden_states)[:full_num_tokens]
            pre_mix = sp_all_gather(pre_mix)[:full_num_tokens]

        if self._mtp_hidden_buffer is not None:
            num_tokens = hidden_states.shape[0]
            self._mtp_hidden_buffer[:num_tokens].copy_(hidden_states.flatten(1))

        # Collapse the hc copies with the pre-mix from the last layer's FFN
        # mixes — the mix the reference applies via
        # ``last_layer.hc_pre(h, pre_mix)`` (v4.1 has no learned hc_head).
        assert pre_mix is not None
        hidden_states = (
            (pre_mix.unsqueeze(-1) * hidden_states.float())
            .sum(dim=1)
            .to(hidden_states.dtype)
        )
        hidden_states = self.norm(hidden_states)
        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "w1", 0),
            ("gate_up_proj", "w3", 1),
            ("attn.fused_wqa_wkv", "attn.wq_a", 0),
            ("attn.fused_wqa_wkv", "attn.wkv", 1),
            ("compressor.fused_wkv_wgate", "compressor.wkv", 0),
            ("compressor.fused_wkv_wgate", "compressor.wgate", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        def _resolve_param_name(name: str) -> str:
            """Resolve scale naming across ROCm FP8 and MXFP8 methods."""
            if name.endswith("_inv"):
                direct_name = name.removesuffix("_inv")
                if direct_name in params_dict:
                    return direct_name
            inv_name = f"{name}_inv"
            if name not in params_dict and inv_name in params_dict:
                return inv_name
            return name

        # TP for attention
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        n_head = self.config.num_attention_heads
        n_local_head = n_head // tp_size
        head_rank_start = n_local_head * tp_rank
        head_rank_end = n_local_head * (tp_rank + 1)

        # Pre-compute expert mapping ONCE.
        expert_mapping = self.get_expert_mapping()

        # Block-FP8 shared experts: pad the intermediate up to the TP-uniform
        # block count so the standard loaders below slice it evenly (trailing
        # ranks land on the zero pad). SP / unquantized ones need no padding.
        pad_shared_expert = (
            getattr(self.quant_config, "weight_block_size", None) is not None
            and not self.use_sequence_parallel
        )

        for name, loaded_weight in weights:
            if name.startswith(("vision.", "aligner.", "image_")):
                # Vision weights are loaded by the outer multimodal wrapper.
                logger.warning_once("Skipping non-text weight: %s", name)
                continue
            if pad_shared_expert and ".shared_experts." in name:
                loaded_weight = self._pad_shared_expert_weight(
                    self.quant_config, name, loaded_weight
                )
            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if ".experts." in name:
                    continue
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                if is_pp_missing_parameter(name, self):
                    break
                if name not in params_dict:
                    head, _, leaf = name.rpartition(".")
                    suffixed = f"{head}.base_layer.{leaf}"
                    if suffixed in params_dict:
                        name = suffixed
                name = _resolve_param_name(name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                break
            else:
                if ".experts." in name:
                    # E8M0 scales are stored as float8_e8m0fnu in
                    # checkpoints but the MoE param is uint8. copy_()
                    # would do a numeric conversion (e.g. 2^-7 → 0),
                    # destroying the raw exponent bytes.
                    if (
                        "weight_scale" in name
                        and loaded_weight.dtype == torch.float8_e8m0fnu
                    ):
                        loaded_weight = loaded_weight.view(torch.uint8)
                    for mapping in expert_mapping:
                        param_name, weight_name, expert_id, expert_shard_id = mapping
                        if weight_name not in name:
                            continue
                        name_mapped = name.replace(weight_name, param_name)
                        if is_pp_missing_parameter(name_mapped, self):
                            continue
                        name_mapped = _resolve_param_name(name_mapped)
                        param = params_dict[name_mapped]
                        # We should ask the weight loader to return success or not
                        # here since otherwise we may skip experts with other
                        # available replicas.
                        weight_loader = typing.cast(
                            Callable[..., bool], param.weight_loader
                        )
                        success = weight_loader(
                            param,
                            loaded_weight,
                            name_mapped,
                            shard_id=expert_shard_id,
                            expert_id=expert_id,
                            return_success=True,
                        )
                        if success:
                            name = name_mapped
                            break
                    loaded_params.add(name_mapped)
                    continue
                elif "attn_sink" in name:
                    if is_pp_missing_parameter(name, self):
                        continue
                    narrow_weight = loaded_weight[head_rank_start:head_rank_end]
                    n = narrow_weight.shape[0]
                    params_dict[name][:n].copy_(narrow_weight)
                    loaded_params.add(name)
                    continue
                else:
                    if is_pp_missing_parameter(name, self):
                        continue
                    # Non-LoRA params on a LoRA-wrapped module live at
                    # ``<head>.base_layer.<leaf>``; the checkpoint is plain.
                    if name not in params_dict:
                        head, _, leaf = name.rpartition(".")
                        suffixed = f"{head}.base_layer.{leaf}"
                        if suffixed in params_dict:
                            name = suffixed
                    name = _resolve_param_name(name)
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)
                    continue

        return loaded_params

    @staticmethod
    def _pad_shared_expert_weight(
        quant_config: QuantizationConfig | None,
        name: str,
        loaded_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Zero-pad a block-FP8 shared-expert weight/scale on its intermediate
        axis so the standard TP loaders split it into even, block-aligned shards
        (trailing ranks get the zero pad). gate (w1)/up (w3) [I, H] pad dim 0;
        down (w2 -> down_proj) [H, I] pads dim 1.
        """
        block_size = getattr(quant_config, "weight_block_size", None)
        assert block_size is not None
        # Round the intermediate axis up to a whole number of TP shards. The axis
        # is in elements for weights (step = block) and in blocks for scales.
        step = (
            1 if name.endswith(("weight_scale_inv", "weight_scale")) else block_size[0]
        )
        dim = 1 if ".down_proj." in name else 0
        mult = get_tensor_model_parallel_world_size() * step
        pad = cdiv(loaded_weight.shape[dim], mult) * mult - loaded_weight.shape[dim]
        if pad == 0:
            return loaded_weight
        pad_shape = list(loaded_weight.shape)
        pad_shape[dim] = pad
        return torch.cat([loaded_weight, loaded_weight.new_zeros(pad_shape)], dim=dim)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        return fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.n_routed_experts,
        )

    def finalize_mega_moe_weights(self) -> None:
        return

    def finalize_mhc_broadcast_weights(self) -> None:
        if not get_pp_group().is_first_rank or self.start_layer >= self.end_layer:
            return
        layer = self.layers[self.start_layer]
        if isinstance(layer, DeepseekV4DecoderLayer):
            broadcast = (
                layer.hc_attn_fn.detach()
                .view(-1, layer.hc_mult, layer.hidden_size)
                .sum(dim=1)
            )
            if layer.hc_attn_fn_broadcast is None:
                layer.hc_attn_fn_broadcast = broadcast
            else:
                layer.hc_attn_fn_broadcast.copy_(broadcast)


def _linear_scale_param_name(vllm_config: VllmConfig, expert_dtype: str) -> str:
    """Parameter name the linear quant method registers for weight scales.

    Native MXFP8 checkpoints ([32, 32] blocks with MXFP4 experts) route linear
    layers through ModelOptLinearMethod, which registers ``weight_scale``;
    block-FP8 linear layers register ``weight_scale_inv``.
    """
    use_mxfp8 = (
        getattr(vllm_config.quant_config, "weight_block_size", None) == [32, 32]
        and expert_dtype == "fp4"
    )
    return "weight_scale" if use_mxfp8 else "weight_scale_inv"


def _make_deepseek_v4_weights_mapper(
    expert_dtype: str, linear_scale_name: str = "weight_scale_inv"
) -> WeightsMapper:
    if expert_dtype == "fp4":
        # MXFP4 experts use Mxfp4MoEMethod, which registers scales as
        # ``w{1,2,3}_weight_scale`` (no _inv suffix). Linear scales use the
        # parameter name registered by their quantization method.
        scale_regex = {
            # ``.base_layer.``-namespace variant (LoRA-wrapped experts).
            re.compile(
                r"(\.experts\.\d+\.w[123]\.base_layer)\.scale$"
            ): r"\1.weight_scale",
            re.compile(r"(\.experts\.\d+\.w[123])\.scale$"): r"\1.weight_scale",
            # The ``embed.weight`` -> ``embed_tokens.weight`` suffix rule
            # renames the engram fp8 table but not its scale; route the
            # scale explicitly to the same module.
            re.compile(r"(engram\.embed)\.scale$"): r"\1_tokens.weight_scale_inv",
            re.compile(r"\.scale$"): f".{linear_scale_name}",
        }
    else:
        # FP8 experts use Fp8MoEMethod (block_quant=True), which registers
        # scales as ``w{13,2}_weight_scale_inv``. Map all ``.scale`` keys
        # there.
        scale_regex = {
            # ``.base_layer.``-namespace variant of the above.
            re.compile(
                r"(\.experts\.\d+\.w[123]\.base_layer)\.scale$"
            ): r"\1.weight_scale_inv",
            # Same engram reroute as the fp4 branch above.
            re.compile(r"(engram\.embed)\.scale$"): r"\1_tokens.weight_scale_inv",
            re.compile(r"\.scale$"): f".{linear_scale_name}",
        }
    return WeightsMapper(
        orig_to_new_prefix={
            "layers.": "model.layers.",
            "embed.": "model.embed.",
            "norm.": "model.norm.",
            "mtp.": "model.mtp.",
        },
        orig_to_new_regex=scale_regex,
        orig_to_new_suffix={
            "head.weight": "lm_head.weight",
            "embed.weight": "embed_tokens.weight",
            ".ffn.gate.bias": ".ffn.gate.e_score_correction_bias",
        },
        orig_to_new_substr={
            ".shared_experts.w2": ".shared_experts.down_proj",
            "mtp.": None,
            # The v4.1 checkpoint declares the VL arch even for text-only
            # use; the text backbone drops the vision-tower weights.
            "vision.": None,
            "aligner.": None,
            "image_": None,
        },
    )


class DeepseekV4MixtureOfExperts(MixtureOfExperts):
    moe_mlp_layers: list["DeepseekV4MoE"]

    def extract_moe_parameters(self, example_moe: "DeepseekV4MoE | None") -> None:
        if example_moe is None:
            self.num_moe_layers = 0
            self.num_expert_groups = 0
            self.num_logical_experts = 0
            self.num_physical_experts = 0
            self.num_local_physical_experts = 0
            self.num_routed_experts = 0
            self.num_shared_experts = 0
            self.num_redundant_experts = 0
            return
        self.num_logical_experts = example_moe.n_logical_experts
        self.num_physical_experts = example_moe.n_physical_experts
        self.num_local_physical_experts = example_moe.n_local_physical_experts
        self.num_routed_experts = example_moe.n_routed_experts
        self.num_shared_experts = example_moe.n_shared_experts
        self.num_redundant_experts = example_moe.n_redundant_experts

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for moe in self.moe_mlp_layers:
            moe.n_local_physical_experts = num_local_physical_experts
            moe.n_physical_experts = num_physical_experts
            moe.n_redundant_experts = self.num_redundant_experts
            moe.experts.update_expert_map()


class DeepseekV41LLMForCausalLM(
    nn.Module,
    SupportsPP,
    SupportsEagle3,
    SupportsLoRA,
    DeepseekV4MixtureOfExperts,
):
    model_cls = DeepseekV4Model

    # Default mapper assumes the original FP4-expert checkpoint layout.
    # Overridden per-instance in __init__ when expert_dtype != "fp4".
    hf_to_vllm_mapper = _make_deepseek_v4_weights_mapper("fp4")

    packed_modules_mapping = {
        "gate_up_proj": ["w1", "w3"],
        "fused_wqa_wkv": ["wq_a", "wkv"],
        "fused_wkv_wgate": ["wkv", "wgate"],
    }

    # The MTP draft head is not LoRA-adapted.
    lora_skip_prefixes = ["mtp."]

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        self.config = config
        expert_dtype = getattr(config, "expert_dtype", "fp4")
        self.hf_to_vllm_mapper = _make_deepseek_v4_weights_mapper(
            expert_dtype, _linear_scale_param_name(vllm_config, expert_dtype)
        )

        self.model = self.model_cls(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (  # type: ignore[method-assign]
            self.model.make_empty_intermediate_tensors
        )

        self.set_moe_parameters()

    def set_moe_parameters(self) -> None:
        self.num_expert_groups = getattr(self.config, "n_group", 1)
        self.num_moe_layers = self.config.num_hidden_layers
        self.moe_layers: list[nn.Module] = []
        self.moe_mlp_layers: list[DeepseekV4MoE] = []
        example_moe: DeepseekV4MoE | None = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            if not isinstance(layer, DeepseekV4DecoderLayer):
                continue
            if isinstance(layer.ffn, DeepseekV4MoE):
                example_moe = layer.ffn
                self.moe_mlp_layers.append(layer.ffn)
                self.moe_layers.append(layer.ffn.experts)

        self.num_moe_layers = len(self.moe_layers)
        self.extract_moe_parameters(example_moe)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def compute_logits_local(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states, skip_gather=True)

    @staticmethod
    def get_model_state_cls():
        from ..nvidia.model_state import DeepseekV41ModelState

        return DeepseekV41ModelState

    @property
    def token_lookback_depth(self) -> int:
        """Tokens before a chunk start the engram hash needs; the model runner
        passes them as `lookback_token_ids`."""
        engram_hash = self.model.engram_hash
        return engram_hash.lookback_depth if engram_hash is not None else 0

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        lookback_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            lookback_token_ids=lookback_token_ids,
        )
        return hidden_states

    def get_mtp_target_hidden_states(self) -> torch.Tensor | None:
        """Pre-collapse residual stream buffer (max_num_batched_tokens,
        hc_mult * hidden_size) for the MTP draft model. Populated by
        forward(); valid after each target step."""
        return getattr(self.model, "_mtp_hidden_buffer", None)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        loaded_params = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
        self.process_weights_after_loading()
        return loaded_params

    def process_weights_after_loading(self) -> None:
        self.model.finalize_mega_moe_weights()
        self.model.finalize_mhc_broadcast_weights()

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()
