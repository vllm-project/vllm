# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dense Muse Spark DSpark draft model for CUDA and ROCm."""

from collections.abc import Iterable
from copy import copy
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model
from vllm.model_executor.models.qwen3_dspark import (
    DSparkConfidenceHead,
    DSparkMarkovHead,
    Qwen3DSparkForCausalLM,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    get_draft_quant_config,
    maybe_prefix,
)
from vllm.models.muse_spark.common.layers import (
    MuseSpark1xRMSNorm,
    materialize_residual_path_constants,
)
from vllm.models.muse_spark.nvidia.specialized_decode.runtime import decode_backend
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.muse_spark import MuseSparkDSparkConfig

if TYPE_CHECKING or not current_platform.is_rocm():
    from vllm.models.muse_spark.nvidia.model import (
        MuseSpark1xDecoderLayer,
        gather_columns,
    )
else:
    from vllm.models.muse_spark.amd.model import (  # type: ignore[assignment]
        MuseSpark1xDecoderLayer,
    )

if TYPE_CHECKING:
    from vllm.models.muse_spark.nvidia.specialized_decode.runtime.draft import (
        DraftDecode,
    )
    from vllm.models.muse_spark.nvidia.specialized_decode.runtime.markov import (
        MarkovSampler,
    )
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

logger = init_logger(__name__)


def _use_column_parallel_fc(enabled: bool, hidden_size: int, tp_size: int) -> bool:
    """Whether to try the column-parallel draft fc, which the specialized
    decode (``enabled``) selects."""
    if not enabled:
        return False
    if tp_size == 1 or hidden_size % tp_size != 0:
        logger.warning_once(
            "The column-parallel DSpark draft fc needs a tensor parallel size "
            "above 1 that divides the draft hidden size %d (got %d); using the "
            "replicated DSpark draft fc.",
            hidden_size,
            tp_size,
        )
        return False
    return True


def _keep_column_parallel_fc(fc: ColumnParallelLinear) -> bool:
    """Whether the built column-parallel fc is unquantized, the only tested case."""
    if not isinstance(fc.quant_method, UnquantizedLinearMethod):
        logger.warning_once(
            "The column-parallel DSpark draft fc supports only an unquantized "
            "draft fc (got %s); using the replicated DSpark draft fc.",
            type(fc.quant_method).__name__,
        )
        return False
    logger.info_once("Using the column-parallel DSpark draft fc.", scope="process")
    return True


def _uses_specialized_decode(vllm_config: VllmConfig) -> bool:
    """Whether the deployment selects the Muse Spark specialized decode
    (``additional_config={"decode_backend": "latency_specialized"}``) for the
    draft. Not supported on ROCm, where the draft keeps the stock path."""
    if decode_backend(vllm_config) != "latency_specialized":
        return False
    if current_platform.is_rocm():
        logger.warning_once(
            "The Muse Spark specialized decode is not supported on ROCm; the "
            "DSpark draft keeps its stock layers and Markov loop."
        )
        return False
    return True


class MuseSparkDSparkDecoderLayer(MuseSpark1xDecoderLayer):
    """Dense Muse layer with the draft's materialized, gated FP32 residual."""

    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config, prefix)
        # The draft stream stays replicated even when the target runs SP.
        if getattr(self, "is_sequence_parallel", False):
            self.is_sequence_parallel = False
            self.is_sequence_parallel_input = False
            self.self_attn.is_sequence_parallel_output = False
            self.self_attn.o_proj.reduce_results = True
        config = vllm_config.model_config.hf_config
        self.self_attn.causal = True
        if (self.self_attn.rotary_emb is not None) != config.use_rope:
            raise ValueError(
                "Muse Spark DSpark layers must share their positional mode"
            )

    def forward(  # type: ignore[override]
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, None]:
        stream = hidden_states.to(self.residual_dtype)
        attn_out = self.self_attn(
            positions=positions,
            # Round to the draft compute dtype before each trained input norm.
            hidden_states=self.self_attn.pre_qkv_proj_norm(
                stream.to(self.default_dtype), out_dtype=self.default_dtype
            ),
        )
        stream = self.add_attn(stream=stream, branch=self.post_attn_norm(attn_out))
        ffn_out = self.feed_forward(self.pre_ffn_norm(stream.to(self.default_dtype)))
        stream = self.add_ffn(stream=stream, branch=self.post_ffn_norm(ffn_out))
        return stream, None


@support_torch_compile
class MuseSparkDSparkModel(DFlashQwen3Model):
    """Muse draft backbone with independently cached context K/V."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        if not isinstance(config, MuseSparkDSparkConfig):
            raise TypeError("MuseSparkDSparkModel requires MuseSparkDSparkConfig")
        self.config = config
        self.default_dtype = torch.get_default_dtype()
        self.vocab_size = config.vocab_size
        self.quant_config = get_draft_quant_config(vllm_config)
        self.use_aux_hidden_state = True
        self.target_layer_ids = tuple(config.target_layer_ids)
        # Avoid rerunning VllmConfig validation against the target's shared configs.
        draft_config = copy(vllm_config)
        draft_config.model_config = vllm_config.speculative_config.draft_model_config
        draft_config.quant_config = self.quant_config

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.mask_token_id = config.mask_token_id
        self.register_buffer(
            "mask_embedding",
            torch.zeros(config.hidden_size, dtype=vllm_config.model_config.dtype),
            persistent=False,
        )
        self.has_separate_mask_embedding = False
        fc_input_size = config.target_hidden_size * len(self.target_layer_ids)
        self.fc: ReplicatedLinear | ColumnParallelLinear
        specialized = _uses_specialized_decode(vllm_config)
        column_parallel_fc = _use_column_parallel_fc(
            specialized,
            config.hidden_size,
            get_tensor_model_parallel_world_size(),
        )
        if column_parallel_fc:
            self.fc = ColumnParallelLinear(
                input_size=fc_input_size,
                output_size=config.hidden_size,
                bias=False,
                gather_output=False,
                params_dtype=vllm_config.model_config.dtype,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "fc"),
                return_bias=False,
            )
            column_parallel_fc = _keep_column_parallel_fc(self.fc)
        if not column_parallel_fc:
            self.fc = ReplicatedLinear(
                input_size=fc_input_size,
                output_size=config.hidden_size,
                bias=False,
                params_dtype=vllm_config.model_config.dtype,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "fc"),
                return_bias=False,
            )
        self.hidden_norm = MuseSpark1xRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            has_weight=config.norm_affine,
            zero_centered_gamma=config.norm_zero_centered_gamma,
        )
        self.layers = nn.ModuleList(
            MuseSparkDSparkDecoderLayer(
                draft_config,
                prefix=maybe_prefix(prefix, f"layers.{layer_idx + start_layer_id}"),
            )
            for layer_idx in range(config.num_hidden_layers)
        )
        self.norm = MuseSpark1xRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            has_weight=config.norm_affine,
            zero_centered_gamma=config.norm_zero_centered_gamma,
        )
        markov_head_cls: type[DSparkMarkovHead] = DSparkMarkovHead
        if specialized:
            from vllm.models.muse_spark.nvidia.specialized_decode.runtime import (
                markov,
            )

            markov_head_cls = markov.VocabParallelMarkovHead
        self.markov_head = markov_head_cls(
            config.vocab_size,
            config.draft_vocab_size,
            config.markov_rank,
            prefix=maybe_prefix(prefix, "markov_head"),
            quant_config=self.quant_config,
        )
        self.confidence_head: DSparkConfidenceHead | None = None
        if config.enable_confidence_head:
            confidence_width = config.hidden_size + (
                config.markov_rank if config.confidence_head_with_markov else 0
            )
            self.confidence_head = DSparkConfidenceHead(
                confidence_width,
                prefix=maybe_prefix(prefix, "confidence_head"),
                bias=True,
                with_markov=config.confidence_head_with_markov,
            )
        # Under the specialized decode, the draft's forward once the weights
        # are loaded (MuseSparkDSparkForCausalLM.load_weights).
        self.specialized_decode = specialized
        self.specialized_draft: DraftDecode | None = None
        self._compilation_mode = vllm_config.compilation_config.mode

    def _build_fused_kv_buffers(self) -> None:
        layers_attn = [layer.self_attn for layer in self.layers]
        attn0 = layers_attn[0]
        self._context_pre_norms = tuple(attn.pre_qkv_proj_norm for attn in layers_attn)
        self._stacked_kv_weight = torch.stack(
            [
                attn.qkv_proj.weight[attn.q_size : attn.q_size + 2 * attn.kv_size]
                for attn in layers_attn
            ],
            dim=0,
        )
        self._num_attn_layers = len(layers_attn)
        self._kv_size = attn0.kv_size
        self._head_dim = attn0.head_dim
        self._num_kv_heads = attn0.num_kv_heads
        self._rms_norm_eps = attn0.qk_norm.variance_epsilon
        self._rotary_emb = attn0.rotary_emb
        self._attn_layers = [attn.attn for attn in layers_attn]
        for attn in layers_attn[1:]:
            if (
                attn.kv_size != self._kv_size
                or attn.head_dim != self._head_dim
                or attn.num_kv_heads != self._num_kv_heads
                or attn.qk_norm.variance_epsilon != self._rms_norm_eps
            ):
                raise ValueError("Muse Spark DSpark layers must share KV geometry")

    def _project_muse_context_kv(self, context_states: torch.Tensor) -> torch.Tensor:
        """Apply the two trained Muse input norms, then each layer's K/V map."""
        fc_output = self.hidden_norm(context_states)
        per_layer_context = torch.stack(
            [norm(fc_output) for norm in self._context_pre_norms], dim=0
        )
        return torch.bmm(per_layer_context, self._stacked_kv_weight.transpose(1, 2))

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | None = None,
    ) -> None:
        if self.specialized_draft is not None and (
            self.specialized_draft.store_context_kv(
                context_states, context_positions, context_slot_mapping
            )
        ):
            return
        if not hasattr(self, "_num_attn_layers"):
            self._build_fused_kv_buffers()
        num_ctx = context_states.shape[0]
        num_layers = self._num_attn_layers
        all_kv_flat = self._project_muse_context_kv(context_states)
        all_kv = (
            all_kv_flat.view(
                num_layers,
                num_ctx,
                2,
                self._num_kv_heads,
                self._head_dim,
            )
            .permute(2, 0, 1, 3, 4)
            .contiguous()
        )
        all_k = all_kv[0]
        all_v = all_kv[1]
        k_fp32 = all_k.float()
        variance = k_fp32.pow(2).mean(dim=-1, keepdim=True)
        all_k = (k_fp32 * torch.rsqrt(variance + self._rms_norm_eps)).to(all_k.dtype)
        all_k_flat = all_k.view(
            num_layers * num_ctx, self._num_kv_heads, self._head_dim
        )
        if self._rotary_emb is not None:
            positions = context_positions.repeat(num_layers)
            all_k_flat, _ = self._rotary_emb(
                positions,
                all_k_flat,
                # CUDA RoPE updates both operands in place. Aliasing the same
                # context-key tensor as Q and K rotates it twice.
                None,
            )
        if context_slot_mapping is None:
            return
        all_k = all_k_flat.view(num_layers, num_ctx, self._num_kv_heads, self._head_dim)
        per_layer = isinstance(context_slot_mapping, (list, tuple))
        for layer_idx, attn in enumerate(self._attn_layers):
            slot_mapping = (
                context_slot_mapping[layer_idx] if per_layer else context_slot_mapping
            )
            if slot_mapping is None:
                continue
            attn.impl.do_kv_cache_update(
                attn,
                all_k[layer_idx],
                all_v[layer_idx],
                attn.kv_cache,
                slot_mapping,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = (
            self.embed_input_ids(input_ids) if input_embeds is None else input_embeds
        )
        if self.specialized_draft is not None:
            return self.specialized_draft.run(positions, hidden_states)
        hidden_states = hidden_states.to(self.config.residual_dtype)
        for layer in self.layers:
            hidden_states, _ = layer(positions, hidden_states, None)
        return self.norm(hidden_states.to(self.default_dtype))


class MuseSparkDSparkForCausalLM(Qwen3DSparkForCausalLM):
    """Muse DSpark head with the target's output-distribution transform."""

    has_own_lm_head = True

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_stacked={
            ".self_attn.q_proj.": (".self_attn.qkv_proj.", "q"),
            ".self_attn.k_proj.": (".self_attn.qkv_proj.", "k"),
            ".self_attn.v_proj.": (".self_attn.qkv_proj.", "v"),
            ".self_attn.o_gate_proj.": (".self_attn.qkv_proj.", "o_gate"),
            ".feed_forward.gate_proj.": (".feed_forward.gate_up_proj.", 0),
            ".feed_forward.up_proj.": (".feed_forward.gate_up_proj.", 1),
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        assert vllm_config.speculative_config is not None
        self.draft_model_config = vllm_config.speculative_config.draft_model_config
        self.config = self.draft_model_config.hf_config
        target_layer_num = vllm_config.model_config.get_total_num_hidden_layers()
        self.model = MuseSparkDSparkModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            start_layer_id=target_layer_num,
        )
        self.lm_head = ParallelLMHead(
            self.config.draft_vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(self.config.draft_vocab_size)
        self.target_vocab_size = vllm_config.model_config.get_vocab_size()
        self.draft_id_to_target_id = nn.Parameter(
            torch.zeros(self.config.draft_vocab_size, dtype=torch.long),
            requires_grad=False,
        )
        metap = self.config.metap
        if metap.use_metap and metap.router_has_one_inf_dim:
            width_mult = self.config.hidden_size / metap.base_width
            if metap.metap_mode == "mup":
                self.output_multiplier = width_mult**-1
            elif metap.metap_mode == "sp":
                self.output_multiplier = width_mult**-0.5
            else:
                raise ValueError(f"Unknown Metap mode {metap.metap_mode!r}")
        else:
            self.output_multiplier = 1.0
        self.output_soft_cap_temp = self.config.output_soft_cap_temp
        topk = getattr(self.config, "dspark_draft_topk", None)
        if self.model.specialized_decode and topk is not None:
            # The speculator's top-k loop never calls sample_sequential.
            raise ValueError(
                "The Muse Spark specialized decode runs the fused DSpark Markov "
                "loop, which does not run here: dspark_draft_topk is set"
            )
        # The fused Markov loop under the specialized decode (sample_sequential).
        self._markov_sampler: MarkovSampler | None = None

    def apply_output_logit_transform(self, logits: torch.Tensor) -> torch.Tensor:
        logits = logits.float() * self.output_multiplier
        if self.output_soft_cap_temp is not None:
            logits = self.output_soft_cap_temp * torch.tanh(
                logits / self.output_soft_cap_temp
            )
        return logits

    def output_logit_transform_params(self) -> tuple[float, float | None]:
        """``(multiplier, soft_cap)`` of ``apply_output_logit_transform``.

        The fused DSpark Markov loop computes the transform from these values
        (``logits * multiplier``, then ``soft_cap * tanh(x / soft_cap)`` unless
        ``soft_cap`` is None) instead of calling it, so a subclass that changes
        the transform must override this too.
        """
        return self.output_multiplier, self.output_soft_cap_temp

    def sample_sequential(
        self,
        speculator: "DSparkSpeculator",
        num_reqs: int,
        head_hidden: torch.Tensor,
    ) -> bool:
        """DSparkSpeculator._sample_sequential asks this first. Under the
        specialized decode the fused Markov sampling loop samples the drafts
        and this returns True; otherwise it returns False, having done nothing,
        and the speculator runs its own loop."""
        if not self.model.specialized_decode:
            return False
        if self._markov_sampler is None:
            # Built on the first call (the memory-profiling run), from the
            # speculator's sampling state.
            self._markov_sampler = self._build_markov_sampler(speculator)
        assert speculator.draft_logits is not None
        self._markov_sampler.sample(
            num_reqs,
            head_hidden,
            sample_indices=speculator.sample_indices,
            sample_idx_mapping=speculator.sample_idx_mapping,
            sample_pos=speculator.sample_pos,
            input_ids=speculator.input_buffers.input_ids,
            anchor_idx=speculator._anchor_idx,
            seeds=speculator.seeds,
            temperature=speculator.temperature,
            draft_tokens=speculator.draft_tokens,
            draft_logits=speculator.draft_logits,
        )
        return True

    def _build_markov_sampler(self, speculator: "DSparkSpeculator") -> "MarkovSampler":
        if speculator.use_confidence_head or speculator.use_acceptance_estimator:
            raise ValueError(
                "The Muse Spark specialized decode's fused DSpark Markov loop "
                "computes neither the confidence head's outputs nor the "
                "acceptance estimates: it does not run with adaptive verification."
            )
        from vllm.models.muse_spark.nvidia.specialized_decode.runtime.markov import (
            build_markov_sampler,
        )

        return build_markov_sampler(
            self,
            draft_logits=speculator.draft_logits,
            draft_topk=speculator._draft_topk,
            watermarked=speculator.draft_watermarker is not None,
            use_fp64_gumbel=speculator.use_fp64_gumbel,
            sample_from_anchor=speculator.sample_from_anchor,
            max_num_reqs=speculator.max_num_reqs,
            num_steps=speculator.num_speculative_steps,
        )

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Identical full-vocabulary heads may be shared with an FP32 target head.
        return self.logits_processor(
            self.lm_head, hidden_states.to(self.lm_head.weight.dtype)
        )

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Target residuals are FP32; the training collator casts features before FC.
        hidden_states = hidden_states.to(self.model.fc.weight.dtype)
        out = super().combine_hidden_states(hidden_states)
        if isinstance(self.model.fc, ColumnParallelLinear):
            # Each rank holds its block of output columns for every token.
            out = gather_columns(out.view(-1, out.shape[-1])).view(*out.shape[:-1], -1)
        return out

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        model_weights = {}
        includes_lm_head = False
        includes_d2t = False
        for name, loaded_weight in weights:
            if name == "t2d":
                continue
            if name == "d2t":
                name = "draft_id_to_target_id"
                includes_d2t = True
            elif "lm_head" not in name:
                name = "model." + name
            if "lm_head" in name:
                includes_lm_head = True
            model_weights[name] = loaded_weight
        if not includes_lm_head:
            raise ValueError("Muse Spark DSpark requires its own lm_head")
        if not includes_d2t:
            raise ValueError("Muse Spark DSpark requires a d2t vocabulary mapping")
        mapper = WeightsMapper(
            orig_to_new_substr={
                "model.embed_tokens": None,
                "model.mask_embedding": None,
            },
            orig_to_new_stacked=self.hf_to_vllm_mapper.orig_to_new_stacked,
        )
        loaded = AutoWeightsLoader(self).load_weights(
            model_weights.items(), mapper=mapper
        )
        self.model._build_fused_kv_buffers()
        materialize_residual_path_constants(self.model)
        if self.model.specialized_decode:
            from vllm.models.muse_spark.nvidia.specialized_decode.runtime.draft import (
                build_draft_decode,
            )

            self.model.specialized_draft = build_draft_decode(
                self.model, self.model._compilation_mode
            )
        return loaded


EntryClass = MuseSparkDSparkForCausalLM
