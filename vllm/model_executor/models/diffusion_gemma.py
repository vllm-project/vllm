# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DiffusionGemma model, ModelState, and Sampler for vLLM.

Single Gemma4 backbone run in two modes (like YOCO):
- encoder mode: causal attention, writes KV cache
- decoder mode: bidirectional attention, reads encoder KV, doesn't write

Same weights, same layers. The only decoder-unique component is a
self-conditioning MLP.

Multimodal support: the model always includes a vision tower (shared with Gemma4).
Images are encoded through the vision tower and projected into the LM embedding space
via Gemma4MultimodalEmbedder.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
)
from vllm.model_executor.models.gemma4 import Gemma4Model
from vllm.model_executor.models.gemma4_mm import (
    Gemma4DummyInputsBuilder,
    Gemma4ForConditionalGeneration,
    Gemma4MultimodalEmbedder,
    Gemma4MultiModalProcessor,
    Gemma4ProcessingInfo,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.transformers.utils import recursive_replace_linear
from vllm.model_executor.models.utils import WeightsMapper, maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor, async_copy_to_gpu
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.sample.logprob import compute_topk_logprobs
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.penalties import use_penalty
from vllm.v1.worker.gpu.states import RequestState

from .diffusion_gemma_rowchunk import (
    diffusion_gemma_softcap_row_chunked_sample_soft_embeds,
)
from .interfaces import (
    SupportsMultiModal,
    SupportsPP,
    SupportsQuant,
)

logger = init_logger(__name__)

_DIFFUSION_GEMMA_SAMPLER_RESERVE_ENV = (
    "VLLM_DIFFUSION_GEMMA_SAMPLER_MEMORY_RESERVE_MIB"
)
_DIFFUSION_GEMMA_SAMPLER_RESERVE_SCALE_ENV = (
    "VLLM_DIFFUSION_GEMMA_SAMPLER_MEMORY_RESERVE_SCALE"
)
_DIFFUSION_GEMMA_SAMPLER_BACKEND_ENV = "VLLM_DIFFUSION_GEMMA_SAMPLER_BACKEND"
_DIFFUSION_GEMMA_ROW_CHUNK_ENV = "VLLM_DIFFUSION_GEMMA_ROW_CHUNK"
_DIFFUSION_GEMMA_ROW_CHUNK_SCRATCH_MIB_ENV = (
    "VLLM_DIFFUSION_GEMMA_ROW_CHUNK_SCRATCH_MIB"
)
_DIFFUSION_GEMMA_SAMPLER_BACKENDS = {"materialized", "row_chunked"}
_DIFFUSION_GEMMA_ROW_CHUNK_BYTES_PER_ELEMENT = 64
_DIFFUSION_GEMMA_ROW_CHUNK_ALIGNMENT = 64


def _get_diffusion_gemma_sampler_memory_reserve_bytes(
    reserve_spec: str,
    reserve_scale: float,
    *,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    canvas_length: int,
    vocab_size: int,
) -> int:
    """Estimate extra KV-sizing reserve for DiffusionGemma sampler scratch."""
    reserve_spec = reserve_spec.strip()
    if not reserve_spec or reserve_spec == "0":
        return 0

    if reserve_spec.lower() != "auto":
        try:
            reserve_mib = int(reserve_spec)
        except ValueError as err:
            raise ValueError(
                f"{_DIFFUSION_GEMMA_SAMPLER_RESERVE_ENV} must be >= 0 or 'auto'"
            ) from err
        if reserve_mib < 0:
            raise ValueError(
                f"{_DIFFUSION_GEMMA_SAMPLER_RESERVE_ENV} must be >= 0 or 'auto'"
            )
        return reserve_mib * (1 << 20)

    if canvas_length <= 0:
        raise ValueError("DiffusionGemma canvas_length must be positive")
    if max_num_seqs <= 0:
        raise ValueError("DiffusionGemma max_num_seqs must be positive")
    if max_num_batched_tokens <= 0:
        raise ValueError("DiffusionGemma max_num_batched_tokens must be positive")
    if vocab_size <= 0:
        raise ValueError("DiffusionGemma vocab_size must be positive")
    if reserve_scale < 0:
        raise ValueError(
            f"{_DIFFUSION_GEMMA_SAMPLER_RESERVE_SCALE_ENV} must be >= 0"
        )

    max_decode_reqs = min(
        int(max_num_seqs),
        max(1, int(max_num_batched_tokens) // int(canvas_length)),
    )
    sampler_rows = max_decode_reqs * int(canvas_length)
    # DiffusionGemma's materialized decode path can have more than one
    # full-vocab fp32 transient live across logits softcap/gather/sampler.
    # Reserve two [sampler_rows, global_vocab] fp32 buffers for the auto
    # setting; callers can still tune this with the scale env.
    return int(sampler_rows * int(vocab_size) * 4 * 2 * reserve_scale)


def _get_diffusion_gemma_sampler_backend() -> str:
    backend = envs.VLLM_DIFFUSION_GEMMA_SAMPLER_BACKEND.strip().lower()
    if backend not in _DIFFUSION_GEMMA_SAMPLER_BACKENDS:
        supported = ", ".join(sorted(_DIFFUSION_GEMMA_SAMPLER_BACKENDS))
        raise ValueError(
            f"{_DIFFUSION_GEMMA_SAMPLER_BACKEND_ENV}={backend!r} is not "
            f"supported; expected one of: {supported}"
        )
    return backend


def _should_use_diffusion_gemma_row_chunked_sampler(
    *,
    row_chunked_input: bool,
    real_logprobs_enabled: bool,
) -> bool:
    # Logprobs require materialized vocab logits for top-k logprob output.
    # Do not raise here: this helper is called from the EngineCore worker path,
    # where uncaught exceptions kill the engine. The caller handles the safe
    # materialized fallback when hidden states were returned by compute_logits.
    return row_chunked_input and not real_logprobs_enabled


def _is_diffusion_gemma_sampler_warmup_batch(input_batch: Any) -> bool:
    req_ids = getattr(input_batch, "req_ids", ())
    num_reqs = getattr(input_batch, "num_reqs", 0)
    return num_reqs > 0 and all(
        str(req_id).startswith("_warmup_") for req_id in req_ids[:num_reqs]
    )


def _is_diffusion_gemma_hidden_state_input(
    logits: torch.Tensor, vocab_size: int, hidden_size: int
) -> bool:
    last_dim = logits.shape[-1]
    is_materialized_logits = last_dim == vocab_size
    is_hidden_states = last_dim == hidden_size
    if not is_materialized_logits and not is_hidden_states:
        raise ValueError(
            "DiffusionGemma row-chunked sampler expected hidden states "
            f"with hidden size {hidden_size} or materialized logits with "
            f"vocab size {vocab_size}, got shape {tuple(logits.shape)}."
        )
    return is_hidden_states


def _get_diffusion_gemma_row_chunk_size(rows: int, vocab_size: int) -> int:
    if rows < 0:
        raise ValueError("rows must be >= 0")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if rows == 0:
        return 1

    explicit = envs.VLLM_DIFFUSION_GEMMA_ROW_CHUNK
    if explicit < 0:
        raise ValueError(f"{_DIFFUSION_GEMMA_ROW_CHUNK_ENV} must be >= 0")
    if explicit > 0:
        return min(rows, explicit)

    scratch_mib = envs.VLLM_DIFFUSION_GEMMA_ROW_CHUNK_SCRATCH_MIB
    if scratch_mib <= 0:
        raise ValueError(
            f"{_DIFFUSION_GEMMA_ROW_CHUNK_SCRATCH_MIB_ENV} must be > 0"
        )
    scratch_bytes = scratch_mib * (1 << 20)
    per_row = vocab_size * _DIFFUSION_GEMMA_ROW_CHUNK_BYTES_PER_ELEMENT
    chunk = max(1, scratch_bytes // max(1, per_row))
    if chunk >= _DIFFUSION_GEMMA_ROW_CHUNK_ALIGNMENT:
        chunk = (
            chunk
            // _DIFFUSION_GEMMA_ROW_CHUNK_ALIGNMENT
            * _DIFFUSION_GEMMA_ROW_CHUNK_ALIGNMENT
        )
    return max(1, min(rows, int(chunk)))


class DiffusionGemmaSelfConditioning(nn.Module):
    """Gated MLP that processes soft embeddings from the previous denoising step.

    Structurally identical to Gemma4MLP but with self_conditioning_size
    and post_norm without learned scale.
    """

    def __init__(
        self, hidden_size: int, self_conditioning_size: int, eps: float = 1e-6
    ):
        super().__init__()
        self.pre_norm = RMSNorm(hidden_size, eps=eps)
        self.post_norm = RMSNorm(hidden_size, eps=eps, has_weight=False)
        self.gate_proj = nn.Linear(hidden_size, self_conditioning_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, self_conditioning_size, bias=False)
        self.down_proj = nn.Linear(self_conditioning_size, hidden_size, bias=False)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        soft_embeds: torch.Tensor,
    ) -> torch.Tensor:
        x = self.pre_norm(soft_embeds)
        sc_signal = self.down_proj(
            F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x)
        )
        return self.post_norm(inputs_embeds + sc_signal)


# ---------------------------------------------------------------------------
# Multimodal processing info (overrides Gemma4 config type check)
# ---------------------------------------------------------------------------


class DiffusionGemmaProcessingInfo(Gemma4ProcessingInfo):
    """Processing info for DiffusionGemma.

    Overrides ``get_hf_config`` to accept ``DiffusionGemmaConfig``
    (which inherits from ``PretrainedConfig``, not ``Gemma4Config``).
    Supports image and video modalities.
    """

    def get_hf_config(self):
        # DiffusionGemmaConfig doesn't inherit from Gemma4Config, so we
        # accept any PretrainedConfig here.
        return self.ctx.get_hf_config()

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        # DiffusionGemma supports image and video inputs.
        return {"image": None, "video": None}

    def get_mm_max_tokens_per_item(
        self, seq_len: int, mm_counts: Mapping[str, int]
    ) -> Mapping[str, int] | None:
        return super().get_mm_max_tokens_per_item(seq_len, mm_counts)


@torch.compile(dynamic=True)
def _softcap_logits(logits: torch.Tensor, cap: float) -> torch.Tensor:
    # fp32 before tanh for numerical stability (matches HF DiffusionGemma).
    # Compiling fuses the cast/div/tanh/mul into one elementwise kernel over
    # the [num_tokens, vocab] logits instead of four separate passes.
    logits = logits.float()
    return torch.tanh(logits / cap) * cap


def _materialize_diffusion_gemma_logits_from_hidden(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    vocab_size: int,
    final_logit_softcapping: float | None,
) -> torch.Tensor:
    """Project row-chunk backend hidden states to vocab logits for fallback paths."""
    logits = F.linear(hidden_states, lm_head_weight[:vocab_size])
    if final_logit_softcapping is not None:
        logits = _softcap_logits(logits, final_logit_softcapping)
    return logits


@MULTIMODAL_REGISTRY.register_processor(
    Gemma4MultiModalProcessor,
    info=DiffusionGemmaProcessingInfo,
    dummy_inputs=Gemma4DummyInputsBuilder,
)
class DiffusionGemmaForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsQuant,
    SupportsPP,
):
    """DiffusionGemma for vLLM.

    Single Gemma4 backbone that switches between encoder and decoder mode.
    The encoder path uses standard Gemma4 layers (causal attention, KV write).
    The decoder path uses the same weights with bidirectional attention and
    KV read-only, plus self-conditioning.

    Always includes a vision tower (same as Gemma4) for image understanding.

    In practice, the model's forward() dispatches based on the `mode` kwarg
    set by DiffusionGemmaModelState.prepare_inputs().
    """

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.decoder.": "model.",
            "model.encoder.language_model.": "model.",
            "model.encoder.vision_tower.": "vision_tower.",
            "model.encoder.embed_vision.": "embed_vision.",
        },
        orig_to_new_substr={
            ".experts.": ".moe.experts.",
        },
    )

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    @staticmethod
    def get_model_state_cls():
        return DiffusionGemmaModelState

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        text_config = vllm_config.model_config.hf_text_config
        self.config = config
        self.model_dtype = vllm_config.model_config.dtype

        # DiffusionGemma's full-attention layers have NO v_proj — V is
        # computed from k_proj's output (`value_states = key_states` before
        # k_norm in `DiffusionGemmaDecoderTextAttention.forward`). This is
        # the "k_eq_v" variant in our Gemma4 backbone. The checkpoint has no
        # v_proj weights for full-attention layers; without this flag they
        # would silently load with random V projections.
        text_config.attention_k_eq_v = True

        # ---- Vision tower ----
        vision_config = getattr(config, "vision_config", None)
        if vision_config is not None:
            quant_config = vllm_config.quant_config
            if quant_config and quant_config.get_name() in [
                "bitsandbytes",
                "torchao",
                "compressed-tensors",
            ]:
                tower_quant = quant_config
            else:
                quantizable = (
                    vision_config.hidden_size % 64 == 0
                    and vision_config.intermediate_size % 64 == 0
                )
                tower_quant = quant_config if quantizable else None

            with self._mark_tower_model(vllm_config, {"image", "video"}):
                self.vision_tower = AutoModel.from_config(config=vision_config)
                self.embed_vision = Gemma4MultimodalEmbedder(
                    vision_config,
                    text_config,
                    quant_config=tower_quant,
                    prefix=maybe_prefix(prefix, "embed_vision"),
                )
                recursive_replace_linear(
                    self.vision_tower,
                    tower_quant,
                    prefix=maybe_prefix(prefix, "vision_tower"),
                )
        else:
            self.vision_tower = None
            self.embed_vision = None

        # ---- Language backbone (Gemma4Model) ----
        # Use maybe_prefix to ensure correct weight name prefixes for
        # quantization. The quantization config uses hf_to_vllm_mapper to
        # match checkpoint weight names to model parameter names.
        self.model = Gemma4Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )

        self.lm_head = ParallelLMHead(
            num_embeddings=text_config.vocab_size,
            embedding_dim=text_config.hidden_size,
        )

        if text_config.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)

        # HF DiffusionGemma applies the final-logit softcap in fp32, before
        # any other processing. Do it manually in `compute_logits` so the
        # LogitsProcessor only handles the lm_head GEMM.
        self.final_logit_softcapping = getattr(
            text_config, "final_logit_softcapping", None
        )

        self.sampler_backend = _get_diffusion_gemma_sampler_backend()
        self.use_row_chunked_sampler = self.sampler_backend == "row_chunked"
        if self.use_row_chunked_sampler:
            tensor_parallel_size = vllm_config.parallel_config.tensor_parallel_size
            if tensor_parallel_size != 1:
                raise ValueError(
                    "DiffusionGemma row-chunked sampler currently requires "
                    f"tensor_parallel_size=1, got {tensor_parallel_size}. TP "
                    "support needs a global LSE/soft-embed merge before this "
                    "pressure-mode path can be enabled."
                )
            if self.lm_head.weight.shape[0] != text_config.vocab_size:
                raise ValueError(
                    "DiffusionGemma row-chunked sampler requires an unsharded "
                    "lm_head with one row per vocab entry, got "
                    f"{self.lm_head.weight.shape[0]} local rows for global "
                    f"vocab {text_config.vocab_size}."
                )
            if self.final_logit_softcapping is None:
                raise ValueError(
                    "DiffusionGemma row-chunked sampler requires "
                    "final_logit_softcapping so it can match the materialized "
                    "sampler numerics."
                )
            logger.warning_once(
                "DiffusionGemma row-chunked sampler backend is enabled "
                "(%s=%s). This opt-in pressure-mode path bounds sampler "
                "scratch for large decode batches; logprobs are not supported "
                "with this backend.",
                _DIFFUSION_GEMMA_SAMPLER_BACKEND_ENV,
                self.sampler_backend,
            )
        self.logits_processor = LogitsProcessor(
            text_config.vocab_size,
            soft_cap=None,
        )

        sc_size = (
            getattr(config, "self_conditioning_size", None)
            or text_config.intermediate_size
        )
        self.self_conditioning = DiffusionGemmaSelfConditioning(
            hidden_size=text_config.hidden_size,
            self_conditioning_size=sc_size,
            eps=getattr(text_config, "rms_norm_eps", 1e-6),
        )

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def compute_self_conditioning(
        self,
        inputs_embeds: torch.Tensor,
        probs: torch.Tensor,
    ) -> torch.Tensor:
        embed_weight = self.model.embed_tokens.weight
        soft_embeds = torch.matmul(
            probs.to(embed_weight.dtype), embed_weight
        ) * self.model.normalizer.to(inputs_embeds.dtype)
        return self.self_conditioning(inputs_embeds, soft_embeds)

    # ------------------------------------------------------------------ #
    # Multimodal: reuse Gemma4's image parsing, processing & embedding
    # ------------------------------------------------------------------ #
    # The vision tower, pooler, embed_vision, and their processing logic
    # are architecturally identical to Gemma4.  Delegate to avoid
    # maintaining a duplicate copy.

    _parse_and_validate_image_input = (
        Gemma4ForConditionalGeneration._parse_and_validate_image_input
    )
    _parse_and_validate_video_input = (
        Gemma4ForConditionalGeneration._parse_and_validate_video_input
    )
    _parse_and_validate_multimodal_inputs = (
        Gemma4ForConditionalGeneration._parse_and_validate_multimodal_inputs
    )
    _encoder_chunk = staticmethod(Gemma4ForConditionalGeneration._encoder_chunk)
    _process_image_input = Gemma4ForConditionalGeneration._process_image_input
    _process_video_input = Gemma4ForConditionalGeneration._process_video_input
    embed_multimodal = Gemma4ForConditionalGeneration.embed_multimodal

    def get_mm_mapping(self) -> MultiModelKeys:
        """Get the module prefix mapping for multimodal models."""
        return MultiModelKeys.from_string_field(
            language_model="model",
            connector=["embed_vision"],
            tower_model=["vision_tower"],
        )

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Any | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if intermediate_tensors is not None:
            inputs_embeds = None
        return self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        if self.use_row_chunked_sampler:
            return hidden_states
        logits = self.logits_processor(self.lm_head, hidden_states)
        if logits is not None and self.final_logit_softcapping is not None:
            logits = _softcap_logits(logits, self.final_logit_softcapping)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """Load weights from checkpoint.

        Checkpoint layout (HF DiffusionGemma):
          model.encoder.vision_tower.*            → vision tower
          model.encoder.embed_vision.*            → vision embedder
          model.encoder.language_model.layers.*   → backbone
          model.decoder.layers.*                  → backbone (tied)
          model.decoder.embed_tokens.*            → embeddings
          model.decoder.self_conditioning.*       → self-conditioning MLP
          lm_head.*                               → LM head (tied)

        We load encoder weights into our single ``Gemma4Model`` backbone,
        skip duplicate decoder backbone weights, handle vision tower and
        self-conditioning separately.
        """

        sc_params = dict(
            (n, p)
            for n, p in self.named_parameters()
            if n.startswith("self_conditioning.")
        )

        # Collect vision tower + embedder parameters AND buffers for manual
        # loading.  The HF vision tower registers std_bias / std_scale as
        # buffers (not parameters) when config.standardize is True, so we
        # must include named_buffers() to avoid "not found in model" warnings.
        vision_params: dict[str, torch.Tensor] = {}
        for n, p in self.named_parameters():
            if n.startswith(("vision_tower.", "embed_vision.")):
                vision_params[n] = p
        for n, b in self.named_buffers():
            if n.startswith(("vision_tower.", "embed_vision.")):
                vision_params[n] = b

        def _remap_weights():
            # Use full weight names (including suffixes like .weight_scale,
            # .weight_packed) for dedup instead of just the base layer name. Critical
            # for quantized checkpoints where each weight has multiple tensors;
            # tracking only base names skips scales as duplicates.
            seen_weights: set[str] = set()
            for name, weight in weights:
                # Self-conditioning lives under model.decoder.self_conditioning.*
                # in the checkpoint but at self_conditioning.* in our model.
                if "self_conditioning" in name:
                    sc_name = name.split("self_conditioning.", 1)[1]
                    sc_name = "self_conditioning." + sc_name
                    if sc_name in sc_params:
                        sc_params[sc_name].data.copy_(weight)
                    continue

                # Vision tower: model.encoder.vision_tower.* → vision_tower.*
                # In HF, the vision tower is a sibling of language_model
                # under the encoder module.
                if name.startswith("model.encoder.vision_tower."):
                    vt_name = name[len("model.encoder.") :]
                    if vt_name in vision_params:
                        vision_params[vt_name].data.copy_(weight)
                    else:
                        logger.warning(
                            "Vision tower weight %s (mapped to %s) not found in model",
                            name,
                            vt_name,
                        )
                    continue

                # Vision embedder: model.encoder.embed_vision.* → embed_vision.*
                if name.startswith("model.encoder.embed_vision."):
                    ev_name = name[len("model.encoder.") :]
                    if ev_name in vision_params:
                        vision_params[ev_name].data.copy_(weight)
                    else:
                        logger.warning(
                            "Embed vision weight %s (mapped to %s) not found in model",
                            name,
                            ev_name,
                        )
                    continue

                # Skip vestigial embed_vision.embedding weights.
                if "embed_vision.embedding." in name:
                    continue

                # Encoder backbone → model.*
                if name.startswith("model.encoder.language_model."):
                    name = name.replace("model.encoder.language_model.", "model.")
                # Decoder backbone → model.* (skip exact duplicates)
                elif name.startswith("model.decoder."):
                    name = name.replace("model.decoder.", "model.")

                # Skip only if we've seen the exact same weight name (including scales)
                if name in seen_weights:
                    continue
                seen_weights.add(name)
                yield name, weight

        # Delegate to Gemma4ForCausalLM.load_weights for the backbone,
        # which handles stacked params, MoE, k_eq_v, etc.
        # Temporarily set self.config to text_config since Gemma4's
        # load_weights expects it (e.g. tie_word_embeddings, layer_types).
        from vllm.model_executor.models.gemma4 import Gemma4ForCausalLM

        saved_config = self.config
        self.config = self.model.config
        try:
            Gemma4ForCausalLM.load_weights(self, _remap_weights())
        finally:
            self.config = saved_config

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality == "image":
            return "<image_soft_token>"
        if modality == "video":
            return "<|video|>"
        raise ValueError(f"Unsupported modality: {modality}")


@torch.compile(dynamic=True)
def _compute_num_rejected(
    num_logits: torch.Tensor,
    num_sampled: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> torch.Tensor:
    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    num_rejected = num_logits - num_sampled
    is_denoise = (num_logits > 0) & (num_sampled == 0)
    return torch.where(is_denoise, query_lens, num_rejected)


@torch.compile(dynamic=True)
def _compiled_sample_step(
    # Logits from the model [num_decode * CL, vocab]
    logits: torch.Tensor,
    # Request mapping
    decode_slots: torch.Tensor,  # [num_decode] int64 → slot indices
    decode_idx: torch.Tensor,  # [num_decode] int64 → position in num_reqs
    all_slots: torch.Tensor,  # [num_reqs] int64 → all slot indices
    valid_canvas_len: torch.Tensor,  # [num_decode] int64 → real canvas length (<=CL)
    # State tensors (modified in-place)
    canvas: torch.Tensor,  # [max_num_reqs, CL]
    argmax_canvas: torch.Tensor,  # [max_num_reqs, CL]
    step_tensor: torch.Tensor,  # [max_num_reqs]
    is_encoder_phase: torch.Tensor,  # [max_num_reqs]
    confident_tensor: torch.Tensor,  # [max_num_reqs]
    sc_embeds: torch.Tensor,  # [max_num_reqs, CL, hidden]
    embed_weight: torch.Tensor,  # [vocab, hidden]
    normalizer: torch.Tensor,
    history: torch.Tensor,  # [max_num_reqs, ST, CL]
    history_len_tensor: torch.Tensor,  # [max_num_reqs]
    # Output tensors (modified in-place)
    sampled: torch.Tensor,  # [num_reqs, CL]
    num_sampled: torch.Tensor,  # [num_reqs]
    draft_tokens: torch.Tensor,  # [max_num_reqs, >=CL]
    # Scalar config
    max_denoising_steps: float,
    t_min: float,
    t_max: float,
    confidence_threshold: float,
    vocab_size: int,
    CL: int,
    ST: int,
    # Sampler config
    entropy_bound: float,
    # Tensor-parallel vocab sharding for the self-conditioning matmul.
    # ``embed_weight`` is vocab-sharded ([vocab/tp, hidden]) while ``probs``
    # spans the full vocab; [sc_vocab_start, sc_vocab_end) is this rank's slice.
    sc_vocab_start: int,
    sc_vocab_end: int,
    tp_size: int,
    tp_group_name: str,
) -> torch.Tensor:
    """Compiled decode step: temperature → Gumbel sample → probs/confidence →
    accept/renoise → convergence, all as vectorized PyTorch ops.

    Returns the temperature-scaled logits ``[num_decode, CL, vocab]`` so the
    caller can compute logprobs outside the compiled region."""
    num_decode = decode_slots.shape[0]
    device = decode_slots.device

    # Clear outputs so prefill / non-decode slots report 0 (decode slots are
    # overwritten below).
    sampled.zero_()
    num_sampled.zero_()

    # ---- Phase 1: Temperature schedule ----
    steps_f = step_tensor[decode_slots].float()
    remaining = (max_denoising_steps - steps_f).clamp(min=1.0)
    temp = t_min + (t_max - t_min) * (remaining / max_denoising_steps)

    # ---- Phase 2: Temperature scaling + Gumbel-max sampling ----
    logits_3d = logits.reshape(num_decode, CL, -1).float()
    scaled = logits_3d / temp[:, None, None].clamp(min=1e-10)

    # Gumbel-max trick: argmax(logits/T + Gumbel) ~ sample from softmax(logits/T)
    u = torch.rand_like(scaled).clamp(min=1e-20)
    gumbel = -torch.log(-torch.log(u))
    # Zero noise when temp==0 (greedy)
    noisy = scaled + gumbel * (temp[:, None, None] > 0).float()
    new_tokens = noisy.view(-1, noisy.shape[-1]).argmax(dim=-1).view(num_decode, CL)
    argmax_tokens = (
        scaled.view(-1, scaled.shape[-1]).argmax(dim=-1).view(num_decode, CL)
    )

    # ---- Phase 3: Probs, self-conditioning, confidence ----
    log_probs = scaled.log_softmax(dim=-1)
    probs = log_probs.exp()

    token_entropy = -(probs * log_probs).sum(dim=-1)  # [num_decode, CL]
    # A canvas truncated near max_model_len is zero-padded up to CL by the
    # caller; those padded rows are uniform (max entropy, argmax 0), so they
    # never trigger early convergence and are stable, and only the real
    # ``valid_canvas_len`` tokens are committed (num_sampled below).
    mean_entropy = token_entropy.mean(dim=-1)  # [num_decode]
    confident_tensor[decode_slots] = mean_entropy < confidence_threshold

    # ---- Phase 4: Entropy-bound acceptance mask ----
    sorted_ent, sorted_idx = torch.sort(token_entropy, dim=-1)
    cumsum_ent = torch.cumsum(sorted_ent, dim=-1)
    cummax_ent = torch.cummax(sorted_ent, dim=-1).values
    sorted_mask = (cumsum_ent - cummax_ent) <= entropy_bound
    eb_mask = torch.zeros_like(sorted_mask)
    eb_mask.scatter_(1, sorted_idx, sorted_mask)

    # ---- Phase 5: Post-sample ----
    is_commit = is_encoder_phase[decode_slots]  # [num_decode]
    is_denoise = ~is_commit
    cur_step = step_tensor[decode_slots].float()

    # Step update: +1 for denoise, reset to 0 for commit
    new_step_val = torch.where(
        is_denoise,
        (cur_step + 1).to(step_tensor.dtype),
        step_tensor.new_zeros(num_decode),
    )
    step_tensor[decode_slots] = new_step_val

    # Random tokens for renoise / canvas reinit
    random_tokens = torch.randint(
        0, vocab_size, (num_decode, CL), device=device, dtype=canvas.dtype
    )

    # Compute denoise canvas (accept/renoise)
    denoise_canvas = torch.where(eb_mask, new_tokens, random_tokens)

    # Canvas: commit → random reinit, denoise → accept/renoise result
    canvas[decode_slots] = torch.where(
        is_commit.unsqueeze(1), random_tokens, denoise_canvas
    )

    # History: write argmax_tokens for denoise requests at circular position
    hist_len = history_len_tensor[decode_slots]
    write_pos = hist_len % ST
    for i in range(ST):
        write_here = ((write_pos == i) & is_denoise).unsqueeze(1)
        history[decode_slots, i] = torch.where(
            write_here, argmax_tokens, history[decode_slots, i]
        )

    # Argmax canvas: update for denoise, preserve for commit
    argmax_canvas[decode_slots] = torch.where(
        is_denoise.unsqueeze(1), argmax_tokens, argmax_canvas[decode_slots]
    )

    # History length: increment for denoise, reset for commit
    new_hist_len = torch.where(is_denoise, hist_len + 1, hist_len.new_zeros(num_decode))
    history_len_tensor[decode_slots] = new_hist_len

    # Sampled output: commit → emit argmax_canvas, denoise → 0 (pre-zeroed)
    sampled[decode_idx] = argmax_canvas[decode_slots].to(
        sampled.dtype
    ) * is_commit.unsqueeze(1).to(sampled.dtype)
    # Commit only the real canvas length (== CL except for a canvas truncated
    # near max_model_len); the padded tail positions are never emitted.
    num_sampled[decode_idx] = is_commit.to(num_sampled.dtype) * valid_canvas_len.to(
        num_sampled.dtype
    )

    # ---- Phase 6: Stability + convergence ----
    ref = history[decode_slots, 0]
    mismatch = torch.zeros(num_decode, device=device, dtype=torch.int32)
    for h in range(1, ST):
        mismatch = mismatch + (ref != history[decode_slots, h]).sum(dim=-1).int()
    stable = mismatch == 0

    step_after = step_tensor[decode_slots]
    converged = (stable & confident_tensor[decode_slots] & (new_hist_len >= ST)) | (
        step_after >= max_denoising_steps
    )
    # Commit done → denoise next (False); denoise converged → commit next (True)
    is_encoder_phase[decode_slots] = torch.where(
        is_commit, is_commit.new_zeros(num_decode), converged
    )

    # SC soft embedding: store ``probs @ embed_weight`` (the value the next step's
    # self-conditioning MLP consumes) only for slots that will denoise next — i.e.
    # this step denoised AND it isn't about to commit (is_encoder_phase now False).
    # Masking here (rather than in the consumer) lets _apply_self_conditioning read
    # sc_embeds directly. Storing the [.., hidden] soft embed instead of the full
    # [.., vocab] probs avoids a giant persistent buffer.
    sc_keep = (is_denoise & ~is_encoder_phase[decode_slots])[:, None, None]
    # Self-conditioning soft embed = probs @ embed_tokens.weight. Under tensor
    # parallelism the embedding is vocab-sharded ([vocab/tp, hidden]) while
    # probs spans the full vocab, so each rank multiplies its local vocab slice
    # [sc_vocab_start, sc_vocab_end) and the partials are summed across ranks.
    local_probs = probs[..., sc_vocab_start:sc_vocab_end].to(embed_weight.dtype)
    soft_embeds = torch.matmul(
        local_probs, embed_weight[: sc_vocab_end - sc_vocab_start]
    )
    if tp_size > 1:
        soft_embeds = torch.ops.vllm.all_reduce(soft_embeds, group_name=tp_group_name)
    soft_embeds = soft_embeds * normalizer
    sc_embeds[decode_slots] = soft_embeds * sc_keep

    # Overwrite canvas with argmax for newly converged denoise requests
    newly_converged = (converged & is_denoise).unsqueeze(1)
    canvas[decode_slots] = torch.where(
        newly_converged, argmax_canvas[decode_slots], canvas[decode_slots]
    )

    # ---- Phase 7: Copy canvas → draft_tokens for all slots ----
    draft_tokens[all_slots, :CL] = canvas[all_slots]

    return scaled


@torch.compile(dynamic=True)
def _compiled_sample_step_from_row_chunked(
    new_tokens: torch.Tensor,
    argmax_tokens: torch.Tensor,
    token_entropy: torch.Tensor,
    soft_embeds: torch.Tensor,
    decode_slots: torch.Tensor,
    decode_idx: torch.Tensor,
    all_slots: torch.Tensor,
    valid_canvas_len: torch.Tensor,
    canvas: torch.Tensor,
    argmax_canvas: torch.Tensor,
    step_tensor: torch.Tensor,
    is_encoder_phase: torch.Tensor,
    confident_tensor: torch.Tensor,
    sc_embeds: torch.Tensor,
    normalizer: torch.Tensor,
    history: torch.Tensor,
    history_len_tensor: torch.Tensor,
    sampled: torch.Tensor,
    num_sampled: torch.Tensor,
    draft_tokens: torch.Tensor,
    max_denoising_steps: float,
    confidence_threshold: float,
    vocab_size: int,
    CL: int,
    ST: int,
    entropy_bound: float,
) -> torch.Tensor:
    """Post-process row-chunked sampler outputs without full-vocab logits."""
    num_decode = decode_slots.shape[0]
    device = decode_slots.device

    sampled.zero_()
    num_sampled.zero_()

    mean_entropy = token_entropy.mean(dim=-1)
    confident_tensor[decode_slots] = mean_entropy < confidence_threshold

    sorted_ent, sorted_idx = torch.sort(token_entropy, dim=-1)
    cumsum_ent = torch.cumsum(sorted_ent, dim=-1)
    cummax_ent = torch.cummax(sorted_ent, dim=-1).values
    sorted_mask = (cumsum_ent - cummax_ent) <= entropy_bound
    eb_mask = torch.zeros_like(sorted_mask)
    eb_mask.scatter_(1, sorted_idx, sorted_mask)

    is_commit = is_encoder_phase[decode_slots]
    is_denoise = ~is_commit
    cur_step = step_tensor[decode_slots].float()

    new_step_val = torch.where(
        is_denoise,
        (cur_step + 1).to(step_tensor.dtype),
        step_tensor.new_zeros(num_decode),
    )
    step_tensor[decode_slots] = new_step_val

    random_tokens = torch.randint(
        0, vocab_size, (num_decode, CL), device=device, dtype=canvas.dtype
    )
    denoise_canvas = torch.where(eb_mask, new_tokens, random_tokens)
    canvas[decode_slots] = torch.where(
        is_commit.unsqueeze(1), random_tokens, denoise_canvas
    )

    hist_len = history_len_tensor[decode_slots]
    write_pos = hist_len % ST
    for i in range(ST):
        write_here = ((write_pos == i) & is_denoise).unsqueeze(1)
        history[decode_slots, i] = torch.where(
            write_here, argmax_tokens, history[decode_slots, i]
        )

    argmax_canvas[decode_slots] = torch.where(
        is_denoise.unsqueeze(1), argmax_tokens, argmax_canvas[decode_slots]
    )

    new_hist_len = torch.where(is_denoise, hist_len + 1, hist_len.new_zeros(num_decode))
    history_len_tensor[decode_slots] = new_hist_len

    sampled[decode_idx] = argmax_canvas[decode_slots].to(
        sampled.dtype
    ) * is_commit.unsqueeze(1).to(sampled.dtype)
    num_sampled[decode_idx] = is_commit.to(num_sampled.dtype) * valid_canvas_len.to(
        num_sampled.dtype
    )

    ref = history[decode_slots, 0]
    mismatch = torch.zeros(num_decode, device=device, dtype=torch.int32)
    for h in range(1, ST):
        mismatch = mismatch + (ref != history[decode_slots, h]).sum(dim=-1).int()
    stable = mismatch == 0

    step_after = step_tensor[decode_slots]
    converged = (stable & confident_tensor[decode_slots] & (new_hist_len >= ST)) | (
        step_after >= max_denoising_steps
    )
    is_encoder_phase[decode_slots] = torch.where(
        is_commit, is_commit.new_zeros(num_decode), converged
    )

    sc_keep = (is_denoise & ~is_encoder_phase[decode_slots])[:, None, None]
    sc_embeds[decode_slots] = (soft_embeds * normalizer * sc_keep).to(
        sc_embeds.dtype
    )

    newly_converged = (converged & is_denoise).unsqueeze(1)
    canvas[decode_slots] = torch.where(
        newly_converged, argmax_canvas[decode_slots], canvas[decode_slots]
    )
    draft_tokens[all_slots, :CL] = canvas[all_slots]

    return token_entropy.new_empty((0,))


class DiffusionGemmaRequestStates:
    """Pre-allocated GPU tensors for DiffusionGemma per-request state.

    Follows the indexed-slot pattern used by ``RequestState``.
    """

    def __init__(
        self,
        max_num_reqs: int,
        canvas_length: int,
        vocab_size: int,
        max_denoising_steps: int,
        device: torch.device,
        hidden_size: int,
        stability_threshold: int,
    ):
        self.max_num_reqs = max_num_reqs
        self.canvas_length = canvas_length
        self.vocab_size = vocab_size
        self.max_denoising_steps = max_denoising_steps
        self.stability_threshold = stability_threshold
        self.device = device

        self.is_encoder_phase = torch.zeros(
            max_num_reqs, dtype=torch.bool, device=device
        )
        # Canvas tokens [max_num_reqs, canvas_length]
        self.canvas = torch.zeros(
            max_num_reqs, canvas_length, dtype=torch.int64, device=device
        )
        # Step counter (counts up from 0 to max_denoising_steps)
        self.step = torch.zeros(
            max_num_reqs,
            dtype=torch.int32,
            device=device,
        )
        # Accepted canvas history for stability check
        self.accepted_canvas_history = torch.zeros(
            max_num_reqs,
            stability_threshold,
            canvas_length,
            dtype=torch.int64,
            device=device,
        )
        self.accepted_canvas_history_len = torch.zeros(
            max_num_reqs, dtype=torch.int32, device=device
        )
        # Latest argmax(processed_logits) per slot — what we COMMIT.
        # NOT `current_canvas` (which is the post-renoise stochastic input for
        # the next denoise step). We keep this separate from `canvas` because
        # canvas gets renoised in-place during denoise, while argmax_canvas is
        # the deterministic best-guess we ultimately emit.
        self.argmax_canvas = torch.zeros(
            max_num_reqs, canvas_length, dtype=torch.int64, device=device
        )

        # Per-slot prompt length (set by add_request).
        self.prompt_len = torch.zeros(
            max_num_reqs,
            dtype=torch.int32,
            device=device,
        )

        # Per-slot confidence flag, set by the sampler each step.
        self.confident = torch.zeros(max_num_reqs, dtype=torch.bool, device=device)

        # Per-slot self-conditioning soft embedding (probs @ embed_weight) from
        # the previous denoise step. Storing the [.., hidden] soft embed instead
        # of the full [.., vocab] distribution shrinks this buffer by
        # vocab/hidden (~170x) and moves the matmul to denoise time; the result
        # is identical (SC consumes probs @ embed_weight anyway).
        self.self_conditioning_embeds = torch.zeros(
            max_num_reqs, canvas_length, hidden_size, dtype=torch.float32, device=device
        )

    def init_canvas(self, slot_indices_np: np.ndarray) -> None:
        """Initialize canvas with random tokens for the given slots."""
        n = slot_indices_np.shape[0]
        self.canvas[slot_indices_np] = torch.randint(
            0,
            self.vocab_size,
            (n, self.canvas_length),
            dtype=torch.int64,
            device=self.device,
        )

    def add_request(self, slot_idx: int) -> None:
        self.is_encoder_phase[slot_idx] = True
        self.init_canvas(torch.tensor([slot_idx], device=self.device))
        self.step[slot_idx] = 0
        self.accepted_canvas_history_len[slot_idx] = 0
        self.self_conditioning_embeds[slot_idx] = 0

    def remove_request(self, slot_idx: int) -> None:
        self.is_encoder_phase[slot_idx] = False
        self.accepted_canvas_history_len[slot_idx] = 0
        self.self_conditioning_embeds[slot_idx] = 0


class DiffusionGemmaModelState(ModelState):
    """ModelState for DiffusionGemma.

    Single Gemma4 backbone in two modes:
    - encoder mode (num_draft_tokens == 0): causal attention, writes KV
    - decoder mode (num_draft_tokens > 0): bidirectional attention, reads KV
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: Any,
        device: torch.device,
    ) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)

        # Per-step MM data produced by get_mm_embeddings and consumed by
        # prepare_inputs.  Stored as raw (mm_embeds, is_mm_embed) so that
        # prepare_inputs can call embed_input_ids directly into the
        # persistent _inputs_embeds_buf, avoiding the intermediate copy
        # through encoder_runner.inputs_embeds.
        self._pending_mm_embeds: tuple[list[torch.Tensor], torch.Tensor] | None = None

        diffusion_config = vllm_config.diffusion_config
        canvas_length = diffusion_config.canvas_length if diffusion_config else 32

        text_config = self.model_config.hf_text_config
        self.gen_config = self.model_config.try_get_generation_config()
        max_denoising_steps = (
            diffusion_config.max_denoising_steps if diffusion_config else None
        ) or self.gen_config.get("max_denoising_steps", 48)
        self.diffusion_states = DiffusionGemmaRequestStates(
            max_num_reqs=self.max_num_reqs,
            canvas_length=canvas_length,
            vocab_size=self.model_config.get_vocab_size(),
            max_denoising_steps=max_denoising_steps,
            device=device,
            hidden_size=text_config.hidden_size,
            stability_threshold=self.gen_config["stability_threshold"],
        )
        self._req_id_to_index: dict[str, int] = {}

        # Persistent buffer for per-request causal flags, updated in-place
        # so FULL CUDA graph replay sees the latest values.
        self._causal_buf = torch.zeros(
            self.max_num_reqs, dtype=torch.bool, device=device
        )

        # Persistent inputs_embeds buffer — required so FULL CUDA graph
        # capture and runtime point at the SAME memory address.
        # `prepare_dummy_inputs` (capture path) and `prepare_inputs` (runtime
        # path) both must hand the captured graph a tensor at this address.
        self._inputs_embeds_buf = torch.zeros(
            self.max_num_tokens,
            text_config.hidden_size,
            dtype=self.model_config.dtype,
            device=device,
        )

    def get_supported_generation_tasks(self):
        return ("generate",)

    def get_extra_non_kv_cache_memory_bytes(self) -> int:
        reserve = _get_diffusion_gemma_sampler_memory_reserve_bytes(
            envs.VLLM_DIFFUSION_GEMMA_SAMPLER_MEMORY_RESERVE_MIB,
            envs.VLLM_DIFFUSION_GEMMA_SAMPLER_MEMORY_RESERVE_SCALE,
            max_num_seqs=self.max_num_reqs,
            max_num_batched_tokens=self.max_num_tokens,
            canvas_length=self.diffusion_states.canvas_length,
            # LogitsProcessor all-gathers vocab-parallel logits before the
            # materialized sampler, so reserve global-vocab scratch.
            vocab_size=self.model_config.get_vocab_size(),
        )
        if reserve > 0:
            logger.info_once(
                "DiffusionGemma sampler memory reserve: %s GiB "
                "(%s=%s, %s=%s)",
                reserve / (1 << 30),
                _DIFFUSION_GEMMA_SAMPLER_RESERVE_ENV,
                envs.VLLM_DIFFUSION_GEMMA_SAMPLER_MEMORY_RESERVE_MIB,
                _DIFFUSION_GEMMA_SAMPLER_RESERVE_SCALE_ENV,
                envs.VLLM_DIFFUSION_GEMMA_SAMPLER_MEMORY_RESERVE_SCALE,
            )
        return reserve

    def custom_sampler(self, sampler: Any) -> tuple[Any, Any] | None:
        diffusion_config = self.vllm_config.diffusion_config
        gen = self.gen_config
        sampler_cfg = gen.get("sampler_config") or {}
        if "EntropyBound" not in sampler_cfg.get("_cls_name", ""):
            raise ValueError("DiffusionGemma requires an EntropyBound sampler_config")
        entropy_bound = sampler_cfg.get("entropy_bound")
        if entropy_bound is None or entropy_bound <= 0:
            raise ValueError(
                f"entropy_bound must be a positive float (got {entropy_bound})"
            )
        # The self-conditioning matmul (probs @ embed_tokens.weight) runs over a
        # vocab-parallel embedding shard. Hand the sampler this rank's vocab
        # slice and TP group so it can all-reduce the partial products.
        embed_tokens = self.model.model.embed_tokens
        shard = embed_tokens.shard_indices
        tp_group = get_tp_group()
        return DiffusionSampler(
            sampler=sampler,
            diffusion_config=diffusion_config,
            vocab_size=self.model_config.get_vocab_size(),
            diffusion_states=self.diffusion_states,
            t_min=gen["t_min"],
            t_max=gen["t_max"],
            entropy_bound=entropy_bound,
            confidence_threshold=gen["confidence_threshold"],
            embed_weight=embed_tokens.weight,
            lm_head_weight=self.model.lm_head.weight,
            final_logit_softcapping=self.model.final_logit_softcapping,
            normalizer=self.model.model.normalizer,
            sampler_backend=self.model.sampler_backend,
            use_row_chunked_sampler=self.model.use_row_chunked_sampler,
            sc_vocab_start=shard.org_vocab_start_index,
            sc_vocab_end=shard.org_vocab_end_index,
            tp_size=tp_group.world_size,
            tp_group_name=tp_group.unique_name,
        ), None

    def apply_staged_writes(self) -> None:
        pass

    def add_request(self, req_index: int, new_req_data: Any) -> None:
        self._req_id_to_index[new_req_data.req_id] = req_index
        self.diffusion_states.add_request(req_index)
        if not new_req_data.req_id.startswith("_warmup_"):
            prompt_len = len(new_req_data.prompt_token_ids)
            self.diffusion_states.prompt_len[req_index] = prompt_len

    def remove_request(self, req_id: str) -> None:
        idx = self._req_id_to_index.pop(req_id, None)
        if idx is not None:
            self.diffusion_states.remove_request(idx)

    def get_mm_embeddings(
        self,
        scheduled_encoder_inputs: dict[str, list[int]],
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> torch.Tensor | None:
        if not self.supports_mm_inputs:
            return None

        mm_hashes, mm_kwargs = self.encoder_runner.prepare_mm_inputs(
            scheduled_encoder_inputs
        )
        if mm_kwargs:
            encoder_outputs = self.encoder_runner.execute_mm_encoder(mm_kwargs)
            self.encoder_cache.encoder_outputs.update(zip(mm_hashes, encoder_outputs))

        mm_embeds, is_mm_embed = self.gather_mm_embeddings(input_batch)

        if not mm_embeds:
            # No MM tokens in this batch (e.g. all-decode step).
            # prepare_inputs will use embed_input_ids (text-only) directly.
            self._pending_mm_embeds = None
            return None

        # Stash raw MM ingredients for prepare_inputs to merge directly
        # into the persistent buffer, avoiding the intermediate copy
        # through encoder_runner.inputs_embeds.
        self._pending_mm_embeds = (mm_embeds, is_mm_embed)
        return None

    def _apply_self_conditioning(
        self,
        decode_slots_np: np.ndarray,
        decode_idx_np: np.ndarray,
        query_start_loc_np: np.ndarray,
        inputs_embeds: torch.Tensor,
        sc_embeds: torch.Tensor,
    ) -> None:
        # One self-conditioning MLP call per decode request, over that request's
        # query span [start, end) = its canvas. The span is the full canvas (CL)
        # or, for the final canvas truncated near max_model_len, fewer than CL
        # positions. sc_embeds already holds probs @ embed_weight from the prior
        # denoise step, masked to zero by the sampler for slots not denoising
        # this step; only the MLP runs here. CPU metadata -> no GPU syncs.
        for slot, idx in zip(decode_slots_np.tolist(), decode_idx_np.tolist()):
            start = int(query_start_loc_np[idx])
            end = int(query_start_loc_np[idx + 1])
            canvas = slice(start, end)
            soft = sc_embeds[slot, : end - start]
            inputs_embeds[canvas] = self.model.self_conditioning(
                inputs_embeds[canvas], soft.to(inputs_embeds.dtype)
            )

    def prepare_inputs(self, input_batch, req_states) -> dict[str, Any]:
        states = self.diffusion_states
        num_tokens = input_batch.num_tokens
        num_reqs = input_batch.num_reqs

        # Write into the PERSISTENT inputs_embeds buffer so FULL CUDA graph
        # replay sees the latest values at the captured address.
        num_tokens_padded = input_batch.num_tokens_after_padding
        inputs_embeds = self._inputs_embeds_buf[:num_tokens_padded]

        # Populate embeddings: merge MM features when available,
        # otherwise embed input_ids as text-only.
        input_ids = input_batch.input_ids[:num_tokens]
        if self._pending_mm_embeds is not None:
            mm_embeds, is_mm_embed = self._pending_mm_embeds
            self._pending_mm_embeds = None
            inputs_embeds[:num_tokens].copy_(
                self.model.embed_input_ids(
                    input_ids,
                    multimodal_embeddings=mm_embeds,
                    is_multimodal=is_mm_embed,
                )
            )
        else:
            inputs_embeds[:num_tokens].copy_(self.model.embed_input_ids(input_ids))

        # Apply self-conditioning ONLY for denoising decode requests.
        if input_batch.num_draft_tokens > 0 and self._req_id_to_index:
            slots_np = input_batch.idx_mapping_np[:num_reqs]
            num_logits_np = np.diff(input_batch.cu_num_logits_np[: num_reqs + 1])
            is_decode_indices_np = np.where(num_logits_np > 0)[0]
            self._apply_self_conditioning(
                slots_np[is_decode_indices_np],
                is_decode_indices_np,
                input_batch.query_start_loc_np,
                inputs_embeds,
                states.self_conditioning_embeds,
            )

        return {"inputs_embeds": inputs_embeds}

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        # CUDA graph capture path — return a slice of the SAME persistent
        # inputs_embeds buffer that `prepare_inputs` writes to at runtime,
        # so the captured graph and runtime point to identical addresses.
        return {"inputs_embeds": self._inputs_embeds_buf[:num_tokens]}

    def postprocess_state(
        self, idx_mapping, num_sampled, num_computed_tokens=None
    ) -> None:
        return None

    def prepare_attn(
        self,
        input_batch,
        cudagraph_mode,
        block_tables,
        slot_mappings,
        attn_groups,
        kv_cache_config,
        for_capture=False,
    ) -> dict[str, Any]:
        if cudagraph_mode == CUDAGraphMode.FULL:
            num_reqs = input_batch.num_reqs_after_padding
            num_tokens = input_batch.num_tokens_after_padding
        else:
            num_reqs = input_batch.num_reqs
            num_tokens = input_batch.num_tokens

        query_start_loc_cpu = torch.from_numpy(input_batch.query_start_loc_np)
        max_query_len = input_batch.num_scheduled_tokens.max().item()

        # Per-request causal mode: encoder (commit) = causal,
        # denoise = bidirectional. Pass GPU tensor so the attention
        # backend can handle mixed batches.
        actual_num_reqs = input_batch.num_reqs
        slots = input_batch.idx_mapping[:actual_num_reqs]
        # Invariant: the sampler flips is_encoder_phase to False only after a
        # request's FINAL prompt chunk, so a prompt spanning multiple chunks
        # (longer than the token budget) stays causal for every chunk.
        self._causal_buf[:actual_num_reqs] = self.diffusion_states.is_encoder_phase[
            slots
        ]
        if actual_num_reqs < num_reqs:
            self._causal_buf[actual_num_reqs:num_reqs] = False
        causal: bool | torch.Tensor = self._causal_buf[:num_reqs]

        return build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=input_batch.query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=max_query_len,
            seq_lens=input_batch.seq_lens,
            max_seq_len=self.max_model_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            causal=causal,
        )

    num_new_sampled_tokens_per_step: int = 0


# Penalty stub for the diffusion path: the runner reads
# penalties_state.output_bin_counts, and post_update treats None as
# "no penalty bookkeeping".
_NO_PENALTIES_STATE = SimpleNamespace(output_bin_counts=None)


class DiffusionSampler:
    """Batched accept/renoise sampler for DiffusionGemma.

    Follows the same structure as ``vllm.v1.worker.gpu.sample.sampler.Sampler``:
    decomposed into named methods, all GPU state in pre-allocated buffers,
    no GPU→CPU syncs on the hot path.
    """

    def __init__(
        self,
        sampler: Any,
        diffusion_config: Any,
        vocab_size: int,
        diffusion_states: DiffusionGemmaRequestStates | None = None,
        *,
        confidence_threshold: float,
        t_min: float,
        t_max: float,
        entropy_bound: float,
        embed_weight: torch.Tensor,
        lm_head_weight: torch.Tensor,
        final_logit_softcapping: float | None,
        normalizer: torch.Tensor,
        sampler_backend: str = "materialized",
        use_row_chunked_sampler: bool = False,
        sc_vocab_start: int = 0,
        sc_vocab_end: int | None = None,
        tp_size: int = 1,
        tp_group_name: str = "",
    ):
        self.sampling_states = sampler.sampling_states
        self.req_states = sampler.req_states
        # Self-conditioning soft embed = probs @ embed_weight * normalizer,
        # computed in the sampler (see _compiled_sample_step). ``embed_weight``
        # is the vocab-parallel shard; [sc_vocab_start, sc_vocab_end) is this
        # rank's slice of the full vocab and tp_* drive the cross-rank
        # all-reduce.
        self.embed_weight = embed_weight
        self.lm_head_weight = lm_head_weight
        self.final_logit_softcapping = final_logit_softcapping
        self.normalizer = normalizer
        self.sampler_backend = sampler_backend
        self.use_row_chunked_sampler = use_row_chunked_sampler
        self._row_chunked_sampler_logged = False
        self.sc_vocab_start = sc_vocab_start
        self.sc_vocab_end = sc_vocab_end if sc_vocab_end is not None else vocab_size
        self.tp_size = tp_size
        self.tp_group_name = tp_group_name
        if self.use_row_chunked_sampler and self.tp_size != 1:
            raise ValueError(
                "DiffusionGemma row-chunked sampler backend currently supports "
                "tensor_parallel_size=1 only; use the materialized backend for TP."
            )
        self.canvas_length = (
            diffusion_config.canvas_length if diffusion_config is not None else 32
        )
        self.t_min = t_min
        self.t_max = t_max
        self.confidence_threshold = confidence_threshold
        self.vocab_size = vocab_size
        self.diffusion_states = diffusion_states
        self.entropy_bound = entropy_bound

        max_num_reqs = diffusion_states.max_num_reqs
        device = diffusion_states.device
        self._sampled = torch.zeros(
            max_num_reqs,
            self.canvas_length,
            dtype=torch.int32,
            device=device,
        )
        self._num_sampled = torch.zeros(
            max_num_reqs,
            dtype=torch.int32,
            device=device,
        )
        self._decode_slots = UvaBackedTensor(max_num_reqs, dtype=torch.int64)
        self._decode_idx = UvaBackedTensor(max_num_reqs, dtype=torch.int64)
        self._query_lens = UvaBackedTensor(max_num_reqs, dtype=torch.int32)
        self._num_logits = UvaBackedTensor(max_num_reqs, dtype=torch.int32)

        # Per-slot stash for logprobs computed on the converging denoise step.
        # Populated after the post-sample kernel detects convergence; consumed
        # on the subsequent commit step when num_sampled=CANVAS_LEN.
        self._pending_logprobs: dict[int, LogprobsTensors] = {}
        self._row_chunk_token_offsets: torch.Tensor | None = None
        self._canvas_offsets: torch.Tensor | None = None

    def _get_row_chunk_token_offsets(self, device: torch.device) -> torch.Tensor:
        token_offsets = self._row_chunk_token_offsets
        if (
            token_offsets is None
            or token_offsets.device != device
            or token_offsets.shape[0] != self.vocab_size
        ):
            token_offsets = torch.arange(
                self.vocab_size, device=device, dtype=torch.int64
            )
            self._row_chunk_token_offsets = token_offsets
        return token_offsets

    def _get_canvas_offsets(self, device: torch.device, length: int) -> torch.Tensor:
        canvas_offsets = self._canvas_offsets
        if (
            canvas_offsets is None
            or canvas_offsets.device != device
            or canvas_offsets.shape[0] != length
        ):
            canvas_offsets = torch.arange(length, device=device, dtype=torch.int64)
            self._canvas_offsets = canvas_offsets
        return canvas_offsets

    def add_request(self, req_idx: int, prompt_len: int, sampling_params: Any) -> None:
        if use_penalty(sampling_params):
            logger.warning_once(
                "DiffusionGemma does not support repetition/frequency/presence "
                "penalties; ignoring them for this request."
            )
        # Purge any stale logprobs stashed under this slot by a prior request
        # that was aborted between its converging denoise and commit steps.
        self._pending_logprobs.pop(req_idx, None)
        self.sampling_states.add_request(req_idx, sampling_params)

    def apply_staged_writes(self) -> None:
        self.sampling_states.apply_staged_writes()

    @property
    def penalties_state(self):
        # Diffusion applies no penalties. The runner reads
        # penalties_state.output_bin_counts, so expose a stub holding None;
        # post_update treats None bin counts as "no penalty bookkeeping".
        return _NO_PENALTIES_STATE

    # ------------------------------------------------------------------
    # Prefill
    # ------------------------------------------------------------------

    def _finish_prefills(
        self, input_batch: Any, prefill_indices_np: np.ndarray
    ) -> None:
        """Transition requests whose prompt completes this step to denoising.

        Initializes their canvas, seeds draft tokens, and flips
        is_encoder_phase to False. Mid-chunk requests (prompt longer than the
        token budget) are left untouched so is_encoder_phase stays True and
        prepare_attn keeps causal attention for their remaining chunks.
        """
        states = self.diffusion_states
        done_prefill_np = (
            input_batch.num_computed_prefill_tokens_np[prefill_indices_np]
            + input_batch.num_scheduled_tokens[prefill_indices_np]
            >= input_batch.prefill_len_np[prefill_indices_np]
        )
        ps = input_batch.idx_mapping_np[prefill_indices_np[done_prefill_np]]
        if len(ps) == 0:
            return
        states.init_canvas(ps)
        self.req_states.draft_tokens[ps, : self.canvas_length] = states.canvas[ps]
        ps_gpu = async_copy_to_gpu(
            ps.astype(np.int64), device=states.is_encoder_phase.device
        )
        states.is_encoder_phase.index_fill_(0, ps_gpu, False)

    def _handle_prefill(
        self,
        input_batch: Any,
        device: torch.device,
    ) -> SamplerOutput:
        num_reqs = input_batch.num_reqs
        self._finish_prefills(input_batch, np.arange(num_reqs))
        sampled = self._sampled[:num_reqs, :1]
        sampled.zero_()
        num_sampled = self._num_sampled[:num_reqs]
        num_sampled.zero_()
        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=num_sampled,
            num_rejected=num_sampled,
        )

    # ------------------------------------------------------------------
    # Decode helpers
    # ------------------------------------------------------------------

    def _build_output(
        self,
        input_batch: Any,
        sampled: torch.Tensor,
        num_sampled: torch.Tensor,
        per_req_nlogits_np: np.ndarray,
        device: torch.device,
        logprobs_tensors: LogprobsTensors | None = None,
    ) -> SamplerOutput:
        """Compute num_rejected and build SamplerOutput."""
        num_reqs = input_batch.num_reqs

        self._query_lens.np[:num_reqs] = np.diff(
            input_batch.query_start_loc_np[: num_reqs + 1]
        )
        self._num_logits.np[:num_reqs] = per_req_nlogits_np
        self._query_lens.copy_to_uva()
        self._num_logits.copy_to_uva()

        num_rejected = _compute_num_rejected(
            self._num_logits.gpu[:num_reqs],
            num_sampled,
            input_batch.query_start_loc[: num_reqs + 1],
        )

        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=logprobs_tensors,
            num_nans=None,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def __call__(
        self,
        logits: torch.Tensor,
        input_batch: Any,
        draft_logits: torch.Tensor | None = None,
    ) -> SamplerOutput:
        num_reqs = input_batch.num_reqs
        device = logits.device

        if input_batch.num_draft_tokens == 0:
            return self._handle_prefill(input_batch, device)

        # --- CPU/NumPy setup (outside compile): split decode vs prefill, init
        # canvas for any new prefills, and stage decode slot indices to GPU. ---
        states = self.diffusion_states
        CL = self.canvas_length
        slots_np = input_batch.idx_mapping_np[:num_reqs]
        per_req_nlogits_np = np.diff(input_batch.cu_num_logits_np[: num_reqs + 1])

        decode_indices_np = np.where(per_req_nlogits_np > 0)[0]
        prefill_indices_np = np.where(per_req_nlogits_np == 0)[0]
        decode_slots_np = slots_np[decode_indices_np]

        if len(prefill_indices_np) > 0:
            self._finish_prefills(input_batch, prefill_indices_np)

        num_decode = len(decode_indices_np)
        self._decode_slots.np[:num_decode] = decode_slots_np
        self._decode_idx.np[:num_decode] = decode_indices_np
        self._decode_slots.copy_to_uva()
        self._decode_idx.copy_to_uva()
        decode_slots = self._decode_slots.gpu[:num_decode]
        decode_idx = self._decode_idx.gpu[:num_decode]

        # Real canvas length per decode request. Equals CL except when a canvas
        # was truncated near max_model_len, in which case the scheduler gave us
        # fewer than CL logits for that request.
        valid_canvas_len_np = per_req_nlogits_np[per_req_nlogits_np > 0]
        valid_canvas_len = async_copy_to_gpu(
            valid_canvas_len_np.astype(np.int64), device=device
        )

        # Pad any truncated canvas back to CL so the uniform-CL sampler math
        # holds. Phantom (padded) positions are zeroed → uniform logits → high
        # entropy (no premature convergence) and argmax 0 (stable); they are
        # never committed (num_sampled == real length).
        if num_decode > 0 and valid_canvas_len_np.min() < CL:
            ar = self._get_canvas_offsets(device, CL)
            starts = valid_canvas_len.cumsum(0) - valid_canvas_len  # row offset per req
            valid = ar.unsqueeze(0) < valid_canvas_len.unsqueeze(1)  # [num_decode, CL]
            src = (starts.unsqueeze(1) + ar.unsqueeze(0)).clamp_max(logits.shape[0] - 1)
            logits = logits[src.reshape(-1)] * valid.reshape(-1, 1).to(logits.dtype)

        # Cleared inside _compiled_sample_step so prefill/non-decode slots stay 0.
        sampled = self._sampled[:num_reqs]
        num_sampled = self._num_sampled[:num_reqs]

        all_slots = input_batch.idx_mapping[:num_reqs]

        # Snapshot which slots are committing BEFORE the compiled step runs,
        # since it mutates is_encoder_phase (commit→False, converge→True).
        is_committing = states.is_encoder_phase[decode_slots].clone()

        slots_np = input_batch.idx_mapping_np[:num_reqs]
        max_num_logprobs = self.sampling_states.max_num_logprobs(slots_np)
        is_sampler_warmup = _is_diffusion_gemma_sampler_warmup_batch(input_batch)
        real_logprobs_enabled = max_num_logprobs >= 0 and not is_sampler_warmup
        row_chunked_input = False
        if self.use_row_chunked_sampler:
            row_chunked_input = _is_diffusion_gemma_hidden_state_input(
                logits, self.vocab_size, self.lm_head_weight.shape[1]
            )
        sampler_rows = num_decode * CL
        use_row_chunked = _should_use_diffusion_gemma_row_chunked_sampler(
            row_chunked_input=row_chunked_input,
            real_logprobs_enabled=real_logprobs_enabled,
        )
        if use_row_chunked:
            row_chunk_size = _get_diffusion_gemma_row_chunk_size(
                sampler_rows, self.vocab_size
            )
            if not self._row_chunked_sampler_logged:
                logger.warning_once(
                    "DiffusionGemma row-chunked sampler path is active "
                    "(row_chunk_env=%d, row_chunk_scratch_mib=%d, "
                    "resolved_row_chunk=%d).",
                    envs.VLLM_DIFFUSION_GEMMA_ROW_CHUNK,
                    envs.VLLM_DIFFUSION_GEMMA_ROW_CHUNK_SCRATCH_MIB,
                    row_chunk_size,
                )
                self._row_chunked_sampler_logged = True

            steps_f = states.step[decode_slots].float()
            remaining = (float(states.max_denoising_steps) - steps_f).clamp(min=1.0)
            temp = self.t_min + (self.t_max - self.t_min) * (
                remaining / float(states.max_denoising_steps)
            )
            temperature = temp[:, None].expand(num_decode, CL).reshape(-1)
            canvas_offsets = self._get_canvas_offsets(device, CL)
            request_seeds = self.sampling_states.seeds.gpu[decode_slots].to(
                torch.int64
            )
            step_offsets = states.step[decode_slots].to(torch.int64)
            row_seed_offsets = (
                (request_seeds[:, None] + (step_offsets[:, None] + 1) * 1_000_003)
                * CL
                + canvas_offsets[None, :]
            ).reshape(-1)

            _, token_entropy, new_tokens, argmax_tokens, soft_embeds = (
                diffusion_gemma_softcap_row_chunked_sample_soft_embeds(
                    logits,
                    self.lm_head_weight[: self.vocab_size],
                    self.embed_weight[: self.vocab_size],
                    float(self.final_logit_softcapping),
                    temperature,
                    # Request seeds are already folded into row_seed_offsets.
                    0,
                    row_chunk_size=row_chunk_size,
                    row_seed_offsets=row_seed_offsets,
                    token_offsets=self._get_row_chunk_token_offsets(device),
                    temperature_is_positive=min(self.t_min, self.t_max) > 0,
                )
            )
            scaled = _compiled_sample_step_from_row_chunked(
                new_tokens.view(num_decode, CL),
                argmax_tokens.view(num_decode, CL),
                token_entropy.view(num_decode, CL),
                soft_embeds.view(num_decode, CL, -1),
                decode_slots,
                decode_idx,
                all_slots,
                valid_canvas_len,
                states.canvas,
                states.argmax_canvas,
                states.step,
                states.is_encoder_phase,
                states.confident,
                states.self_conditioning_embeds,
                self.normalizer,
                states.accepted_canvas_history,
                states.accepted_canvas_history_len,
                sampled,
                num_sampled,
                self.req_states.draft_tokens,
                max_denoising_steps=float(states.max_denoising_steps),
                confidence_threshold=self.confidence_threshold,
                vocab_size=self.vocab_size,
                CL=self.canvas_length,
                ST=states.stability_threshold,
                entropy_bound=self.entropy_bound,
            )
        else:
            if row_chunked_input:
                logger.warning_once(
                    "DiffusionGemma row-chunked sampler disabled because this "
                    "batch needs logprobs; falling back to materialized logits "
                    "for correctness."
                )
                logits = _materialize_diffusion_gemma_logits_from_hidden(
                    logits,
                    self.lm_head_weight,
                    self.vocab_size,
                    self.final_logit_softcapping,
                )

            # --- Single compiled call: temp → sample → probs → post-process ---
            scaled = _compiled_sample_step(
                logits,
                decode_slots,
                decode_idx,
                all_slots,
                valid_canvas_len,
                # State
                states.canvas,
                states.argmax_canvas,
                states.step,
                states.is_encoder_phase,
                states.confident,
                states.self_conditioning_embeds,
                self.embed_weight,
                self.normalizer,
                states.accepted_canvas_history,
                states.accepted_canvas_history_len,
                # Output
                sampled,
                num_sampled,
                self.req_states.draft_tokens,
                # Config
                max_denoising_steps=float(states.max_denoising_steps),
                t_min=self.t_min,
                t_max=self.t_max,
                confidence_threshold=self.confidence_threshold,
                vocab_size=self.vocab_size,
                CL=self.canvas_length,
                ST=states.stability_threshold,
                entropy_bound=self.entropy_bound,
                sc_vocab_start=self.sc_vocab_start,
                sc_vocab_end=self.sc_vocab_end,
                tp_size=self.tp_size,
                tp_group_name=self.tp_group_name,
            )

        # --- Logprobs: stash on convergence, return on commit ---
        is_decode_np = per_req_nlogits_np > 0

        logprobs_tensors = None
        if real_logprobs_enabled:
            # Denoise steps that just converged: the compiled step flipped
            # is_encoder_phase from False→True. Detect as slots where
            # is_encoder_phase is now True but is_committing was False.
            converged_mask = states.is_encoder_phase[decode_slots]
            just_converged = converged_mask & ~is_committing
            if just_converged.any():
                flat_logits = scaled.reshape(-1, scaled.shape[-1])
                argmax_tokens = scaled.argmax(dim=-1)
                for local_idx in just_converged.nonzero(as_tuple=True)[0]:
                    li = local_idx.item()
                    slot = decode_slots[local_idx]
                    # Stash only the real canvas positions (== CL unless this
                    # canvas was truncated near max_model_len); padded tail
                    # positions are never emitted.
                    k_i = int(valid_canvas_len_np[li])
                    start = li * CL
                    self._pending_logprobs[slot.item()] = compute_topk_logprobs(
                        flat_logits[start : start + k_i],
                        max_num_logprobs,
                        argmax_tokens[local_idx][:k_i],
                    )

            # Commit steps: is_committing was True at entry. Reassemble
            # previously stashed logprobs and attach to SamplerOutput.
            if is_committing.any() and self._pending_logprobs:
                parts_ids, parts_lp, parts_ranks = [], [], []
                cu_gen: list[int] = []
                flat_offset = 0
                for i in range(num_reqs):
                    cu_gen.append(flat_offset)
                    slot = int(slots_np[i])
                    if is_decode_np[i] and slot in self._pending_logprobs:
                        lp = self._pending_logprobs.pop(slot)
                        parts_ids.append(lp.logprob_token_ids)
                        parts_lp.append(lp.logprobs)
                        parts_ranks.append(lp.selected_token_ranks)
                        flat_offset += lp.logprobs.shape[0]
                if parts_ids:
                    logprobs_tensors = LogprobsTensors(
                        logprob_token_ids=torch.cat(parts_ids),
                        logprobs=torch.cat(parts_lp),
                        selected_token_ranks=torch.cat(parts_ranks),
                        cu_num_generated_tokens=cu_gen,
                    )

        return self._build_output(
            input_batch,
            sampled,
            num_sampled,
            per_req_nlogits_np,
            device,
            logprobs_tensors=logprobs_tensors,
        )
