# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-token prediction (MTP) head for HY V4 (NVIDIA)."""

import copy
import typing
from collections.abc import Callable, Iterable

import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import fused_moe_make_expert_params_mapping
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.deepseek_v2 import _try_load_fp8_indexer_wk
from vllm.model_executor.models.utils import (
    get_pp_missing_layer_names,
    is_pp_missing_parameter,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from .model import HYV4DecoderLayer, _normalize_hyv4_config

logger = init_logger(__name__)

_MTP_SCALE_ONLY_PARAM_SUBSTRINGS = (
    "_scale",
    "_weight_scale",
    "_weight_scale_inv",
    "_input_scale",
)

# Names of the quant-config attribute that stores the unquantized-module
# prefix list, in the order we look them up. Different QuantizationConfig
# subclasses expose the list under different names:
#   * ``ignored_layers`` -- Fp8Config, GPTQ / AWQ configs.
#   * ``exclude_modules`` -- ModelOpt* configs (Fp8, NvFp4, MxFp8, Mixed).
_MTP_QUANT_EXCLUSION_ATTRS = ("ignored_layers", "exclude_modules")


def _get_spec_layer_idx_from_weight_name(
    config: PretrainedConfig, weight_name: str
) -> int | None:
    """Return the MTP layer index a checkpoint weight belongs to, or None.

    Compatible with``num_nextn_predict_layers`` of 1 and 2 for a single MTP
    head, matching how the HY V4 checkpoints are exported.
    """
    if getattr(config, "num_nextn_predict_layers", 0) < 1:
        return None
    layer_idx = config.num_hidden_layers
    num_mtp_layers = max(config.num_nextn_predict_layers - 1, 1)
    for i in range(num_mtp_layers):
        if weight_name.startswith(f"model.layers.{layer_idx + i}."):
            return layer_idx + i
    return None


def _should_skip_missing_mtp_scale_param(
    quant_config: QuantizationConfig | None,
    name: str,
) -> bool:
    """Whether an unmatched scale parameter can be silently ignored."""
    return quant_config is None and any(
        scale_name in name for scale_name in _MTP_SCALE_ONLY_PARAM_SUBSTRINGS
    )


def _resolve_fused_expert_param(
    param_base: str,
    ckpt_suffix: str,
    params_dict: dict,
) -> str | None:
    """Map a fused expert checkpoint suffix onto a draft parameter name.

    Fused checkpoints pack every expert into one tensor and name the companion
    scales by suffixing the projection (``experts.gate_up_proj_scale_inv``),
    while the draft model owns ``experts.routed_experts.w13_weight`` plus a
    separate ``..._scale_inv`` parameter. Dropping the suffix would push the
    scale into the weight parameter and leave the scale at its sentinel init.

    Args:
        param_base: Draft parameter holding the packed weight.
        ckpt_suffix: Checkpoint text following the projection name.
        params_dict: The draft model's named parameters.

    Returns:
        The matching draft parameter name, or None when there is none.
    """
    if not ckpt_suffix:
        return param_base if param_base in params_dict else None
    tail = ckpt_suffix.lstrip(".").removeprefix("weight_")
    for candidate in (
        f"{param_base}_{tail}",
        f"{param_base}_scale_inv",
        f"{param_base}_scale",
    ):
        if candidate in params_dict:
            return candidate
    return None


def _prepare_mtp_fp8_expert_scale(
    quant_config: QuantizationConfig | None,
    name: str,
    loaded_weight: torch.Tensor,
) -> tuple[str, torch.Tensor]:
    """Normalize block-wise FP8 expert scale names and dtypes."""
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    if (
        isinstance(quant_config, Fp8Config)
        and quant_config.weight_block_size is not None
        and ".mlp.experts." in name
        and name.endswith(".scale")
    ):
        name = name[: -len(".scale")] + ".weight_scale_inv"
        if (
            getattr(quant_config, "is_scale_e8m0", False)
            and loaded_weight.dtype == torch.uint8
        ):
            # UE8M0 scale exponents are stored as raw uint8 bytes: reinterpret
            # the bits instead of numerically converting them.
            loaded_weight = loaded_weight.view(torch.float8_e8m0fnu)
    return name, loaded_weight


def _create_mtp_quant_config(
    hf_config: PretrainedConfig,
    backbone_quant_config: QuantizationConfig | None = None,
) -> QuantizationConfig | None:
    """Create the quantization config for the MTP layers.

    The MTP quantization algorithm is given by the ``mtp_quant_algo`` field in
    ``config.json``, independently of the backbone's quantization. Supported
    values are ``"FP8"``, ``"NONE"`` (inherit the backbone) and ``"BF16"`` /
    ``"FP16"`` (unquantized). A missing or ``"NONE"`` value falls back to the
    backbone config.

    Args:
        hf_config: The draft model's HF config.
        backbone_quant_config: The target model's quantization config.

    Returns:
        The quantization config to use for the MTP layers, or None when the MTP
        layers are unquantized.
    """
    mtp_quant_algo = getattr(hf_config, "mtp_quant_algo", None)

    if mtp_quant_algo is None or mtp_quant_algo.upper() == "NONE":
        logger.info_once(
            "No explicit MTP quant algo (mtp_quant_algo=%s). "
            "Using the backbone's quant config.",
            mtp_quant_algo,
        )
        return backbone_quant_config

    mtp_quant_algo = mtp_quant_algo.upper()

    if mtp_quant_algo in ("BF16", "FP16"):
        logger.info_once(
            "MTP layers are unquantized (mtp_quant_algo=%s).", mtp_quant_algo
        )
        return None

    if mtp_quant_algo == "FP8":
        from vllm.model_executor.layers.quantization.fp8 import Fp8Config

        logger.info_once("MTP uses FP8 quantization (mtp_quant_algo=FP8).")
        hf_quant_config = getattr(hf_config, "quantization_config", None) or {}
        weight_block_size = hf_quant_config.get("weight_block_size")
        activation_scheme = hf_quant_config.get("activation_scheme", "dynamic")
        if weight_block_size is not None:
            activation_scheme = "dynamic"
        ignored_layers = hf_quant_config.get("ignored_layers") or hf_quant_config.get(
            "modules_to_not_convert"
        )

        fp8_config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme=activation_scheme,
            ignored_layers=ignored_layers or [],
            weight_block_size=weight_block_size,
        )
        if hf_quant_config.get("scale_fmt") == "ue8m0":
            # Fp8LinearMethod reads this through getattr, so it is not a
            # declared Fp8Config field.
            setattr(fp8_config, "is_scale_e8m0", True)  # noqa: B010
        return fp8_config

    logger.warning(
        "Unknown mtp_quant_algo=%s, falling back to the backbone config.",
        mtp_quant_algo,
    )
    return backbone_quant_config


def _remap_mtp_quant_exclusions(
    quant_config: QuantizationConfig | None,
    mtp_start_layer_idx: int,
    num_mtp_layers: int,
) -> QuantizationConfig | None:
    """Translate checkpoint-named MTP quant exclusions to draft prefixes.

    ``modules_to_not_convert`` / ``exclude_modules`` name MTP modules the way
    the checkpoint does (``model.mtp_layers.0.self_attn.linear_gate``), while
    the draft model builds them under ``model.layers.<num_hidden_layers + i>``.
    ``is_layer_skipped`` compares prefixes for exact equality, so without this
    translation an excluded MTP module gets a quant method even though its
    checkpoint weight is BF16 with no ``weight_scale`` / ``weight_scale_inv``
    companion: the scale then keeps its ``finfo(float32).min`` sentinel and the
    layer silently produces garbage.

    The list lives under a different attribute per config class, hence the
    lookup over ``_MTP_QUANT_EXCLUSION_ATTRS``.

    Args:
        quant_config: The MTP quantization config to widen.
        mtp_start_layer_idx: Index of the first MTP layer in the draft model.
        num_mtp_layers: Number of MTP layers.

    Returns:
        A shallow copy with the translated exclusions, or the input unchanged.
    """
    if quant_config is None:
        return None

    # Different QuantizationConfig subclasses expose the exclusion list under
    # different attribute names; only the one that is actually populated on
    # the backbone config governs which draft modules stay unquantized. For
    # ModelOpt checkpoints (``quant_method: modelopt`` and friends) this is
    # ``exclude_modules``; for the legacy fp8 path it is ``ignored_layers``.
    exclusion_attr: str | None = None
    patterns: list[str] | None = None
    for attr in _MTP_QUANT_EXCLUSION_ATTRS:
        value = getattr(quant_config, attr, None)
        if value:
            exclusion_attr = attr
            patterns = list(value)
            break
    if not patterns or exclusion_attr is None:
        return quant_config

    extra: list[str] = []
    for offset in range(num_mtp_layers):
        source_prefix = f"model.mtp_layers.{offset}."
        draft_prefix = f"model.layers.{mtp_start_layer_idx + offset}."
        for pattern in patterns:
            if pattern.startswith(source_prefix):
                extra.append(draft_prefix + pattern[len(source_prefix) :])
            # ModelOpt exclusion patterns may end with a ``*`` wildcard; the
            # remap must preserve that suffix so ``is_layer_skipped`` still
            # matches the draft module prefixes.
            elif pattern.rstrip("*").startswith(source_prefix):
                stripped = pattern.rstrip("*")
                trailing_stars = pattern[len(stripped) :]
                extra.append(
                    draft_prefix + stripped[len(source_prefix) :] + trailing_stars
                )
    if not extra:
        return quant_config

    # The backbone shares this object with the target model; copy before
    # widening the exclusion list. The attribute is only present on the
    # concrete configs that support exclusions, hence the getattr/setattr pair.
    quant_config = copy.copy(quant_config)
    setattr(quant_config, exclusion_attr, patterns + extra)  # noqa: B010
    logger.info_once(
        "HYV4 MTP: mapped %d checkpoint-named quant exclusions (%s) onto the "
        "draft module prefixes: %s",
        len(extra),
        exclusion_attr,
        ", ".join(extra),
    )
    return quant_config


class HYV4SharedHead(nn.Module):
    """Holds the draft LM head shared with the target model."""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        # Use the same ``lm_head`` prefix as the backbone so that quant-config
        # exclusion lists (e.g. ModelOpt's ``exclude_modules: ["lm_head", ...]``)
        # also apply here: an unquantized checkpoint lm_head must not gain a
        # spurious ``weight_scale`` parameter on the draft side, which would
        # otherwise stay at its FP-min sentinel until the proposer swaps the
        # whole ParallelLMHead for the target model's ``lm_head``.
        self.head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix="lm_head",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states


class HYV4MultiTokenPredictorLayer(nn.Module):
    """A single MTP draft block."""

    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        vllm_config: VllmConfig,
        model_config: ModelConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        del model_config  # kept for signature parity with the other MTP heads
        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.shared_head = HYV4SharedHead(config=config, quant_config=quant_config)

        # HYV4DecoderLayer indexes layer_types/mlp_layer_types by the numeric
        # layer id parsed from the prefix. MTP layers live after the backbone
        # layers, so extend the per-layer config lists to cover those ids.
        layer_idx = int(prefix.rsplit(".", 1)[-1])
        mtp_config = copy.deepcopy(config)
        # MTP checkpoints do not carry iHC pre/post weights for mtp_layers, so
        # the draft blocks are built without iHC to match their structure.
        mtp_config.enable_ihc = False
        mtp_config.layer_types = _extend_layer_types(
            getattr(mtp_config, "layer_types", None), layer_idx, "full_attention"
        )
        mtp_config.mlp_layer_types = _extend_layer_types(
            getattr(mtp_config, "mlp_layer_types", None), layer_idx, "sparse"
        )

        self.mtp_block = HYV4DecoderLayer(
            config=mtp_config,
            vllm_config=vllm_config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
            topk_indices_buffer=topk_indices_buffer,
        )
        self.final_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del input_ids  # embeddings are resolved by the caller
        assert inputs_embeds is not None
        inputs_embeds = self.enorm(inputs_embeds)
        previous_hidden_states = self.hnorm(previous_hidden_states)
        hidden_states = self.eh_proj(
            torch.cat([inputs_embeds, previous_hidden_states], dim=-1)
        )
        hidden_states, residual = self.mtp_block(
            positions=positions,
            hidden_states=hidden_states,
            residual=None,
        )
        hidden_states, _ = self.final_layernorm(hidden_states, residual)
        return hidden_states


def _extend_layer_types(
    layer_types: list[str] | None, layer_idx: int, fallback: str
) -> list[str] | None:
    """Pad a per-layer config list so``layer_idx`` is addressable."""
    if layer_types is None or len(layer_types) > layer_idx:
        return layer_types
    fill = layer_types[-1] if layer_types else fallback
    return list(layer_types) + [fill] * (layer_idx + 1 - len(layer_types))


class HYV4MultiTokenPredictor(nn.Module):
    """Owns the MTP draft blocks and their shared embedding / logits path."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        target_config = vllm_config.model_config.hf_config
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None, (
            "HYV4MTP can only be built as a speculative draft model"
        )
        draft_model_config = speculative_config.draft_model_config
        assert draft_model_config is not None
        config = _normalize_hyv4_config(draft_model_config.hf_config)
        self.mtp_start_layer_idx = target_config.num_hidden_layers
        self.num_mtp_layers = max(
            getattr(target_config, "num_nextn_predict_layers", 1), 1
        )
        self.quant_config = _remap_mtp_quant_exclusions(
            _create_mtp_quant_config(config, vllm_config.quant_config),
            self.mtp_start_layer_idx,
            self.num_mtp_layers,
        )
        if self.quant_config is not None:
            self.quant_config.packed_modules_mapping = HYV4MTP.packed_modules_mapping

        # Sparse DSA draft blocks must share the target model's top-k buffer.
        # The proposer injects it through HYV4MTP.set_topk_indices_buffer()
        # before any draft forward.
        self.topk_indices_buffer: torch.Tensor | None = None
        if hasattr(config, "index_topk"):
            from vllm.platforms import current_platform

            self.topk_indices_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                config.index_topk,
                dtype=torch.int32,
                device=current_platform.device_type,
            )

        self.layers = nn.ModuleDict(
            {
                str(idx): HYV4MultiTokenPredictorLayer(
                    config,
                    f"{prefix}.layers.{idx}",
                    vllm_config=vllm_config,
                    model_config=draft_model_config,
                    cache_config=vllm_config.cache_config,
                    quant_config=self.quant_config,
                    topk_indices_buffer=self.topk_indices_buffer,
                )
                for idx in range(
                    self.mtp_start_layer_idx,
                    self.mtp_start_layer_idx + self.num_mtp_layers,
                )
            }
        )
        self.requires_topk_indices_buffer = any(
            layer.mtp_block.self_attn.is_sparse for layer in self.layers.values()
        )

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.spec_step_idx: int = 0

    def set_skip_topk(self, skip: bool) -> None:
        """Toggle the draft indexer for ``index_share_for_mtp_iteration``.

        The proposer clears the flag for draft step 0 so the MTP layer builds
        its own top-k indices, then sets it for steps 1+ so they reuse what
        step 0 wrote into the shared buffer instead of re-running the indexer.
        """
        for layer in self.layers.values():
            layer.mtp_block.self_attn.skip_topk = skip

    def compact_topk_indices(self, slot_ids: torch.Tensor) -> None:
        """Move the top-k rows at ``slot_ids`` to the front of the buffer.

        Step 0 writes one row per query token of the multi-token batch, while
        steps 1+ decode a single token per request and index the buffer from 0.
        Without this gather they would read another token's rows.
        """
        num_slots = slot_ids.numel()
        if self.topk_indices_buffer is None or num_slots == 0:
            return
        self.topk_indices_buffer[:num_slots] = self.topk_indices_buffer[slot_ids]

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        inputs_embeds = torch.where((positions == 0).unsqueeze(-1), 0, inputs_embeds)

        current_step_idx = self.spec_step_idx % self.num_mtp_layers
        return self.layers[str(self.mtp_start_layer_idx + current_step_idx)](
            input_ids,
            positions,
            previous_hidden_states,
            inputs_embeds,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        current_step_idx = self.spec_step_idx % self.num_mtp_layers
        mtp_layer = self.layers[str(self.mtp_start_layer_idx + current_step_idx)]
        lm_head = mtp_layer.shared_head.head
        proj_input = mtp_layer.shared_head(hidden_states)
        return self.logits_processor(lm_head, proj_input)


class HYV4MTP(nn.Module):
    """HY V4 MTP draft head.

    Not a pipeline-parallel stage: the draft head always runs on a single rank,
    matching `DeepseekV32MTP` / `KimiK3MTP` / `HYV3MTP`.
    """

    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
        # MLA runs both latent down-projections as one GEMM.
        "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
        # The indexer fuses wk and weights_proj into one GEMM.
        "wk_weights_proj": ["wk", "weights_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.model = HYV4MultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.quant_config = self.model.quant_config
        self.sampler = Sampler()

    def set_topk_indices_buffer(self, topk_indices_buffer: torch.Tensor) -> None:
        """Share the target sparse-index buffer with every draft consumer.

        Proposers that walk ``named_modules()`` instead of calling this reach
        the same consumers via ``HYV4MLAAttention.topk_indices_buffer``.
        """
        self.model.topk_indices_buffer = topk_indices_buffer
        for layer in self.model.layers.values():
            self_attn = layer.mtp_block.self_attn
            if not self_attn.is_sparse:
                continue

            indexer = self_attn.indexer
            assert indexer is not None, "Sparse HYV4 MTP attention requires an indexer"
            indexer.topk_indices_buffer = topk_indices_buffer
            indexer.indexer_op.topk_indices_buffer = topk_indices_buffer

            attn_impl = self_attn.mla_attn.impl
            assert hasattr(attn_impl, "topk_indices_buffer"), (
                "Sparse HYV4 MTP attention backend requires a top-k indices buffer"
            )
            attn_impl.topk_indices_buffer = topk_indices_buffer

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        del intermediate_tensors  # the MTP head is single-stage
        if (
            self.model.requires_topk_indices_buffer
            and self.model.topk_indices_buffer is None
        ):
            raise RuntimeError(
                "HYV4 sparse MTP requires the target model's top-k indices buffer. "
                "The proposer must call HYV4MTP.set_topk_indices_buffer() before "
                "the first draft forward."
            )
        self.model.spec_step_idx = spec_step_idx
        return self.model(input_ids, positions, hidden_states, inputs_embeds)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        self.model.spec_step_idx = spec_step_idx
        return self.model.compute_logits(hidden_states)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput | None:
        return self.sampler(logits, sampling_metadata)

    def _rewrite_spec_layer_name(self, spec_layer: int, name: str) -> str:
        if f"model.layers.{spec_layer}.embed_tokens" in name:
            return "__skip__"
        if f"model.layers.{spec_layer}.shared_head" in name:
            return "__skip__"

        spec_layer_weight_names = ["enorm", "hnorm", "eh_proj", "final_layernorm"]
        spec_layer_weight = any(
            weight_name in name for weight_name in spec_layer_weight_names
        )
        if not spec_layer_weight:
            name = name.replace(
                f"model.layers.{spec_layer}.",
                f"model.layers.{spec_layer}.mtp_block.",
            )
        return name

    def _load_fused_expert_weights(
        self,
        name: str,
        params_dict: dict,
        loaded_weight: torch.Tensor,
        shard_id: str,
        num_experts: int,
    ) -> bool:
        if name not in params_dict:
            return False
        param = params_dict[name]
        weight_loader = typing.cast(Callable[..., bool], param.weight_loader)
        loaded_local_expert = False
        for expert_id in range(num_experts):
            curr_expert_weight = loaded_weight[expert_id]
            success = weight_loader(
                param,
                curr_expert_weight,
                name,
                shard_id,
                expert_id,
                return_success=True,
            )
            if success:
                loaded_local_expert = True
        return loaded_local_expert

    def _load_expert_weight(
        self,
        name: str,
        loaded_weight: torch.Tensor,
        params_dict: dict,
        loaded_params: set[str],
        split_expert_params_mapping: list[tuple[str, str, int, str]],
        fused_expert_param_names: dict[tuple[str, str], str],
        num_experts: int,
    ) -> bool:
        """Load one routed-expert weight in either checkpoint layout.

        Args:
            name: Weight name already rewritten to draft-module naming.
            loaded_weight: The checkpoint tensor.
            params_dict: The draft model's named parameters.
            loaded_params: Set updated with the parameters that received a value.
            split_expert_params_mapping: Mapping for the per-expert layout.
            fused_expert_param_names: ``(mlp_prefix, tag) -> param name`` for the
                all-experts-packed layout.
            num_experts: Total number of routed experts.

        Returns:
            True when the weight was consumed (even if this rank holds none of
            the addressed experts).
        """
        base = name.split(".experts.")[0]
        for ckpt_proj, tag in (
            (".experts.gate_up_proj", "w13_weight"),
            (".experts.down_proj", "w2_weight"),
        ):
            if ckpt_proj not in name:
                continue
            param_base = fused_expert_param_names.get((base, tag))
            if param_base is None:
                return False
            # Keep the checkpoint suffix (e.g. `_scale_inv`) so block-scale
            # tensors land in the scale parameter, not in the weight.
            target = _resolve_fused_expert_param(
                param_base, name.split(ckpt_proj, 1)[1], params_dict
            )
            if target is None:
                return False
            if tag == "w13_weight":
                chunks = loaded_weight.chunk(2, dim=-2)
                loaded_w1 = self._load_fused_expert_weights(
                    target, params_dict, chunks[0], "w1", num_experts
                )
                loaded_w3 = self._load_fused_expert_weights(
                    target, params_dict, chunks[1], "w3", num_experts
                )
                loaded = loaded_w1 and loaded_w3
            else:
                loaded = self._load_fused_expert_weights(
                    target, params_dict, loaded_weight, "w2", num_experts
                )
            if loaded:
                loaded_params.add(target)
            # The weight belongs to the experts either way; never fall through.
            return True

        consumed = False
        for param_name, weight_name, expert_id, shard_id in split_expert_params_mapping:
            if weight_name not in name:
                continue
            consumed = True
            name_mapped = name.replace(weight_name, param_name)
            if name_mapped not in params_dict:
                continue
            param = params_dict[name_mapped]
            weight_loader = typing.cast(Callable[..., bool], param.weight_loader)
            if weight_loader(
                param,
                loaded_weight,
                name_mapped,
                shard_id=shard_id,
                expert_id=expert_id,
                return_success=True,
            ):
                loaded_params.add(name_mapped)
        return consumed

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        pp_missing_layer_names = get_pp_missing_layer_names(self)
        loaded_params: set[str] = set()

        mtp_start = self.config.num_hidden_layers
        shared_weights = {
            "model.embed_tokens.weight": "model.embed_tokens.weight",
            "lm_head.weight": f"model.layers.{mtp_start}.shared_head.head.weight",
        }

        num_experts = getattr(self.config, "n_routed_experts", 0)
        sink_tp_size = get_tensor_model_parallel_world_size()
        sink_tp_rank = get_tensor_model_parallel_rank()
        n_local_head = self.config.num_attention_heads // sink_tp_size
        head_rank_start = n_local_head * sink_tp_rank
        head_rank_end = n_local_head * (sink_tp_rank + 1)

        # Routed-expert weights come in two checkpoint layouts:
        #   split: mlp.experts.<id>.gate_proj.weight
        #   fused: mlp.experts.gate_up_proj (all experts in one tensor)
        # The split layout is resolved through the shared EPLB helper so the
        # target param names stay in sync with the RoutedExperts module layout;
        # the fused layout is resolved from the live parameter names.
        split_expert_params_mapping = fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=num_experts,
        )
        fused_expert_param_names: dict[tuple[str, str], str] = {}
        for param_name in params_dict:
            for tag in ("w13_weight", "w2_weight"):
                if param_name.endswith(tag) and ".experts." in param_name:
                    base = param_name.split(".experts.")[0]
                    fused_expert_param_names[base, tag] = param_name
        stacked_mapping = [
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
            # MLA runs both latent down-projections as one GEMM.
            (".fused_qkv_a_proj", ".q_a_proj", 0),
            (".fused_qkv_a_proj", ".kv_a_proj_with_mqa", 1),
        ]
        # Sparse DSA draft blocks build an Indexer whose wk / weights_proj are
        # fused into a single MergedColumnParallelLinear (wk_weights_proj).
        indexer_stacked_mapping = [
            (".wk_weights_proj", ".wk", 0),
            (".wk_weights_proj", ".weights_proj", 1),
        ]
        # FP8 indexer wk dequant buffer (weight and scale arrive separately).
        pending_wk_fp8: dict[str, dict[str, torch.Tensor]] = {}

        for name, loaded_weight in weights:
            if name in shared_weights:
                target_name = shared_weights[name]
                if target_name in params_dict:
                    param = params_dict[target_name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(target_name)
                continue

            spec_layer = None
            if name.startswith("model.mtp_layers."):
                parts = name.split(".")
                if len(parts) > 3 and parts[2].isdigit():
                    spec_layer = mtp_start + int(parts[2])
                    name = name.replace(
                        f"model.mtp_layers.{parts[2]}.",
                        f"model.layers.{spec_layer}.",
                    )
            else:
                spec_layer = _get_spec_layer_idx_from_weight_name(self.config, name)

            if spec_layer is None:
                continue

            name = self._rewrite_spec_layer_name(spec_layer, name)
            if name == "__skip__":
                continue
            name, loaded_weight = _prepare_mtp_fp8_expert_scale(
                self.quant_config, name, loaded_weight
            )

            if "mlp.gate.e_score_correction_bias" in name:
                name = name.replace("gate.e_score_correction_bias", "expert_bias")

            is_loaded = False
            for param_name, weight_name, shard_id in stacked_mapping:
                if weight_name not in name or ".experts." in name:
                    continue
                name_mapped = name.replace(weight_name, param_name)
                if name_mapped not in params_dict:
                    if is_pp_missing_parameter(name_mapped, self):
                        is_loaded = True
                    break
                param = params_dict[name_mapped]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name_mapped)
                is_loaded = True
                break
            if is_loaded:
                continue

            # FP8 indexer wk: dequantize to BF16 and load into the fused
            # wk_weights_proj. PP-aware (skips layers not held by this rank).
            if _try_load_fp8_indexer_wk(
                name,
                loaded_weight,
                pending_wk_fp8,
                params_dict,
                loaded_params,
                pp_missing_layer_names,
            ):
                continue

            # BF16 indexer wk / weights_proj: merge into the fused param.
            is_loaded = False
            for param_name, weight_name, shard_id in indexer_stacked_mapping:
                if weight_name not in name or "wk_weights" in name:
                    continue
                name_mapped = name.replace(weight_name, param_name)
                if name_mapped not in params_dict:
                    if is_pp_missing_parameter(name_mapped, self):
                        is_loaded = True
                    break
                param = params_dict[name_mapped]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name_mapped)
                is_loaded = True
                break
            if is_loaded:
                continue

            is_loaded = False
            if ".experts." in name:
                is_loaded = self._load_expert_weight(
                    name,
                    loaded_weight,
                    params_dict,
                    loaded_params,
                    split_expert_params_mapping,
                    fused_expert_param_names,
                    num_experts,
                )
            if is_loaded:
                continue

            if "learnable_sink_param" in name:
                if name in params_dict:
                    narrow_weight = loaded_weight[head_rank_start:head_rank_end]
                    n = narrow_weight.shape[0]
                    with torch.no_grad():
                        params_dict[name][:n].copy_(narrow_weight)
                    loaded_params.add(name)
                continue

            remapped_name = maybe_remap_kv_scale_name(name, params_dict)
            if remapped_name is None:
                continue
            name = remapped_name
            if name not in params_dict:
                if is_pp_missing_parameter(name, self):
                    continue
                if _should_skip_missing_mtp_scale_param(self.quant_config, name):
                    continue
                logger.warning_once("Skipping unknown MTP weight: %s", name)
                continue

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)

        logger.info_once("HYV4 MTP draft model loaded: %d params", len(loaded_params))
        unassigned_all = sorted(set(params_dict) - loaded_params)
        # KVCacheScaleParameter (``k_scale`` / ``v_scale`` / ``q_scale`` /
        # ``prob_scale``) is created by ``BaseKVCacheMethod.create_weights`` for
        # every Attention layer that runs on an fp8-family quant method, and
        # its -1.0 sentinel is replaced by the runtime default (1.0) in
        # ``process_weights_after_loading`` -- which runs *after* this method.
        # HY V4 checkpoints intentionally omit these (``kv_cache_quant_algo:
        # null``); silence them only when the parameter is actually the
        # sentinel type so a genuine mismatch on a same-named tensor still
        # reaches the warning.
        from vllm.model_executor.layers.quantization.kv_cache import (
            KVCacheScaleParameter,
        )

        unassigned = [
            name
            for name in unassigned_all
            if not isinstance(params_dict.get(name), KVCacheScaleParameter)
        ]
        if unassigned:
            # A draft parameter with no checkpoint source keeps its sentinel
            # init value (FP8 scales start at finfo(float32).min), which
            # silently destroys draft quality instead of failing the load.
            logger.warning(
                "HYV4 MTP draft model: %d parameters received no checkpoint value: %s",
                len(unassigned),
                ", ".join(unassigned),
            )
        return loaded_params
