# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.utils.math_utils import round_up

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from vllm.config import CacheConfig, ModelConfig, VllmConfig


logger = init_logger(__name__)


class VerifyAndUpdateConfig:
    @staticmethod
    def verify_and_update_config(vllm_config: "VllmConfig") -> None:
        return

    @staticmethod
    def verify_and_update_model_config(model_config: "ModelConfig") -> None:
        return


class DeepseekV32ForCausalLM(VerifyAndUpdateConfig):
    @classmethod
    def verify_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        hf_config = vllm_config.model_config.hf_config

        # Mirror the check in vllm/model_executor/models/deepseek_v2.py
        is_v32 = hasattr(hf_config, "index_topk")
        assert is_v32

        cache_config = vllm_config.cache_config
        if cache_config.cache_dtype == "bfloat16":
            cache_config.cache_dtype = "auto"
            logger.info("Using bfloat16 kv-cache for DeepSeekV3.2")


class Ernie4_5_VLMoeForConditionalGenerationConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_config(vllm_config: "VllmConfig") -> None:
        # Ernie4.5-VL conditionally executes text/vision MoE branches, so
        # fast_moe_cold_start can silently produce incorrect execution order.
        vllm_config.compilation_config.fast_moe_cold_start = False


class Gemma3TextModelConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_model_config(model_config: "ModelConfig") -> None:
        hf_config = model_config.hf_config
        hf_config.is_causal = not hf_config.use_bidirectional_attention


class UnlimitedOCRForCausalLMConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_config(vllm_config: "VllmConfig") -> None:
        """Configure Unlimited-OCR attention backends for R-SWA and vision.

        Backend selection — controlled by the standard ``--attention-config``
        CLI argument (priority order):

          1. ``--attention-config '{"backend": "FLASH_ATTN"}'``
             → FA4 + rswa_mask_mod.  Exact token-level R-SWA.
               ``flash_attn_version`` is forced to 4 if not already set (R-SWA
               mask_mod requires FA4; FA3 cannot express it).  Raises if FA4 is
               not available on this device.

          2. ``--attention-config '{"backend": "FLEX_ATTENTION"}'``
             → FlexAttention R-SWA via Triton block mask.

          3. ``--attention-config '{"backend": "auto"}'`` (or omitted)
             → Auto-detect: FA4 if available (H20/H100 SM90), else FlexAttention.

        Regardless of backend, prefix caching is disabled for this model: R-SWA
        decode-phase KV is not a pure causal function of the prefix (so decode
        blocks are not reusable), and single-turn image-led OCR prompts rarely
        hit the prefix cache.

        Example — force FlexAttention even on a machine with FA4::

            vllm serve baidu/Unlimited-OCR \\
                --attention-config '{"backend": "FLEX_ATTENTION"}'
        """
        from vllm.v1.attention.backends.fa_utils import is_fa_version_supported
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        attn_config = vllm_config.attention_config
        fa4_available = is_fa_version_supported(4)

        # ── step 1: resolve backend ─────────────────────────────────────────
        # None means the user did not explicitly specify a backend; auto-select.
        if attn_config.backend is None:
            attn_config.backend = (
                AttentionBackendEnum.FLASH_ATTN
                if fa4_available
                else AttentionBackendEnum.FLEX_ATTENTION
            )
            logger.info(
                "Unlimited-OCR: auto-selected attention backend=%s (fa4_available=%s).",
                attn_config.backend.value,
                fa4_available,
            )

        # ── step 2: configure the chosen backend ────────────────────────────
        if attn_config.backend == AttentionBackendEnum.FLASH_ATTN:
            if not fa4_available:
                raise RuntimeError(
                    "Unlimited-OCR: --attention-config backend=FLASH_ATTN "
                    "requires FA4 (rswa_mask_mod), but FA4 is not available on "
                    "this device/installation.  Use backend=FLEX_ATTENTION or "
                    "upgrade vllm-flash-attn."
                )
            # On SM90 (H20), the default FA version is FA3 regardless of FA4
            # availability (FA4 is only auto-upgraded when head_size > 256).
            # The R-SWA mask_mod requires FA4, so force the version globally.
            if attn_config.flash_attn_version is None:
                attn_config.flash_attn_version = 4
            elif attn_config.flash_attn_version < 4:
                logger.warning(
                    "Unlimited-OCR: flash_attn_version=%d cannot express the "
                    "R-SWA mask_mod; upgrading to 4.",
                    attn_config.flash_attn_version,
                )
                attn_config.flash_attn_version = 4
            logger.info(
                "Unlimited-OCR: FlashAttention FA%d + rswa_mask_mod — exact R-SWA.",
                attn_config.flash_attn_version,
            )

        elif attn_config.backend == AttentionBackendEnum.FLEX_ATTENTION:
            logger.info(
                "Unlimited-OCR: FlexAttention — R-SWA via Triton block mask%s.",
                ""
                if not fa4_available
                else (
                    " (FA4 available but not used; pass backend=FLASH_ATTN to upgrade)"
                ),
            )

        else:
            raise ValueError(
                f"Unlimited-OCR: unsupported attention backend "
                f"{attn_config.backend!r} for R-SWA. "
                "Use FLASH_ATTN (FA4) or FLEX_ATTENTION."
            )

        # R-SWA windows the *generated* tokens, so a decode-token's KV is not a
        # pure causal function of the prefix and cannot be safely reused across
        # requests via prefix caching. Only the prompt/image prefix is cacheable,
        # but OCR is single-turn with image-led prompts that rarely share a
        # prefix, so prefix caching brings little benefit while complicating the
        # KV cache manager. Disable it for this model.
        cache_config = vllm_config.cache_config
        if cache_config.enable_prefix_caching:
            cache_config.enable_prefix_caching = False
            logger.info(
                "Unlimited-OCR: disabling prefix caching (R-SWA decode KV is not "
                "cacheable, and single-turn image-led prompts rarely hit the "
                "prefix cache)."
            )

        mm_config = getattr(vllm_config.model_config, "multimodal_config", None)
        if mm_config is not None:
            if mm_config.mm_encoder_attn_backend is None:
                mm_config.mm_encoder_attn_backend = AttentionBackendEnum.FLASH_ATTN
            elif mm_config.mm_encoder_attn_backend == AttentionBackendEnum.FLASHINFER:
                logger.warning(
                    "Unlimited-OCR: FlashInfer is not supported for the vision "
                    "encoder (the CLIP stage runs full attention without "
                    "cu_seqlens); falling back to FlashAttention."
                )
                mm_config.mm_encoder_attn_backend = AttentionBackendEnum.FLASH_ATTN

    @staticmethod
    def verify_and_update_model_config(model_config: "ModelConfig") -> None:
        text_config = model_config.hf_config.text_config
        text_config.architectures = ["DeepseekV2ForCausalLM"]
        if getattr(model_config.hf_config, "rswa_window", None) is None:
            model_config.hf_config.rswa_window = 128
        # Propagate rswa_window to text_config so that DeepseekAttention (which
        # receives text_config as its vllm_config.model_config.hf_config via
        # init_vllm_registered_model) can read it and create RSWAAttention.
        rswa_window = model_config.hf_config.rswa_window
        text_config.rswa_window = rswa_window


class Gemma4Config(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_config(vllm_config: "VllmConfig") -> None:
        """Configure attention for heterogeneous head dimensions.

        Gemma4 uses different head dimensions for sliding window
        (head_dim) vs full attention (global_head_dim) layers. The
        default FA3 on Hopper cannot handle head_dim > 256, which
        causes mixed backend selection and numerical divergence.

        When FA4 is available we force it for ALL layers, giving a
        uniform kernel path and avoiding the mixed FA3+FA4 penalty.
        When FA4 is not available we fall back to Triton.
        """
        hf_text_config = vllm_config.model_config.hf_text_config
        head_dim = getattr(hf_text_config, "head_dim", None)
        global_head_dim = getattr(hf_text_config, "global_head_dim", None)

        if head_dim is None or global_head_dim is None or head_dim == global_head_dim:
            return

        from vllm.v1.attention.backends.fa_utils import is_fa_version_supported
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        max_head_dim = max(head_dim, global_head_dim)

        if is_fa_version_supported(4) and max_head_dim <= 512:
            if (
                vllm_config.attention_config.flash_attn_version is None
                and vllm_config.attention_config.backend
                in (None, AttentionBackendEnum.FLASH_ATTN)
            ):
                vllm_config.attention_config.flash_attn_version = 4
                logger.info(
                    "Gemma4 model has heterogeneous head dimensions "
                    "(head_dim=%d, global_head_dim=%d). Using FA4 for "
                    "all layers to avoid mixed FA3/FA4 penalty.",
                    head_dim,
                    global_head_dim,
                )
        elif vllm_config.attention_config.backend is None:
            vllm_config.attention_config.backend = AttentionBackendEnum.TRITON_ATTN
            logger.info(
                "Gemma4 model has heterogeneous head dimensions "
                "(head_dim=%d, global_head_dim=%d). FA4 not available, "
                "forcing TRITON_ATTN backend.",
                head_dim,
                global_head_dim,
            )


class DiffusionGemmaModelForBlockDiffusionConfig(VerifyAndUpdateConfig):
    @classmethod
    def verify_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        """Set up the diffusion config and defaults for DiffusionGemma.

        Auto-creates DiffusionConfig from the HF config when the user
        didn't pass ``--diffusion-config``. Diffusion sampling params are
        read straight from generation_config.json at sampler-build time
        (see DiffusionGemma's custom_sampler), not injected here.
        """
        # Inherit Gemma4's attention backend selection (FA4 on Hopper,
        # TRITON_ATTN fallback for heterogeneous head dims).
        Gemma4Config.verify_and_update_config(vllm_config)

        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        attention_config = vllm_config.attention_config
        if attention_config.backend == AttentionBackendEnum.FLASHINFER:
            raise ValueError(
                "FlashInfer does not support DiffusionGemma's mixed "
                "causal/bidirectional attention. Use --attention-backend "
                "FLASH_ATTN or TRITON_ATTN instead."
            )
        if attention_config.backend is None and not attention_config.use_non_causal:
            attention_config.use_non_causal = True
            logger.info(
                "DiffusionGemma uses mixed causal/bidirectional attention "
                "within a batch; setting use_non_causal=True to exclude "
                "FlashInfer from auto-selection."
            )

        # Auto-create DiffusionConfig from HF config if not provided.
        if vllm_config.diffusion_config is None:
            from vllm.config.diffusion import DiffusionConfig

            hf_config = vllm_config.model_config.hf_config
            canvas_length = getattr(hf_config, "canvas_length", 256)
            vllm_config.diffusion_config = DiffusionConfig(
                canvas_length=canvas_length,
            )

        # The diffusion sampler materializes [num_seqs, canvas_length, vocab]
        # fp32 transients, so concurrency is memory-bound (>8 OOMs a single H200).
        # Default to 8 when the user didn't pass --max-num-seqs.
        # We can't see the original None here (the engine already filled a generic
        # default), so use >= DEFAULT_MAX_NUM_SEQS as a proxy, (the default is much
        # larger than any deliberate value for this model)
        from vllm.config.scheduler import SchedulerConfig

        sc = vllm_config.scheduler_config
        if sc is not None and sc.max_num_seqs >= SchedulerConfig.DEFAULT_MAX_NUM_SEQS:
            sc.max_num_seqs = 8

        # Remove the model's generation_config.json cap on max_new_tokens
        # (256) so DiffusionGemma behaves like every other model: no
        # server-wide limit, each request controls its own output length
        # via max_tokens.  Setting to None causes get_diff_sampling_param
        # to skip this key entirely.
        model_config = vllm_config.model_config
        if "max_new_tokens" not in model_config.override_generation_config:
            model_config.override_generation_config["max_new_tokens"] = None
            logger.info(
                "DiffusionGemma: removing server-wide max_new_tokens cap "
                "from generation_config.json (use "
                "--override-generation-config to set a custom limit).",
            )


class DeepseekV4ForCausalLMConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_model_config(model_config: "ModelConfig") -> None:
        quant_config = getattr(model_config.hf_config, "quantization_config", None)
        if quant_config is not None and quant_config.get("quant_method") == "fp8":
            model_type = getattr(model_config.hf_config, "model_type", None)
            if model_type == "deepseek_v4":
                model_config.hf_config.quantization_config["quant_method"] = (
                    "deepseek_v4_fp8"
                )

        hf_text_quant_config = getattr(
            model_config.hf_text_config, "quantization_config", None
        )
        if (
            hf_text_quant_config is not None
            and hf_text_quant_config.get("quant_method") == "fp8"
        ):
            model_type = getattr(model_config.hf_text_config, "model_type", None)
            if model_type == "deepseek_v4":
                model_config.hf_text_config.quantization_config["quant_method"] = (
                    "deepseek_v4_fp8"
                )


class GptOssForCausalLMConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_model_config(model_config: "ModelConfig") -> None:
        quant_config = getattr(model_config.hf_config, "quantization_config", None)
        if quant_config is not None and quant_config.get("quant_method") == "mxfp4":
            model_config.hf_config.quantization_config["quant_method"] = "gpt_oss_mxfp4"

        hf_text_quant_config = getattr(
            model_config.hf_text_config, "quantization_config", None
        )
        if (
            hf_text_quant_config is not None
            and hf_text_quant_config.get("quant_method") == "mxfp4"
        ):
            model_config.hf_text_config.quantization_config["quant_method"] = (
                "gpt_oss_mxfp4"
            )

    @staticmethod
    def verify_and_update_config(vllm_config: "VllmConfig") -> None:
        structured_outputs_config = vllm_config.structured_outputs_config
        if structured_outputs_config.reasoning_parser == "":
            structured_outputs_config.reasoning_parser = "openai_gptoss"

        # Increase the max capture size from 512 to 1024 for performance.
        # NOTE(woosuk): This will increase the number of CUDA graphs
        # from 67 to 83.
        compilation_config = vllm_config.compilation_config
        # Only override when the user has not set either of
        # cudagraph_capture_sizes or max_cudagraph_capture_size.
        if (
            compilation_config.cudagraph_capture_sizes is None
            and compilation_config.max_cudagraph_capture_size is None
        ):
            compilation_config.max_cudagraph_capture_size = 1024
            logger.info(
                "Overriding max cuda graph capture size to %d for performance.", 1024
            )


class GteNewModelConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_model_config(model_config: "ModelConfig") -> None:
        config = model_config.hf_config

        assert config.__class__.__name__ == "NewConfig"
        assert config.hidden_act == "gelu"

        config.hidden_act = "geglu"

        head_dim = config.hidden_size // config.num_attention_heads
        rotary_dim = getattr(config, "rotary_emb_dim", head_dim)
        config.rope_parameters["partial_rotary_factor"] = rotary_dim / head_dim
        config.rotary_kwargs = {
            "head_size": head_dim,
            "max_position": config.max_position_embeddings,
            "rope_parameters": config.rope_parameters,
        }


class HybridAttentionMambaModelConfig(VerifyAndUpdateConfig):
    @classmethod
    def verify_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        """
        Perform early validation and setup for hybrid attention/mamba models.

        Block size alignment with mamba page sizes is handled later by
        Platform.update_block_size_for_backend(), which runs after model
        layers are constructed and the attention backend is known.

        Args:
            vllm_config: vLLM Config
        """
        cache_config = vllm_config.cache_config

        # Disable calculate_kv_scales for hybrid models: uninitialized
        # recurrent state corrupts scales during the calibration pass.
        # See issue: https://github.com/vllm-project/vllm/issues/37554

        if cache_config.calculate_kv_scales:
            logger.warning(
                "Disabling calculate_kv_scales for hybrid model '%s'. "
                "Hybrid models with recurrent layers (GDN, Mamba, SSM) "
                "produce unreliable KV cache scales during the "
                "calibration pass because recurrent state is "
                "uninitialized. Using default scale of 1.0 instead.",
                vllm_config.model_config.model,
            )
            cache_config.calculate_kv_scales = False

        # Enable FULL_AND_PIECEWISE by default
        MambaModelConfig.verify_and_update_config(vllm_config)


class JambaForSequenceClassificationConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_model_config(model_config: "ModelConfig") -> None:
        pooler_config = model_config.pooler_config
        if pooler_config.use_activation is None:
            pooler_config.use_activation = False


class JinaForRankingConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_model_config(model_config: "ModelConfig") -> None:
        model_config.hf_config.embedding_size = 512


class JinaRobertaModelConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_model_config(model_config: "ModelConfig") -> None:
        config = model_config.hf_config

        if config.position_embedding_type == "rotary":
            assert config.__class__.__name__ == "XLMRobertaFlashConfig"

            head_dim = config.hidden_size // config.num_attention_heads
            max_position = config.max_position_embeddings
            # Jina-embeddings-v3 has max_position_embeddings=8194, which will cause
            # out-of-bound index issue at RoPE for long prompts with torch.compile,
            # because it can't be divided by triton num_warps(default=4 or 8).
            # To deal with this, we increase max_position to multiple of n_warps,
            # so that triton kernel won't hit out-of-bound index in RoPE cache.
            if not model_config.enforce_eager:
                max_position = round_up(max_position, 8)

            rotary_dim = getattr(config, "rotary_emb_dim", head_dim)
            config.rope_parameters["partial_rotary_factor"] = rotary_dim / head_dim

            config.rotary_kwargs = {
                "head_size": head_dim,
                "max_position": max_position,
                "rope_parameters": config.rope_parameters,
            }


class JinaVLForSequenceClassificationConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_model_config(model_config: "ModelConfig") -> None:
        config = model_config.hf_config
        config.num_labels = 1
        pooler_config = model_config.pooler_config
        if pooler_config.logit_mean is None:
            pooler_config.logit_mean = 2.65


class LlamaBidirectionalConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_model_config(model_config: "ModelConfig") -> None:
        from vllm.config.pooler import SequencePoolingType

        hf_config = model_config.hf_config
        hf_config.is_causal = False

        pooling_type_map: dict[str, SequencePoolingType] = {
            "avg": "MEAN",
            "cls": "CLS",
            "last": "LAST",
        }

        pooling_type = pooling_type_map.get(hf_config.pooling, None)
        if pooling_type is None:
            raise ValueError(f"pool_type {hf_config.pooling!r} not supported")

        model_config.pooler_config.seq_pooling_type = pooling_type


class LlamaNemotronVLConfig(VerifyAndUpdateConfig):
    """Config handler for LlamaNemotronVL embedding models."""

    @staticmethod
    def verify_and_update_model_config(model_config: "ModelConfig") -> None:
        from vllm.config.pooler import SequencePoolingType

        hf_config = model_config.hf_config

        # Set bidirectional attention on the language model config
        hf_config.is_causal = False
        if hasattr(hf_config, "llm_config"):
            hf_config.llm_config.is_causal = False

        if hasattr(hf_config, "vision_config"):
            hf_config.patch_size = hf_config.vision_config.patch_size

        # Set up pooling type
        pooling_type_map: dict[str, SequencePoolingType] = {
            "avg": "MEAN",
            "cls": "CLS",
            "last": "LAST",
        }

        # Get pooling type from config (check both top-level and llm_config)
        pooling = getattr(hf_config, "pooling", None)
        if pooling is None and hasattr(hf_config, "llm_config"):
            pooling = getattr(hf_config.llm_config, "pooling", "avg")

        pooling_type = pooling_type_map.get(pooling)
        if pooling_type is None:
            raise ValueError(f"pool_type {pooling!r} not supported")

        model_config.pooler_config.seq_pooling_type = pooling_type


class MambaModelConfig(VerifyAndUpdateConfig):
    @classmethod
    def verify_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        """
        Enable FULL_AND_PIECEWISE cuda graph mode by default (required
        to get good performance for mamba layers in V1).

        Args:
            vllm_config: vLLM Config
        """
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config

        if cache_config.enable_prefix_caching:
            if cache_config.mamba_cache_mode == "none":
                cache_config.mamba_cache_mode = (
                    "all" if model_config.supports_mamba_prefix_caching else "align"
                )
                logger.warning(
                    "Mamba cache mode is set to '%s' for %s by default "
                    "when prefix caching is enabled",
                    cache_config.mamba_cache_mode,
                    model_config.architecture,
                )
            if (
                cache_config.mamba_cache_mode == "all"
                and not model_config.supports_mamba_prefix_caching
            ):
                cache_config.mamba_cache_mode = "align"
                logger.warning(
                    "Hybrid or mamba-based model detected without support "
                    "for prefix caching with Mamba cache 'all' mode: "
                    "falling back to 'align' mode."
                )
            if cache_config.mamba_cache_mode == "align":
                assert vllm_config.scheduler_config.enable_chunked_prefill, (
                    "Chunked prefill is required for mamba cache mode 'align'."
                )
            logger.info(
                "Warning: Prefix caching in Mamba cache '%s' "
                "mode is currently enabled. "
                "Its support for Mamba layers is experimental. "
                "Please report any issues you may observe.",
                cache_config.mamba_cache_mode,
            )
            # By default, mamba block size will be set to max_model_len (see
            # below). When enabling prefix caching, we align mamba block size
            # to the block size as the basic granularity for prefix caching.
            if cache_config.mamba_block_size is None:
                cache_config.mamba_block_size = cache_config.block_size
        else:
            if cache_config.mamba_cache_mode != "none":
                cache_config.mamba_cache_mode = "none"
                logger.warning(
                    "Mamba cache mode is set to 'none' when prefix caching is disabled"
                )
            if cache_config.mamba_block_size is None:
                cache_config.mamba_block_size = model_config.max_model_len


class NemotronHForCausalLMConfig(VerifyAndUpdateConfig):
    DEFAULT_MAMBA_SSM_CACHE_DTYPE = "float32"
    """Only `float32` is known to have no accuracy issues by default."""

    @classmethod
    def update_mamba_ssm_cache_dtype(
        cls, *, cache_config: "CacheConfig", hf_config: "PretrainedConfig"
    ) -> None:
        """Update mamba_ssm_cache_dtype for NemotronH models when set to 'auto'
        (or not explicitly set), to the value specified in the HF config, or to
        `float32` if not specified.
        """
        if cache_config.mamba_ssm_cache_dtype == "auto":
            mamba_ssm_cache_dtype = getattr(
                hf_config, "mamba_ssm_cache_dtype", cls.DEFAULT_MAMBA_SSM_CACHE_DTYPE
            )
            logger.info(
                "Updating mamba_ssm_cache_dtype to '%s' for NemotronH model",
                mamba_ssm_cache_dtype,
            )
            cache_config.mamba_ssm_cache_dtype = mamba_ssm_cache_dtype

    @classmethod
    def verify_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        cls.update_mamba_ssm_cache_dtype(
            cache_config=vllm_config.cache_config,
            hf_config=vllm_config.model_config.hf_config,
        )


class NemotronHNanoVLV2Config(VerifyAndUpdateConfig):
    @classmethod
    def verify_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        NemotronHForCausalLMConfig.update_mamba_ssm_cache_dtype(
            cache_config=vllm_config.cache_config,
            hf_config=vllm_config.model_config.hf_config.text_config,
        )

    @staticmethod
    def verify_and_update_model_config(model_config: "ModelConfig") -> None:
        mm_config = model_config.multimodal_config
        if mm_config is not None:
            video_kwargs = mm_config.media_io_kwargs.setdefault("video", {})
            video_kwargs.setdefault("video_backend", "nemotron_vl")


class NomicBertModelConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_model_config(model_config: "ModelConfig") -> None:
        config = model_config.hf_config

        assert config.__class__.__name__ == "NomicBertConfig"
        assert config.activation_function in ["swiglu", "gelu"]
        config.position_embedding_type = getattr(
            config, "position_embedding_type", "rope"
        )

        if config.activation_function == "swiglu":
            config.hidden_act = "silu"
        else:
            config.hidden_act = config.activation_function

        assert config.mlp_fc1_bias == config.mlp_fc2_bias == config.qkv_proj_bias
        config.bias = config.qkv_proj_bias

        assert config.rotary_emb_scale_base is None
        assert not config.rotary_emb_interleaved

        config.layer_norm_eps = config.layer_norm_epsilon
        config.intermediate_size = config.n_inner
        config.hidden_size = config.n_embd
        config.num_hidden_layers = config.n_layer
        model_config.model_arch_config.hidden_size = config.hidden_size
        model_config.model_arch_config.total_num_hidden_layers = (
            config.num_hidden_layers
        )

        head_dim = config.hidden_size // config.num_attention_heads
        max_position_embeddings = getattr(config, "max_position_embeddings", 2048)
        max_trained_positions = getattr(
            config, "max_trained_positions", max_position_embeddings
        )

        rope_parameters = {
            "max_trained_positions": max_trained_positions,
            **(config.rope_parameters or {}),
        }

        config.rotary_kwargs = {
            "head_size": head_dim,
            "max_position": model_config.max_model_len,
            "rope_parameters": rope_parameters,
        }


class Qwen2ForProcessRewardModelConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_model_config(model_config: "ModelConfig") -> None:
        pooler_config = model_config.pooler_config

        if pooler_config.step_tag_id is None:
            pooler_config.step_tag_id = 151651


class Qwen2ForRewardModelConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_model_config(model_config: "ModelConfig") -> None:
        pooler_config = model_config.pooler_config

        if pooler_config.use_activation is None:
            pooler_config.use_activation = False


class Qwen3ForSequenceClassificationConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_model_config(model_config: "ModelConfig") -> None:
        config = model_config.hf_config

        is_original_qwen3_reranker = getattr(
            config, "is_original_qwen3_reranker", False
        )

        if not is_original_qwen3_reranker:
            return

        tokens = getattr(config, "classifier_from_token", None)
        assert tokens is not None and len(tokens) == 2, (
            "Try loading the original Qwen3 Reranker?, see: "
            "https://github.com/vllm-project/vllm/tree/main/examples/pooling/score/qwen3_reranker_offline.py"
        )
        text_config = config.get_text_config()
        text_config.method = "from_2_way_softmax"
        text_config.classifier_from_token = tokens


class Qwen3VLForSequenceClassificationConfig(Qwen3ForSequenceClassificationConfig):
    pass


class Qwen3_5ForConditionalGenerationConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_config(vllm_config: "VllmConfig") -> None:
        """Update mamba_ssm_cache_dtype for Qwen3.5 models when set to 'auto'
        (or not explicitly set), to the value specified in the HF config's
        mamba_ssm_dtype field. Warn if the user explicitly overrides it to a
        different value.
        """
        cache_config = vllm_config.cache_config
        hf_text_config = vllm_config.model_config.hf_text_config
        mamba_ssm_dtype = getattr(hf_text_config, "mamba_ssm_dtype", None)
        if cache_config.mamba_ssm_cache_dtype == "auto":
            if mamba_ssm_dtype is not None:
                cache_config.mamba_ssm_cache_dtype = mamba_ssm_dtype
        elif (
            mamba_ssm_dtype is not None
            and cache_config.mamba_ssm_cache_dtype != mamba_ssm_dtype
        ):
            logger.warning(
                "Qwen3.5 model specifies mamba_ssm_dtype='%s' in its config, "
                "but --mamba-ssm-cache-dtype='%s' was passed. "
                "Using the user-specified value.",
                mamba_ssm_dtype,
                cache_config.mamba_ssm_cache_dtype,
            )


class ColQwen3_5Config(Qwen3_5ForConditionalGenerationConfig):
    """ColQwen3.5 (late-interaction retrieval) inherits Qwen3.5's mamba cache
    handling and additionally serves BIDIRECTIONAL attention: ColPali-style
    document/query encoding attends over the whole sequence, not causally. Set
    is_causal=False so Qwen3NextAttention builds its full_attention layers with
    AttentionType.ENCODER_ONLY (the linear_attention GatedDeltaNet layers are
    unaffected). Generation arches keep the parent (causal) and are untouched.
    """

    @staticmethod
    def verify_and_update_model_config(model_config: "ModelConfig") -> None:
        model_config.hf_config.is_causal = False


class SnowflakeGteNewModelConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_model_config(model_config: "ModelConfig") -> None:
        config = model_config.hf_config

        assert config.__class__.__name__ == "GteConfig"
        assert config.hidden_act == "gelu"

        config.hidden_act = "geglu"

        head_dim = config.hidden_size // config.num_attention_heads
        rotary_dim = getattr(config, "rotary_emb_dim", head_dim)
        config.rope_parameters["partial_rotary_factor"] = rotary_dim / head_dim
        config.rotary_kwargs = {
            "head_size": head_dim,
            "max_position": config.max_position_embeddings,
            "rope_parameters": config.rope_parameters,
        }


class VoyageQwen3BidirectionalEmbedModelConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_model_config(model_config: "ModelConfig") -> None:
        model_config.hf_config.is_causal = False
        model_config.hf_config.embedding_size = model_config.hf_config.num_labels


MODELS_CONFIG_MAP: dict[str, type[VerifyAndUpdateConfig]] = {
    "ColBERTJinaRobertaModel": JinaRobertaModelConfig,
    "ColQwen3_5": ColQwen3_5Config,
    "DeepseekV4ForCausalLM": DeepseekV4ForCausalLMConfig,
    "DeepseekV32ForCausalLM": DeepseekV32ForCausalLM,
    "DiffusionGemmaForBlockDiffusion": DiffusionGemmaModelForBlockDiffusionConfig,  # noqa: E501
    "Ernie4_5_VLMoeForConditionalGeneration": Ernie4_5_VLMoeForConditionalGenerationConfig,  # noqa: E501
    "FalconMambaForCausalLM": MambaModelConfig,
    "Gemma3TextModel": Gemma3TextModelConfig,
    "Gemma4ForCausalLM": Gemma4Config,
    "Gemma4ForConditionalGeneration": Gemma4Config,
    "Gemma4UnifiedForConditionalGeneration": Gemma4Config,
    "GptOssForCausalLM": GptOssForCausalLMConfig,
    "GteModel": SnowflakeGteNewModelConfig,
    "GteNewForSequenceClassification": GteNewModelConfig,
    "GteNewModel": GteNewModelConfig,
    "JambaForSequenceClassification": JambaForSequenceClassificationConfig,
    "JinaForRanking": JinaForRankingConfig,
    "JinaVLForRanking": JinaVLForSequenceClassificationConfig,
    "LlamaBidirectionalForSequenceClassification": LlamaBidirectionalConfig,
    "LlamaBidirectionalModel": LlamaBidirectionalConfig,
    "LlamaNemotronVLForSequenceClassification": LlamaNemotronVLConfig,
    "LlamaNemotronVLModel": LlamaNemotronVLConfig,
    "Mamba2ForCausalLM": MambaModelConfig,
    "MambaForCausalLM": MambaModelConfig,
    "NemotronHForCausalLM": NemotronHForCausalLMConfig,
    "NemotronHPuzzleForCausalLM": NemotronHForCausalLMConfig,
    "NemotronH_Nano_VL_V2": NemotronHNanoVLV2Config,
    "NomicBertModel": NomicBertModelConfig,
    "Qwen2ForProcessRewardModel": Qwen2ForProcessRewardModelConfig,
    "Qwen2ForRewardModel": Qwen2ForRewardModelConfig,
    "Qwen3ForSequenceClassification": Qwen3ForSequenceClassificationConfig,
    "Qwen3VLForSequenceClassification": Qwen3VLForSequenceClassificationConfig,
    "Qwen3_5ForConditionalGeneration": Qwen3_5ForConditionalGenerationConfig,
    "Qwen3_5MoeForConditionalGeneration": Qwen3_5ForConditionalGenerationConfig,
    "UnlimitedOCRForCausalLM": UnlimitedOCRForCausalLMConfig,
    "VoyageQwen3BidirectionalEmbedModel": VoyageQwen3BidirectionalEmbedModelConfig,
    "XLMRobertaModel": JinaRobertaModelConfig,
}
