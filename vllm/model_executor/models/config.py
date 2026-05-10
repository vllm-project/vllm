# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from copy import deepcopy
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


class Gemma4Config(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_config(vllm_config: "VllmConfig") -> None:
        """Force unified attention backend for models with heterogeneous
        head dimensions.

        Some Gemma4 variants use different head dimensions for
        sliding window (head_dim) vs full attention (global_head_dim) layers.
        When global_head_dim > 256, FlashAttention rejects those layers
        (head_size <= 256 kernel limit), causing vLLM to select a different
        backend for each layer type. This mixed-backend execution produces
        numerical divergence and output corruption.

        The fix detects heterogeneous head dimensions from the model config
        and forces TRITON_ATTN (which has no head_size ceiling) for all
        layers when the user hasn't explicitly chosen a backend.

        TODO: Heterogeneous head_sizes (head_dim != global_head_dim)
        require NixlConnector changes to support per-layer KV transfer
        with different head dimensions for prefill-decode disaggregation.
        """
        hf_text_config = vllm_config.model_config.hf_text_config
        head_dim = getattr(hf_text_config, "head_dim", None)
        global_head_dim = getattr(hf_text_config, "global_head_dim", None)

        # Only force Triton when head dimensions actually differ AND the
        # larger one exceeds FlashAttention's kernel limit (head_size <= 256).
        # This avoids unnecessary backend forcing on smaller models where
        # the config carries global_head_dim but all layers can still use
        # the same FA backend.
        max_head_dim = max(head_dim or 0, global_head_dim or 0)
        if (
            head_dim is not None
            and global_head_dim is not None
            and head_dim != global_head_dim
            and max_head_dim > 256
            and vllm_config.attention_config.backend is None
        ):
            from vllm.v1.attention.backends.registry import (
                AttentionBackendEnum,
            )

            vllm_config.attention_config.backend = AttentionBackendEnum.TRITON_ATTN
            logger.info(
                "Gemma4 model has heterogeneous head dimensions "
                "(head_dim=%d, global_head_dim=%d). Forcing TRITON_ATTN "
                "backend to prevent mixed-backend numerical divergence.",
                head_dim,
                global_head_dim,
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
                if (
                    model_config.supports_mamba_prefix_caching
                    and vllm_config.speculative_config is not None
                ):
                    cache_config.mamba_cache_mode = "align"
                    logger.warning(
                        "Mamba cache mode is set to 'align' for %s by default "
                        "when prefix caching and speculative decoding are enabled",
                        model_config.architecture,
                    )
                else:
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
        max_trained_positions = getattr(config, "max_trained_positions", 2048)

        config.rotary_kwargs = {
            "head_size": head_dim,
            "max_position": max_trained_positions,
            "rope_parameters": config.rope_parameters,
        }

        # we ignore config.rotary_scaling_factor so that for datasets shorter
        # than max_trained_positions 2048, the results are consistent
        # with SentenceTransformer.
        # The context extension uses vllm style rope_theta and rope_parameters.
        # See #17785 #18755
        if (
            not model_config.hf_overrides
            and model_config.original_max_model_len is None
        ):
            # Default
            # Reset max_model_len to max_trained_positions.
            # nomic-embed-text-v2-moe the length is set to 512
            # by sentence_bert_config.json.
            max_model_len_before = model_config.max_model_len
            max_model_len = min(model_config.max_model_len, max_trained_positions)

            model_config.max_model_len = model_config.get_and_verify_max_len(
                max_model_len
            )

            if model_config.max_model_len != max_model_len_before:
                logger.warning(
                    "Nomic context extension is disabled. "
                    "Changing max_model_len from %s to %s. "
                    "To enable context extension, see: "
                    "https://github.com/vllm-project/vllm/tree/main/examples/features/context_extension/context_extension_offline.py",
                    max_model_len_before,
                    model_config.max_model_len,
                )
        else:
            # We need to re-verify max_model_len to avoid lengths
            # greater than position_embedding.
            hf_text_config = model_config.hf_text_config

            if isinstance(model_config.hf_overrides, dict):
                # hf_overrides_kw
                max_model_len = model_config.hf_overrides.get(
                    "max_model_len", model_config.max_model_len
                )
            else:
                # hf_overrides_fn
                # This might be overridden by sentence_bert_config.json.
                max_model_len = model_config.max_model_len

            # reset hf_text_config for recalculate_max_model_len.
            if hasattr(hf_text_config, "max_model_len"):
                delattr(hf_text_config, "max_model_len")
            hf_text_config.max_position_embeddings = max_trained_positions
            hf_text_config.rope_parameters = config.rotary_kwargs["rope_parameters"]

            # Update the cached derived_max_model_len to enforce the limit
            model_config.model_arch_config.derived_max_model_len_and_key = (
                float(max_trained_positions),
                "max_position_embeddings",
            )

            # The priority of sentence_bert_config.json is higher
            # than max_position_embeddings
            encoder_config = deepcopy(model_config.encoder_config)
            encoder_config.pop("max_seq_length", None)
            model_config.encoder_config = encoder_config

            model_config.max_model_len = model_config.get_and_verify_max_len(
                max_model_len
            )


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
    "ColQwen3_5": Qwen3_5ForConditionalGenerationConfig,
    "DeepseekV4ForCausalLM": DeepseekV4ForCausalLMConfig,
    "DeepseekV32ForCausalLM": DeepseekV32ForCausalLM,
    "Ernie4_5_VLMoeForConditionalGeneration": Ernie4_5_VLMoeForConditionalGenerationConfig,  # noqa: E501
    "FalconMambaForCausalLM": MambaModelConfig,
    "Gemma3TextModel": Gemma3TextModelConfig,
    "Gemma4ForCausalLM": Gemma4Config,
    "Gemma4ForConditionalGeneration": Gemma4Config,
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
    "VoyageQwen3BidirectionalEmbedModel": VoyageQwen3BidirectionalEmbedModelConfig,
    "XLMRobertaModel": JinaRobertaModelConfig,
}
