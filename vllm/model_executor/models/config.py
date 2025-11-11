# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from copy import deepcopy
from math import lcm
from typing import TYPE_CHECKING

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.models import ModelRegistry
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv, round_up
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec, MLAAttentionSpec

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


class VerifyAndUpdateConfig:
    @staticmethod
    def verify_and_update_config(vllm_config: "VllmConfig") -> None:
        raise NotImplementedError


class Gemma3TextModelConfig:
    @staticmethod
    def verify_and_update_config(vllm_config: "VllmConfig") -> None:
        hf_config = vllm_config.model_config.hf_config
        hf_config.is_causal = not hf_config.use_bidirectional_attention


class GteNewModelConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_config(vllm_config: "VllmConfig") -> None:
        config = vllm_config.model_config.hf_config

        assert config.__class__.__name__ == "NewConfig"
        assert config.hidden_act == "gelu"

        config.hidden_act = "geglu"

        head_dim = config.hidden_size // config.num_attention_heads
        config.rotary_kwargs = {
            "head_size": head_dim,
            "rotary_dim": getattr(config, "rotary_emb_dim", head_dim),
            "max_position": config.max_position_embeddings,
            "base": config.rope_theta,
            "rope_scaling": getattr(config, "rope_scaling", None),
        }


class JambaForSequenceClassificationConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_config(vllm_config: "VllmConfig") -> None:
        pooler_config = vllm_config.model_config.pooler_config
        if pooler_config.use_activation is None:
            pooler_config.use_activation = False


class JinaRobertaModelConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_config(vllm_config: "VllmConfig") -> None:
        model_config = vllm_config.model_config
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

            config.rotary_kwargs = {
                "head_size": head_dim,
                "rotary_dim": getattr(config, "rotary_emb_dim", head_dim),
                "max_position": max_position,
                "base": getattr(config, "rope_theta", config.rotary_emb_base),
                "rope_scaling": getattr(config, "rope_scaling", None),
            }


class NomicBertModelConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_config(vllm_config: "VllmConfig") -> None:
        config = vllm_config.model_config.hf_config

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

        head_dim = config.hidden_size // config.num_attention_heads
        rotary_emb_dim = int(head_dim * config.rotary_emb_fraction)
        max_trained_positions = getattr(config, "max_trained_positions", 2048)
        config.rotary_kwargs = {
            "head_size": head_dim,
            "rotary_dim": rotary_emb_dim,
            "max_position": max_trained_positions,
            "base": getattr(config, "rope_theta", config.rotary_emb_base),
            "rope_scaling": getattr(config, "rope_scaling", None),
        }

        # we ignore config.rotary_scaling_factor so that for datasets shorter
        # than max_trained_positions 2048, the results are consistent
        # with SentenceTransformer.
        # The context extension uses vllm style rope_theta and rope_scaling.
        # See #17785 #18755
        if (
            not vllm_config.model_config.hf_overrides
            and vllm_config.model_config.original_max_model_len is None
        ):
            # Default
            # Reset max_model_len to max_trained_positions.
            # nomic-embed-text-v2-moe the length is set to 512
            # by sentence_bert_config.json.
            max_model_len_before = vllm_config.model_config.max_model_len
            max_model_len = min(
                vllm_config.model_config.max_model_len, max_trained_positions
            )

            vllm_config.recalculate_max_model_len(max_model_len)
            logger.warning(
                "Nomic context extension is disabled. "
                "Changing max_model_len from %s to %s. "
                "To enable context extension, see: "
                "https://github.com/vllm-project/vllm/tree/main/examples/offline_inference/context_extension.html",
                max_model_len_before,
                vllm_config.model_config.max_model_len,
            )
        else:
            # We need to re-verify max_model_len to avoid lengths
            # greater than position_embedding.
            model_config = vllm_config.model_config
            hf_text_config = model_config.hf_text_config

            if isinstance(model_config.hf_overrides, dict):
                # hf_overrides_kw
                max_model_len = model_config.hf_overrides.get(
                    "max_model_len", vllm_config.model_config.max_model_len
                )
            else:
                # hf_overrides_fn
                # This might be overridden by sentence_bert_config.json.
                max_model_len = vllm_config.model_config.max_model_len

            # reset hf_text_config for recalculate_max_model_len.
            if hasattr(hf_text_config, "max_model_len"):
                delattr(hf_text_config, "max_model_len")
            hf_text_config.max_position_embeddings = max_trained_positions
            hf_text_config.rope_scaling = config.rotary_kwargs["rope_scaling"]

            # The priority of sentence_bert_config.json is higher
            # than max_position_embeddings
            encoder_config = deepcopy(model_config.encoder_config)
            encoder_config.pop("max_seq_length", None)
            model_config.encoder_config = encoder_config

            vllm_config.recalculate_max_model_len(max_model_len)


class Qwen2ForProcessRewardModelConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_config(vllm_config: "VllmConfig") -> None:
        pooler_config = vllm_config.model_config.pooler_config

        if pooler_config.step_tag_id is None:
            pooler_config.step_tag_id = 151651


class Qwen2ForRewardModelConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_config(vllm_config: "VllmConfig") -> None:
        pooler_config = vllm_config.model_config.pooler_config

        if pooler_config.softmax is None:
            pooler_config.softmax = False


class Qwen3ForSequenceClassificationConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_config(vllm_config: "VllmConfig") -> None:
        config = vllm_config.model_config.hf_config

        is_original_qwen3_reranker = getattr(
            config, "is_original_qwen3_reranker", False
        )

        if not is_original_qwen3_reranker:
            return

        tokens = getattr(config, "classifier_from_token", None)
        assert tokens is not None and len(tokens) == 2, (
            "Try loading the original Qwen3 Reranker?, see: "
            "https://github.com/vllm-project/vllm/tree/main/examples/offline_inference/qwen3_reranker.py"
        )
        vllm_config.model_config.hf_config.method = "from_2_way_softmax"


class JinaVLForSequenceClassificationConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_config(vllm_config: "VllmConfig") -> None:
        config = vllm_config.model_config.hf_config
        config.num_labels = 1
        pooler_config = vllm_config.model_config.pooler_config
        if pooler_config.logit_bias is None:
            pooler_config.logit_bias = 2.65


class SnowflakeGteNewModelConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_config(vllm_config: "VllmConfig") -> None:
        config = vllm_config.model_config.hf_config

        assert config.__class__.__name__ == "GteConfig"
        assert config.hidden_act == "gelu"

        config.hidden_act = "geglu"

        head_dim = config.hidden_size // config.num_attention_heads
        config.rotary_kwargs = {
            "head_size": head_dim,
            "rotary_dim": getattr(config, "rotary_emb_dim", head_dim),
            "max_position": config.max_position_embeddings,
            "base": config.rope_theta,
            "rope_scaling": getattr(config, "rope_scaling", None),
        }


class GptOssForCausalLMConfig(VerifyAndUpdateConfig):
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

        if cache_config.mamba_block_size is None:
            cache_config.mamba_block_size = model_config.max_model_len

        if cache_config.enable_prefix_caching:
            if model_config.supports_mamba_prefix_caching:
                logger.info(
                    "Warning: Prefix caching is currently enabled. "
                    "Its support for Mamba layers is experimental. "
                    "Please report any issues you may observe."
                )
            else:
                logger.info(
                    "Hybrid or mamba-based model detected without "
                    "support for prefix caching: disabling."
                )
                cache_config.enable_prefix_caching = False

        # TODO(tdoublep): remove once cascade attention is supported
        logger.info(
            "Disabling cascade attention since it is not supported for hybrid models."
        )
        model_config.disable_cascade_attn = True


class HybridAttentionMambaModelConfig(VerifyAndUpdateConfig):
    @classmethod
    def verify_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        """
        Ensure that page size of attention layers is greater than or
        equal to the mamba layers. If not, automatically set the attention
        block size to ensure that it is. If the attention page size is
        strictly greater than the mamba page size, we pad the mamba page size
        to make them equal.

        Args:
            vllm_config: vLLM Config
        """
        # Save the user input before it gets modified by MambaModelConfig
        mamba_block_size = vllm_config.cache_config.mamba_block_size
        # Enable FULL_AND_PIECEWISE by default
        MambaModelConfig.verify_and_update_config(vllm_config)

        cache_config = vllm_config.cache_config
        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config

        if cache_config.cache_dtype == "auto":
            kv_cache_dtype = model_config.dtype
        else:
            kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        # get attention page size (for 1 token)
        # Attention backend constraints:
        # - FlashAttention (FA) requires block size to be multiple of 16
        # - MLA (Multi-head Latent Attention) requires larger alignment:
        #   * CUTLASS_MLA backend: kernel_block_size 128 alignment
        #   * Other MLA backends: kernel_block_size 64 alignment
        if model_config.use_mla:
            use_cutlass_mla = envs.VLLM_ATTENTION_BACKEND == "CUTLASS_MLA"
            kernel_block_alignment_size = 128 if use_cutlass_mla else 64
            attn_page_size_1_token = MLAAttentionSpec(
                block_size=1,
                num_kv_heads=model_config.get_num_kv_heads(parallel_config),
                head_size=model_config.get_head_size(),
                dtype=kv_cache_dtype,
            ).page_size_bytes
        else:
            kernel_block_alignment_size = 16
            if (
                current_platform.is_device_capability(100)
                and model_config.get_head_size() == 256
                and (
                    envs.VLLM_ATTENTION_BACKEND is None
                    or envs.VLLM_ATTENTION_BACKEND == "FLASHINFER"
                )
            ):
                # https://github.com/flashinfer-ai/flashinfer/issues/1993 reports that`
                # head size 256 and block size 16 is not supported on blackwell.
                kernel_block_alignment_size = 32
            attn_page_size_1_token = FullAttentionSpec(
                block_size=1,
                num_kv_heads=model_config.get_num_kv_heads(parallel_config),
                head_size=model_config.get_head_size(),
                dtype=kv_cache_dtype,
            ).page_size_bytes

        model_cls, _ = ModelRegistry.resolve_model_cls(
            model_config.architecture,
            model_config=model_config,
        )

        # get mamba page size
        mamba_page_size = MambaSpec(
            shapes=model_cls.get_mamba_state_shape_from_config(vllm_config),
            dtypes=model_cls.get_mamba_state_dtype_from_config(vllm_config),
            block_size=model_config.max_model_len,
        ).page_size_bytes

        # Model may be marked as is_hybrid
        #  but mamba is skipped via config,
        #  return directly
        if mamba_page_size == 0:
            return

        if cache_config.enable_prefix_caching:
            # With prefix caching, select attention block size to
            # optimize for mamba kernel performance

            # Mamba2 SSD kernel uses a chunk_size, e.g. 256
            # Align the block to the kernel: use lowest multiple of chunk_size
            # of attention tokens that would fit mamba_page_size:
            # e.g. for mamba page size = 788kB
            #          attn_1_token = 2kB -> fits ~394 tokens
            #      then round up to a mulitple of 256 -> 512 tokens
            # End result:
            #  attn_block_size = 512
            #  mamba_block_size = 512 (aligned to a multiple of chunk_size)
            # TODO(tdoublep): this constraint can be relaxed fairly
            # easily by changing the way we layout chunks in the
            # mamba2 kernels.

            base_chunk_size = mamba_block_size or model_config.get_mamba_chunk_size()
            attn_tokens_per_mamba_state = cdiv(mamba_page_size, attn_page_size_1_token)
            chunk_size = lcm(base_chunk_size, kernel_block_alignment_size)
            attn_block_size = chunk_size * cdiv(attn_tokens_per_mamba_state, chunk_size)
            cache_config.mamba_block_size = attn_block_size
        else:
            # Without prefix caching, select minimum valid attention block size
            # to minimize mamba state padding

            # Calculate minimum attention block size that satisfies both:
            # 1. Backend alignment requirements (kernel_block_alignment_size)
            # 2. Mamba page size compatibility (attn_page_size >= mamba_page_size)
            attn_block_size = kernel_block_alignment_size * cdiv(
                mamba_page_size, kernel_block_alignment_size * attn_page_size_1_token
            )

        # override attention block size if either (a) the
        # user has not set it or (b) the user has set it
        # too small.
        if cache_config.block_size is None or cache_config.block_size < attn_block_size:
            cache_config.block_size = attn_block_size
            logger.info(
                "Setting attention block size to %d tokens "
                "to ensure that attention page size is >= mamba page size.",
                attn_block_size,
            )

        # compute new attention page size
        attn_page_size = cache_config.block_size * attn_page_size_1_token

        assert attn_page_size >= mamba_page_size

        if attn_page_size == mamba_page_size:
            # don't need to pad mamba page size
            return

        # pad mamba page size to exactly match attention
        if (
            cache_config.mamba_page_size_padded is None
            or cache_config.mamba_page_size_padded != attn_page_size
        ):
            cache_config.mamba_page_size_padded = attn_page_size
            mamba_padding_pct = (
                100 * (attn_page_size - mamba_page_size) / mamba_page_size
            )
            logger.info(
                "Padding mamba page size by %.2f%% to ensure "
                "that mamba page size and attention page size are "
                "exactly equal.",
                mamba_padding_pct,
            )


class DeepseekV32ForCausalLM(VerifyAndUpdateConfig):
    @classmethod
    def verify_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        """
        Updated fp8 cache to custom "fp8_ds_mla" format for DeepSeekV32
        """
        hf_config = vllm_config.model_config.hf_config

        # Mirror the check in vllm/model_executor/models/deepseek_v2.py
        is_v32 = hasattr(hf_config, "index_topk")
        assert is_v32

        # For DeepSeekV3.2, a custom fp8 format is used when fp8 kv-cache is enabled.
        cache_config = vllm_config.cache_config
        if cache_config.cache_dtype.startswith("fp8"):
            cache_config.cache_dtype = "fp8_ds_mla"
            logger.info("Using custom fp8 kv-cache format for DeepSeekV3.2")
        if cache_config.cache_dtype == "bfloat16":
            cache_config.cache_dtype = "auto"
            logger.info("Using bfloat16 kv-cache for DeepSeekV3.2")


MODELS_CONFIG_MAP: dict[str, type[VerifyAndUpdateConfig]] = {
    "GteModel": SnowflakeGteNewModelConfig,
    "GteNewModel": GteNewModelConfig,
    "GteNewForSequenceClassification": GteNewModelConfig,
    "Gemma3TextModel": Gemma3TextModelConfig,
    "NomicBertModel": NomicBertModelConfig,
    "Qwen2ForProcessRewardModel": Qwen2ForProcessRewardModelConfig,
    "Qwen2ForRewardModel": Qwen2ForRewardModelConfig,
    "Qwen3ForSequenceClassification": Qwen3ForSequenceClassificationConfig,
    "XLMRobertaModel": JinaRobertaModelConfig,
    "JinaVLForRanking": JinaVLForSequenceClassificationConfig,
    "JambaForSequenceClassification": JambaForSequenceClassificationConfig,
    "GptOssForCausalLM": GptOssForCausalLMConfig,
    "MambaForCausalLM": MambaModelConfig,
    "Mamba2ForCausalLM": MambaModelConfig,
    "FalconMambaForCausalLM": MambaModelConfig,
    "DeepseekV32ForCausalLM": DeepseekV32ForCausalLM,
}
