# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from copy import deepcopy
from typing import TYPE_CHECKING

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.models import ModelRegistry
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, cdiv
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

if TYPE_CHECKING:

    from vllm.config import VllmConfig

logger = init_logger(__name__)


class VerifyAndUpdateConfig:

    @staticmethod
    def verify_and_update_config(vllm_config: "VllmConfig") -> None:
        raise NotImplementedError


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
            "rope_scaling": getattr(config, "rope_scaling", None)
        }


class JambaForSequenceClassificationConfig(VerifyAndUpdateConfig):

    @staticmethod
    def verify_and_update_config(vllm_config: "VllmConfig") -> None:
        pooler_config = vllm_config.model_config.pooler_config
        if pooler_config.activation is None:
            pooler_config.activation = False


class JinaRobertaModelConfig(VerifyAndUpdateConfig):

    @staticmethod
    def verify_and_update_config(vllm_config: "VllmConfig") -> None:
        config = vllm_config.model_config.hf_config

        if config.position_embedding_type == "rotary":
            assert config.__class__.__name__ == "XLMRobertaFlashConfig"

            head_dim = config.hidden_size // config.num_attention_heads
            config.rotary_kwargs = {
                "head_size": head_dim,
                "rotary_dim": getattr(config, "rotary_emb_dim", head_dim),
                "max_position": config.max_position_embeddings,
                "base": getattr(config, "rope_theta", config.rotary_emb_base),
                "rope_scaling": getattr(config, "rope_scaling", None)
            }


class NomicBertModelConfig(VerifyAndUpdateConfig):

    @staticmethod
    def verify_and_update_config(vllm_config: "VllmConfig") -> None:
        config = vllm_config.model_config.hf_config

        assert config.__class__.__name__ == "NomicBertConfig"
        assert config.activation_function in ["swiglu", "gelu"]
        config.position_embedding_type = getattr(config,
                                                 "position_embedding_type",
                                                 "rope")

        if config.activation_function == "swiglu":
            config.hidden_act = "silu"
        else:
            config.hidden_act = config.activation_function

        assert (config.mlp_fc1_bias == config.mlp_fc2_bias ==
                config.qkv_proj_bias)
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
            "rope_scaling": getattr(config, "rope_scaling", None)
        }

        # we ignore config.rotary_scaling_factor so that for datasets shorter
        # than max_trained_positions 2048, the results are consistent
        # with SentenceTransformer.
        # The context extension uses vllm style rope_theta and rope_scaling.
        # See #17785 #18755
        if (not vllm_config.model_config.hf_overrides
                and vllm_config.model_config.original_max_model_len is None):
            # Default
            # Reset max_model_len to max_trained_positions.
            # nomic-embed-text-v2-moe the length is set to 512
            # by sentence_bert_config.json.
            max_model_len_before = vllm_config.model_config.max_model_len
            max_model_len = min(vllm_config.model_config.max_model_len,
                                max_trained_positions)

            vllm_config.recalculate_max_model_len(max_model_len)
            logger.warning(
                "Nomic context extension is disabled. "
                "Changing max_model_len from %s to %s. "
                "To enable context extension, see: "
                "https://github.com/vllm-project/vllm/tree/main/examples/offline_inference/context_extension.html",
                max_model_len_before, vllm_config.model_config.max_model_len)
        else:
            # We need to re-verify max_model_len to avoid lengths
            # greater than position_embedding.
            model_config = vllm_config.model_config
            hf_text_config = model_config.hf_text_config

            if isinstance(model_config.hf_overrides, dict):
                # hf_overrides_kw
                max_model_len = model_config.hf_overrides.get(
                    "max_model_len", vllm_config.model_config.max_model_len)
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

        is_original_qwen3_reranker = getattr(config,
                                             "is_original_qwen3_reranker",
                                             False)

        if not is_original_qwen3_reranker:
            return

        tokens = getattr(config, "classifier_from_token", None)
        assert tokens is not None and len(tokens) == 2, \
            ("Try loading the original Qwen3 Reranker?, see: "
             "https://github.com/vllm-project/vllm/tree/main/examples/offline_inference/qwen3_reranker.py")
        vllm_config.model_config.hf_config.method = "from_2_way_softmax"


class JinaVLForSequenceClassificationConfig(VerifyAndUpdateConfig):

    @staticmethod
    def verify_and_update_config(vllm_config: "VllmConfig") -> None:
        config = vllm_config.model_config.hf_config

        config.num_labels = 1


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
            "rope_scaling": getattr(config, "rope_scaling", None)
        }


class GraniteMoeHybridModelConfig(VerifyAndUpdateConfig):

    @staticmethod
    def verify_and_update_config(vllm_config: "VllmConfig") -> None:
        config = vllm_config.model_config
        config.max_seq_len_to_capture = config.max_model_len
        logger.info(
            "Setting max_seq_len_to_capture to %d "
            "to ensure that CUDA graph capture "
            "covers sequences of length up to max_model_len.",
            config.max_model_len)


class GptOssForCausalLMConfig(VerifyAndUpdateConfig):

    @staticmethod
    def verify_and_update_config(vllm_config: "VllmConfig") -> None:
        decoding_config = vllm_config.decoding_config
        if decoding_config.reasoning_backend == "":
            decoding_config.reasoning_backend = "GptOss"

        # Increase the max capture size from 512 to 1024 for performance.
        # NOTE(woosuk): This will increase the number of CUDA graphs
        # from 67 to 83.
        scheduler_config = vllm_config.scheduler_config
        if len(scheduler_config.cuda_graph_sizes) == 1:
            max_capture_size = scheduler_config.cuda_graph_sizes[0]
            # FIXME(woosuk): When using full cuda graph with FA3, the max
            # supported size is 992.
            if max_capture_size < 1024:
                cuda_graph_sizes = [1, 2, 4]
                # Step size 8 for small batch sizes
                cuda_graph_sizes += [i for i in range(8, 256, 8)]
                # Step size 16 for larger batch sizes
                cuda_graph_sizes += [i for i in range(256, 1025, 16)]
                scheduler_config.cuda_graph_sizes = cuda_graph_sizes
                logger.info(
                    "Overriding max cuda graph capture size to "
                    "%d for performance.", 1024)


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

        if not envs.VLLM_USE_V1:
            return

        cache_config = vllm_config.cache_config
        model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config

        if cache_config.cache_dtype == "auto":
            kv_cache_dtype = model_config.dtype
        else:
            kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        # get attention page size (for 1 token)
        attn_page_size_1_token = FullAttentionSpec(
            block_size=1,
            num_kv_heads=model_config.get_num_kv_heads(parallel_config),
            head_size=model_config.get_head_size(),
            dtype=kv_cache_dtype,
            use_mla=model_config.use_mla).page_size_bytes

        model_cls, _ = ModelRegistry.resolve_model_cls(
            model_config.architecture,
            model_config=model_config,
        )

        # get mamba page size
        mamba_page_size = MambaSpec(
            shapes=model_cls.get_mamba_state_shape_from_config(vllm_config),
            dtype=kv_cache_dtype,
            block_size=model_config.max_model_len,
        ).page_size_bytes

        # some attention backends (e.g. FA) only support setting
        # block size to multiple of 16, so let's suggest a value
        # that would work (note: FA is currently not compatible
        # with mamba layers, use FlashInfer instead).
        attn_block_size = 16 * cdiv(mamba_page_size,
                                    16 * attn_page_size_1_token)

        # override attention block size if either (a) the
        # user has not set it or (b) the user has set it
        # too small.
        if (cache_config.block_size is None
                or cache_config.block_size < attn_block_size):
            cache_config.block_size = attn_block_size
            logger.info(
                "Setting attention block size to %d tokens "
                "to ensure that attention page size is >= mamba page size.",
                attn_block_size)

        # compute new attention page size
        attn_page_size = \
            cache_config.block_size * attn_page_size_1_token

        assert attn_page_size >= mamba_page_size

        if attn_page_size == mamba_page_size:
            # don't need to pad mamba page size
            return

        # pad mamba page size to exactly match attention
        if (cache_config.mamba_page_size_padded is None
                or cache_config.mamba_page_size_padded != attn_page_size):
            cache_config.mamba_page_size_padded = (attn_page_size)
            mamba_padding_pct = 100 * (attn_page_size -
                                       mamba_page_size) / mamba_page_size
            logger.info(
                "Padding mamba page size by %.2f%% to ensure "
                "that mamba page size and attention page size are "
                "exactly equal.", mamba_padding_pct)


MODELS_CONFIG_MAP: dict[str, type[VerifyAndUpdateConfig]] = {
    "GteModel": SnowflakeGteNewModelConfig,
    "GteNewModel": GteNewModelConfig,
    "NomicBertModel": NomicBertModelConfig,
    "Qwen2ForProcessRewardModel": Qwen2ForProcessRewardModelConfig,
    "Qwen2ForRewardModel": Qwen2ForRewardModelConfig,
    "Qwen3ForSequenceClassification": Qwen3ForSequenceClassificationConfig,
    "XLMRobertaModel": JinaRobertaModelConfig,
    "JinaVLForRanking": JinaVLForSequenceClassificationConfig,
    "JambaForSequenceClassification": JambaForSequenceClassificationConfig,
    "GraniteMoeHybridForCausalLM": GraniteMoeHybridModelConfig,
    "GptOssForCausalLM": GptOssForCausalLMConfig,
}
