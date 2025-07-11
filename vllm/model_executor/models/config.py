# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING

import vllm.envs as envs
from vllm.distributed import divide
from vllm.logger import init_logger
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, cdiv
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

if TYPE_CHECKING:
    from transformers.configuration_utils import PretrainedConfig

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
        rotary_emb_dim = head_dim * config.rotary_emb_fraction
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


class HybridAttentionMambaModelConfig(VerifyAndUpdateConfig):

    @classmethod
    def extra_groups_for_head_shards(cls, ngroups: int, tp_size: int) -> int:
        """Compute the increase in group numbers to account for
        replication in order to accompany the head shards."""

        # in the case ngoups % tp_size == 0, this will be zero
        if ngroups % tp_size == 0:
            return 0

        # for n_groups == 1, this is exactly tp_size - n_groups
        return tp_size - ngroups

    @dataclass
    class MambaConfig:
        expand: int
        n_groups: int
        n_heads: int
        d_head: int
        d_state: int
        d_conv: int

    @classmethod
    def parse_mamba_config(cls, config: "PretrainedConfig") -> MambaConfig:
        return cls.MambaConfig(
            expand=config.mamba_expand,
            n_groups=config.mamba_n_groups,
            n_heads=config.mamba_n_heads,
            d_head=config.mamba_d_head,
            d_state=config.mamba_d_state,
            d_conv=config.mamba_d_conv,
        )

    @classmethod
    def get_mamba_cache_shape(
            cls, vllm_config: "VllmConfig"
    ) -> tuple[tuple[int, int], tuple[int, int]]:

        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_config
        mamba_config = cls.parse_mamba_config(hf_config)

        world_size = parallel_config.tensor_parallel_size
        hidden_size = hf_config.hidden_size
        intermediate_size = mamba_config.expand * hidden_size

        # if n_groups is not divisible by world_size, need to extend the shards
        # to ensure all groups needed by a head is sharded along with it
        n_groups = (mamba_config.n_groups + cls.extra_groups_for_head_shards(
            mamba_config.n_groups, world_size))

        # - heads and n_groups are TP-ed
        conv_dim = (intermediate_size + 2 * n_groups * mamba_config.d_state)
        conv_state_shape = (
            divide(conv_dim, world_size),
            mamba_config.d_conv - 1,
        )

        # These are not TP-ed as they depend on A, dt_bias, D
        # - they are typically small
        #   e.g., (h_heads, d_head, d_state) = (128, 64, 128)
        temporal_state_shape = (
            divide(mamba_config.n_heads, world_size),
            mamba_config.d_head,
            mamba_config.d_state,
        )

        return conv_state_shape, temporal_state_shape

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

        # get mamba page size
        mamba_page_size = MambaSpec(
            shapes=cls.get_mamba_cache_shape(vllm_config),
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


class NemotronHModelConfig(HybridAttentionMambaModelConfig):

    @classmethod
    def parse_mamba_config(
        cls, config: "PretrainedConfig"
    ) -> HybridAttentionMambaModelConfig.MambaConfig:
        return HybridAttentionMambaModelConfig.MambaConfig(
            expand=config.expand,
            n_groups=config.n_groups,
            n_heads=config.mamba_num_heads,
            d_head=config.mamba_head_dim,
            d_state=config.ssm_state_size,
            d_conv=config.conv_kernel,
        )


class Zamba2ModelConfig(HybridAttentionMambaModelConfig):

    @classmethod
    def parse_mamba_config(
        cls, config: "PretrainedConfig"
    ) -> HybridAttentionMambaModelConfig.MambaConfig:
        return HybridAttentionMambaModelConfig.MambaConfig(
            expand=config.mamba_expand,
            n_groups=config.mamba_ngroups,
            n_heads=config.n_mamba_heads,
            d_head=config.mamba_headdim,
            d_state=config.mamba_d_state,
            d_conv=config.mamba_d_conv,
        )


MODELS_CONFIG_MAP: dict[str, type[VerifyAndUpdateConfig]] = {
    "GteModel": SnowflakeGteNewModelConfig,
    "GteNewModel": GteNewModelConfig,
    "NomicBertModel": NomicBertModelConfig,
    "Qwen3ForSequenceClassification": Qwen3ForSequenceClassificationConfig,
    "XLMRobertaModel": JinaRobertaModelConfig,
    "FalconH1ForCausalLM": HybridAttentionMambaModelConfig,
    "BambaForCausalLM": HybridAttentionMambaModelConfig,
    "GraniteMoeHybridForCausalLM": HybridAttentionMambaModelConfig,
    "NemotronHForCausalLM": NemotronHModelConfig,
    "Zamba2ForCausalLM": Zamba2ModelConfig,
    "JinaVLForRanking": JinaVLForSequenceClassificationConfig,
}
