# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import final

import torch
from safetensors.torch import _TYPES as _SAFETENSORS_TO_TORCH_DTYPE
from transformers import PretrainedConfig

from vllm import envs
from vllm.config.model_arch import (
    ModelArchitectureConfig,
)
from vllm.config.utils import getattr_iter
from vllm.logger import init_logger
from vllm.transformers_utils.config import (
    try_get_safetensors_metadata,
)
from vllm.utils.torch_utils import common_broadcastable_dtype

logger = init_logger(__name__)


class ModelArchConfigConvertorBase:
    def __init__(self, hf_config: PretrainedConfig, hf_text_config: PretrainedConfig):
        self.hf_config = hf_config
        self.hf_text_config = hf_text_config

    def get_architecture(self) -> str | None:
        architectures = getattr(self.hf_config, "architectures", None)
        if architectures:
            assert len(architectures) == 1, (
                f"len(architectures) should be 1, got {len(architectures)}"
            )
            return architectures[0]
        return None

    def get_num_hidden_layers(self) -> int:
        return getattr(self.hf_text_config, "num_hidden_layers", 0)

    def get_total_num_attention_heads(self) -> int:
        return getattr(self.hf_text_config, "num_attention_heads", 0)

    def get_vocab_size(self) -> int:
        return getattr(self.hf_text_config, "vocab_size", 0)

    def get_hidden_size(self) -> int:
        return getattr(self.hf_text_config, "hidden_size", 0)

    def get_head_size(self) -> int:
        if self.is_deepseek_mla():
            qk_rope_head_dim = getattr(self.hf_text_config, "qk_rope_head_dim", 0)
            if not envs.VLLM_MLA_DISABLE:
                return self.hf_text_config.kv_lora_rank + qk_rope_head_dim
            else:
                qk_nope_head_dim = getattr(self.hf_text_config, "qk_nope_head_dim", 0)
                if qk_rope_head_dim and qk_nope_head_dim:
                    return qk_rope_head_dim + qk_nope_head_dim

        # NOTE: Some configs may set head_dim=None in the config
        if getattr(self.hf_text_config, "head_dim", None) is not None:
            return self.hf_text_config.head_dim

        # NOTE: Some models (such as PLaMo2.1) use `hidden_size_per_head`
        if getattr(self.hf_text_config, "hidden_size_per_head", None) is not None:
            return self.hf_text_config.hidden_size_per_head

        # FIXME(woosuk): This may not be true for all models.
        return (
            self.hf_text_config.hidden_size // self.hf_text_config.num_attention_heads
        )

    def get_total_num_kv_heads(self) -> int:
        attributes = [
            # For Falcon:
            "n_head_kv",
            "num_kv_heads",
            # For LLaMA-2:
            "num_key_value_heads",
            # For ChatGLM:
            "multi_query_group_num",
        ]
        # For non-grouped-query attention models, the number of KV heads is
        # equal to the number of attention heads.
        default_factory = lambda: self.hf_text_config.num_attention_heads
        return getattr_iter(
            self.hf_text_config, attributes, default_factory=default_factory
        )

    def get_num_experts_from_block_configs(self) -> int:
        """Check block_configs for heterogeneous models (e.g., NemotronH).

        For heterogeneous models with varying expert counts per layer,
        returns the MAX to ensure all expert weights can be loaded.
        """
        max_experts = 0
        block_configs = getattr(self.hf_text_config, "block_configs", None)
        if block_configs:
            for block in block_configs:
                if isinstance(block, dict):
                    if block.get("block_type", "") == "moe":
                        max_experts = max(max_experts, block.get("n_routed_experts", 0))
                else:
                    if getattr(block, "block_type", "") == "moe":
                        max_experts = max(
                            max_experts, getattr(block, "n_routed_experts", 0)
                        )
        return max_experts

    def get_num_experts(self) -> int:
        """Returns the number of experts in the model."""
        num_expert_names = [
            "num_experts",  # Jamba
            "moe_num_experts",  # Dbrx
            "n_routed_experts",  # DeepSeek
            "num_local_experts",  # Mixtral
        ]

        num_experts = getattr_iter(self.hf_text_config, num_expert_names, 0)
        if isinstance(num_experts, list):
            # Ernie VL's remote code uses list[int]...
            # The values are always the same so we just take the first one.
            return num_experts[0]

        if not num_experts:
            num_experts = self.get_num_experts_from_block_configs()
        return num_experts

    @final
    @classmethod
    def get_torch_dtype(
        cls, hf_config: PretrainedConfig, model_id: str, revision: str | None
    ):
        # NOTE: getattr(config, "dtype", torch.float32) is not correct
        # because config.dtype can be None.
        config_dtype = getattr(hf_config, "dtype", None)

        # Fallbacks for multi-modal models if the root config
        # does not define dtype
        if config_dtype is None:
            config_dtype = getattr(hf_config.get_text_config(), "dtype", None)
        if config_dtype is None and hasattr(hf_config, "vision_config"):
            config_dtype = getattr(hf_config.vision_config, "dtype", None)
        if config_dtype is None and hasattr(hf_config, "encoder_config"):
            config_dtype = getattr(hf_config.encoder_config, "dtype", None)

        # Try to read the dtype of the weights if they are in safetensors format
        if config_dtype is None:
            repo_mt = try_get_safetensors_metadata(model_id, revision=revision)

            if repo_mt and (files_mt := repo_mt.files_metadata):
                param_dtypes: set[torch.dtype] = {
                    _SAFETENSORS_TO_TORCH_DTYPE[dtype_str]
                    for file_mt in files_mt.values()
                    for dtype_str in file_mt.parameter_count
                    if dtype_str in _SAFETENSORS_TO_TORCH_DTYPE
                }

                if param_dtypes:
                    return common_broadcastable_dtype(param_dtypes)

        if config_dtype is None:
            config_dtype = torch.float32

        return config_dtype

    def _normalize_quantization_config(self, config: PretrainedConfig):
        quant_cfg = getattr(config, "quantization_config", None)
        if quant_cfg is None:
            # compressed-tensors uses a "compression_config" key
            quant_cfg = getattr(config, "compression_config", None)

        else:
            # Set quant_method for ModelOpt models.
            producer_name = quant_cfg.get("producer", {}).get("name")
            if producer_name == "modelopt":
                quant_algo = quant_cfg.get("quantization", {}).get("quant_algo")
                if quant_algo is not None:
                    quant_algo_upper = str(quant_algo).upper()
                    if quant_algo_upper in {
                        "FP8",
                        "FP8_PER_CHANNEL_PER_TOKEN",
                        "FP8_PB_WO",
                    }:
                        quant_cfg["quant_method"] = "modelopt"
                    elif quant_algo_upper == "NVFP4":
                        quant_cfg["quant_method"] = "modelopt_fp4"
                    else:
                        raise ValueError(f"Unknown ModelOpt quant algo: {quant_algo}")

        if quant_cfg is not None:
            # Use the community standard 'quant_method'
            quant_method = quant_cfg.get("quant_method", "").lower()

            # Normalize library names
            quant_method = quant_method.replace(
                "compressed_tensors", "compressed-tensors"
            )

            quant_cfg["quant_method"] = quant_method

        return quant_cfg

    def get_quantization_config(self):
        quant_cfg = self._normalize_quantization_config(self.hf_config)
        if quant_cfg is None and (
            text_config := getattr(self.hf_config, "text_config", None)
        ):
            # Check the text config as well for multi-modal models.
            quant_cfg = self._normalize_quantization_config(text_config)
        return quant_cfg

    def is_deepseek_mla(self) -> bool:
        if not hasattr(self.hf_text_config, "model_type"):
            return False
        elif self.hf_text_config.model_type in (
            "deepseek_v2",
            "deepseek_v3",
            "deepseek_v32",
            "deepseek_mtp",
            "glm4_moe_lite",
            "glm4_moe_lite_mtp",
            "kimi_k2",
            "kimi_linear",
            "longcat_flash",
            "pangu_ultra_moe",
            "pangu_ultra_moe_mtp",
        ):
            return self.hf_text_config.kv_lora_rank is not None
        elif self.hf_text_config.model_type == "eagle":
            # if the model is an EAGLE module, check for the
            # underlying architecture
            return (
                self.hf_text_config.model.model_type
                in ("deepseek_v2", "deepseek_v3", "deepseek_v32", "deepseek_mtp")
                and self.hf_text_config.kv_lora_rank is not None
            )
        return False

    def derive_max_model_len_and_key(self) -> tuple[float, str | None]:
        derived_max_model_len = float("inf")
        possible_keys = [
            # OPT
            "max_position_embeddings",
            # GPT-2
            "n_positions",
            # MPT
            "max_seq_len",
            # ChatGLM2
            "seq_length",
            # Command-R
            "model_max_length",
            # Whisper
            "max_target_positions",
            # Others
            "max_sequence_length",
            "max_seq_length",
            "seq_len",
        ]
        # Choose the smallest "max_length" from the possible keys
        max_len_key = None
        for key in possible_keys:
            max_len = getattr(self.hf_text_config, key, None)
            if max_len is not None:
                if max_len < derived_max_model_len:
                    max_len_key = key
                derived_max_model_len = min(derived_max_model_len, max_len)

        # For Command-R / Cohere, Cohere2 / Aya Vision models
        if tmp_max_len := getattr(self.hf_text_config, "model_max_length", None):
            max_len_key = "model_max_length"
            derived_max_model_len = tmp_max_len
        return derived_max_model_len, max_len_key

    def convert(self) -> ModelArchitectureConfig:
        model_arch_config = ModelArchitectureConfig(
            architecture=self.get_architecture(),
            model_type=self.hf_config.model_type,
            text_model_type=getattr(self.hf_text_config, "model_type", None),
            hidden_size=self.get_hidden_size(),
            total_num_hidden_layers=self.get_num_hidden_layers(),
            total_num_attention_heads=self.get_total_num_attention_heads(),
            head_size=self.get_head_size(),
            vocab_size=self.get_vocab_size(),
            total_num_kv_heads=self.get_total_num_kv_heads(),
            num_experts=self.get_num_experts(),
            quantization_config=self.get_quantization_config(),
            is_deepseek_mla=self.is_deepseek_mla(),
            derived_max_model_len_and_key=self.derive_max_model_len_and_key(),
        )

        return model_arch_config


class MambaModelArchConfigConvertor(ModelArchConfigConvertorBase):
    def get_head_size(self) -> int:
        return 0

    def get_total_num_kv_heads(self) -> int:
        return 0


class TerratorchModelArchConfigConvertor(ModelArchConfigConvertorBase):
    def get_head_size(self) -> int:
        return 0

    def get_total_num_kv_heads(self) -> int:
        return 0


class MedusaModelArchConfigConvertor(ModelArchConfigConvertorBase):
    def get_head_size(self) -> int:
        return 0

    def get_total_num_kv_heads(self) -> int:
        return 0


class Zamba2ModelArchConfigConvertor(ModelArchConfigConvertorBase):
    def get_head_size(self) -> int:
        return getattr(self.hf_text_config, "attention_head_dim", 0)


class FalconModelArchConfigConvertor(ModelArchConfigConvertorBase):
    def get_total_num_kv_heads(self) -> int:
        # NOTE: for falcon, when new_decoder_architecture is True, the
        # multi_query flag is ignored and we use n_head_kv for the number of
        # KV heads.
        new_decoder_arch_falcon = getattr(
            self.hf_text_config, "new_decoder_architecture", False
        )

        if not new_decoder_arch_falcon and getattr(
            self.hf_text_config, "multi_query", False
        ):
            # Multi-query attention, only one KV head.
            return 1

        # Use the base implementation which checks n_head_kv, num_kv_heads, etc.
        return super().get_total_num_kv_heads()


class MPTModelArchConfigConvertor(ModelArchConfigConvertorBase):
    def get_total_num_kv_heads(self) -> int:
        if "kv_n_heads" in self.hf_text_config.attn_config:
            return self.hf_text_config.attn_config["kv_n_heads"]
        return self.hf_text_config.num_attention_heads


class DbrxModelArchConfigConvertor(ModelArchConfigConvertorBase):
    def get_total_num_kv_heads(self) -> int:
        return getattr(
            self.hf_text_config.attn_config,
            "kv_n_heads",
            self.hf_text_config.num_attention_heads,
        )


class NemotronNasModelArchConfigConvertor(ModelArchConfigConvertorBase):
    def get_total_num_kv_heads(self) -> int:
        for block in self.hf_text_config.block_configs:
            if not block.attention.no_op:
                return (
                    self.hf_text_config.num_attention_heads
                    // block.attention.n_heads_in_group
                )
        raise RuntimeError(
            "Could not determine the number of key-value attention heads "
            "from model configuration. "
            f"Architecture: {self.get_architecture()}. "
            "This usually indicates an unsupported model architecture or "
            "missing configuration. "
            "Please check if your model is supported at: "
            "https://docs.vllm.ai/en/latest/models/supported_models.html"
        )


class DeepSeekMTPModelArchConfigConvertor(ModelArchConfigConvertorBase):
    def get_num_hidden_layers(self) -> int:
        return getattr(self.hf_text_config, "num_nextn_predict_layers", 0)


class MimoMTPModelArchConfigConvertor(ModelArchConfigConvertorBase):
    def get_num_hidden_layers(self) -> int:
        return getattr(self.hf_text_config, "num_nextn_predict_layers", 0)


class GLM4MoeMTPModelArchConfigConvertor(ModelArchConfigConvertorBase):
    def get_num_hidden_layers(self) -> int:
        return getattr(self.hf_text_config, "num_nextn_predict_layers", 0)


class ErnieMTPModelArchConfigConvertor(ModelArchConfigConvertorBase):
    def get_num_hidden_layers(self) -> int:
        return getattr(self.hf_text_config, "num_nextn_predict_layers", 0)


class Qwen3NextMTPModelArchConfigConvertor(ModelArchConfigConvertorBase):
    def get_num_hidden_layers(self) -> int:
        return getattr(self.hf_text_config, "num_nextn_predict_layers", 0)


class PanguUltraMoeMTPModelArchConfigConvertor(ModelArchConfigConvertorBase):
    def get_num_hidden_layers(self) -> int:
        return getattr(self.hf_text_config, "num_nextn_predict_layers", 0)


class LongCatFlashMTPModelArchConfigConvertor(ModelArchConfigConvertorBase):
    def get_num_hidden_layers(self) -> int:
        return getattr(self.hf_text_config, "num_nextn_predict_layers", 1)


# hf_config.model_type -> convertor class
MODEL_ARCH_CONFIG_CONVERTORS = {
    "mamba": MambaModelArchConfigConvertor,
    "falcon_mamba": MambaModelArchConfigConvertor,
    "timm_wrapper": TerratorchModelArchConfigConvertor,
    "medusa": MedusaModelArchConfigConvertor,
    "zamba2": Zamba2ModelArchConfigConvertor,
    "mpt": MPTModelArchConfigConvertor,
    "dbrx": DbrxModelArchConfigConvertor,
    "falcon": FalconModelArchConfigConvertor,
    "RefinedWeb": FalconModelArchConfigConvertor,
    "RefinedWebModel": FalconModelArchConfigConvertor,
    "nemotron-nas": NemotronNasModelArchConfigConvertor,
    "deepseek_mtp": DeepSeekMTPModelArchConfigConvertor,
    "qwen3_next_mtp": Qwen3NextMTPModelArchConfigConvertor,
    "mimo_mtp": MimoMTPModelArchConfigConvertor,
    "glm4_moe_mtp": GLM4MoeMTPModelArchConfigConvertor,
    "glm_ocr_mtp": GLM4MoeMTPModelArchConfigConvertor,
    "ernie_mtp": ErnieMTPModelArchConfigConvertor,
    "pangu_ultra_moe_mtp": PanguUltraMoeMTPModelArchConfigConvertor,
    "longcat_flash_mtp": LongCatFlashMTPModelArchConfigConvertor,
}
