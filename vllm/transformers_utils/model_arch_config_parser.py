# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import huggingface_hub
from torch import nn
from transformers import AutoConfig, PretrainedConfig
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_MAPPING_NAMES,
)

from vllm import envs
from vllm.config.model_arch import (
    ModelArchitectureAudioConfig,
    ModelArchitectureConfig,
    ModelArchitectureTextConfig,
    ModelArchitectureVisionConfig,
)
from vllm.logger import init_logger
from vllm.transformers_utils.config import (
    _CONFIG_REGISTRY,
    _get_hf_token,
    _maybe_update_auto_config_kwargs,
    file_or_path_exists,
    get_hf_file_to_dict,
)
from vllm.utils.import_utils import LazyLoader

logger = init_logger(__name__)


class ModelArchConfigParserBase(ABC):
    @abstractmethod
    def parse(
        self,
        model: str | Path,
        trust_remote_code: bool,
        revision: str | None = None,
        code_revision: str | None = None,
        **kwargs,
    ) -> tuple[dict[str, Any], "ModelArchitectureConfig"]:
        raise NotImplementedError


def extract_num_hidden_layers(config_dict: dict[str, Any], model_type: str) -> int:
    if model_type in [
        "deepseek_mtp",
        "mimo_mtp",
        "glm4_moe_mtp",
        "ernie_mtp",
        "qwen3_next_mtp",
    ]:
        total_num_hidden_layers = config_dict.pop("num_nextn_predict_layers", 0)
    elif model_type == "longcat_flash_mtp":
        total_num_hidden_layers = config_dict.pop("num_nextn_predict_layers", 1)
    else:
        total_num_hidden_layers = config_dict.pop("num_hidden_layers", 0)

    return total_num_hidden_layers


def extract_use_deepseek_mla(
    config_dict: dict[str, Any], model_type: str | None
) -> bool:
    if not model_type:
        return False
    elif model_type in (
        "deepseek_v2",
        "deepseek_v3",
        "deepseek_v32",
        "deepseek_mtp",
        "kimi_k2",
        "kimi_linear",
        "longcat_flash",
    ):
        return config_dict.get("kv_lora_rank") is not None
    elif model_type == "eagle":
        # if the model is an EAGLE module, check for the
        # underlying architecture
        return (
            config_dict["model"]["model_type"]
            in ("deepseek_v2", "deepseek_v3", "deepseek_v32")
            and config_dict.get("kv_lora_rank") is not None
        )
    return False


def extract_head_size(
    config_dict: dict[str, Any], standard_fields: dict[str, Any]
) -> int:
    # TODO remove hard code
    if standard_fields["use_deepseek_mla"]:
        qk_rope_head_dim = config_dict.get("qk_rope_head_dim", 0)
        if not envs.VLLM_MLA_DISABLE:
            return config_dict["kv_lora_rank"] + qk_rope_head_dim
        else:
            qk_nope_head_dim = config_dict.get("qk_nope_head_dim", 0)
            if qk_rope_head_dim and qk_nope_head_dim:
                return qk_rope_head_dim + qk_nope_head_dim

    if standard_fields["model_type"] == "zamba2":
        return config_dict.pop("attention_head_dim")

    # TODO(xingyuliu): Check attention_free

    # NOTE: Some configs may set head_dim=None in the config
    if config_dict.get("head_dim") is not None:
        return config_dict.pop("head_dim")

    # NOTE: Some models (such as PLaMo2.1) use `hidden_size_per_head`
    if config_dict.get("hidden_size_per_head") is not None:
        return config_dict.pop("hidden_size_per_head")

    # FIXME(woosuk): This may not be true for all models.
    return standard_fields["hidden_size"] // standard_fields["num_attention_heads"]


def extract_total_num_kv_heads(
    config_dict: dict[str, Any], standard_fields: dict[str, Any]
) -> int:
    """Returns the total number of KV heads."""
    model_type = standard_fields["model_type"]
    # For GPTBigCode & Falcon:
    # NOTE: for falcon, when new_decoder_architecture is True, the
    # multi_query flag is ignored and we use n_head_kv for the number of
    # KV heads.
    falcon_model_types = ["falcon", "RefinedWeb", "RefinedWebModel"]
    new_decoder_arch_falcon = model_type in falcon_model_types and config_dict.get(
        "new_decoder_architecture", False
    )
    if not new_decoder_arch_falcon and config_dict.get("multi_query", False):
        # Multi-query attention, only one KV head.
        # Currently, tensor parallelism is not supported in this case.
        return 1

    # For DBRX and MPT
    if model_type == "mpt":
        if "kv_n_heads" in config_dict["attn_config"]:
            return config_dict["attn_config"]["kv_n_heads"]
        return standard_fields["num_attention_heads"]
    if model_type == "dbrx":
        attn_config = config_dict["attn_config"]
        return attn_config.get("kv_n_heads", standard_fields["num_attention_heads"])

    if model_type == "nemotron-nas":
        for block in config_dict["block_configs"]:
            if not block.attention.no_op:
                return (
                    standard_fields["num_attention_heads"]
                    // block.attention.n_heads_in_group
                )

        raise RuntimeError("Couldn't determine number of kv heads")

    # TODO(xingyuliu): Check attention_free

    attributes = [
        # For Falcon:
        "n_head_kv",
        "num_kv_heads",
        # For LLaMA-2:
        "num_key_value_heads",
        # For ChatGLM:
        "multi_query_group_num",
    ]
    for attr in attributes:
        num_kv_heads = config_dict.get(attr)
        if num_kv_heads is not None:
            config_dict.pop(attr)
            return num_kv_heads

    # For non-grouped-query attention models, the number of KV heads is
    # equal to the number of attention heads.
    return standard_fields["num_attention_heads"]


def extract_num_experts(config_dict: dict[str, Any]) -> int:
    """Returns the number of experts in the model."""
    num_expert_names = [
        "num_experts",  # Jamba
        "moe_num_experts",  # Dbrx
        "n_routed_experts",  # DeepSeek
        "num_local_experts",  # Mixtral
    ]
    for attr in num_expert_names:
        num_experts = config_dict.get(attr)
        if num_experts is not None:
            config_dict.pop(attr)
            return num_experts

    return 0


def extract_standard_text_config_field(
    config_dict: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    standard_fields = {}
    if "text_config" in config_dict:
        text_config_dict = config_dict["text_config"]
    else:
        text_config_dict = deepcopy(config_dict)
    standard_fields["model_type"] = text_config_dict.pop("model_type")

    standard_fields["hidden_size"] = text_config_dict.pop("hidden_size")

    (standard_fields["num_hidden_layers"]) = extract_num_hidden_layers(
        text_config_dict, standard_fields["model_type"]
    )
    standard_fields["num_attention_heads"] = text_config_dict.pop("num_attention_heads")

    standard_fields["use_deepseek_mla"] = extract_use_deepseek_mla(
        text_config_dict, standard_fields["model_type"]
    )
    standard_fields["head_dim"] = extract_head_size(text_config_dict, standard_fields)
    standard_fields["vocab_size"] = text_config_dict.pop("vocab_size")
    standard_fields["num_key_value_heads"] = extract_total_num_kv_heads(
        text_config_dict, standard_fields
    )
    standard_fields["num_experts"] = extract_num_experts(text_config_dict)

    return standard_fields, text_config_dict


if TYPE_CHECKING:
    import vllm.model_executor.models as me_models
else:
    me_models = LazyLoader("model_executor", globals(), "vllm.model_executor.models")


def get_per_layer_attention_cls(
    architectures: list[str],
    model_impl: str,
    text_config: ModelArchitectureTextConfig,
) -> list[type[nn.Module]]:
    assert len(architectures) == 1, "Only support len(architectures) == 1 for now"
    assert model_impl == "auto" or model_impl == "vllm"
    assert architectures[0] in me_models.ModelRegistry.models
    model_arch = architectures[0]
    model_cls = me_models.registry._try_load_model_cls(
        model_arch, me_models.ModelRegistry.models[model_arch]
    )
    # TODO: need to split sliding window attention from vllm.attention.layer.Attention
    per_layer_attention_cls = model_cls.get_per_layer_attention_cls(text_config)

    return per_layer_attention_cls


def get_quantization_config(
    model: str | Path, revision: str | None, config_dict: dict[str, Any]
) -> dict[str, Any]:
    # ModelOpt 0.31.0 and after saves the quantization config in the model
    # config file.
    quantization_config = config_dict.pop("quantization_config", None)

    # ModelOpt 0.29.0 and before saves the quantization config in a separate
    # "hf_quant_config.json" in the same directory as the model config file.
    if quantization_config is None and file_or_path_exists(
        model, "hf_quant_config.json", revision
    ):
        quantization_config = get_hf_file_to_dict(
            "hf_quant_config.json", model, revision
        )

    if quantization_config is not None:
        # config.quantization_config = quantization_config
        # auto-enable DeepGEMM UE8M0 if model config requests it
        scale_fmt = quantization_config.get("scale_fmt", None)
        if scale_fmt in ("ue8m0",):
            if not envs.is_set("VLLM_USE_DEEP_GEMM_E8M0"):
                os.environ["VLLM_USE_DEEP_GEMM_E8M0"] = "1"
                logger.info_once(
                    (
                        "Detected quantization_config.scale_fmt=%s; "
                        "enabling UE8M0 for DeepGEMM."
                    ),
                    scale_fmt,
                )
            elif not envs.VLLM_USE_DEEP_GEMM_E8M0:
                logger.warning_once(
                    (
                        "Model config requests UE8M0 "
                        "(quantization_config.scale_fmt=%s), but "
                        "VLLM_USE_DEEP_GEMM_E8M0=0 is set; "
                        "UE8M0 for DeepGEMM disabled."
                    ),
                    scale_fmt,
                )

    return quantization_config or {}


def get_torch_dtype(config_dict: dict[str, Any]):
    config_dtype = config_dict.pop("dtype", None)

    # Fallbacks for multi-modal models if the root config
    # does not define dtype
    if config_dtype is None:
        config_dtype = config_dict["text_config"].get("dtype", None)
    if config_dtype is None and "vision_config" in config_dict:
        config_dtype = config_dict["vision_config"].get("dtype", None)
    if config_dtype is None and hasattr(config_dict, "encoder_config"):
        config_dtype = config_dict["encoder_config"].get("dtype", None)

    return config_dtype


class HFModelArchConfigParser(ModelArchConfigParserBase):
    def parse(
        self,
        model: str | Path,
        trust_remote_code: bool,
        revision: str | None = None,
        code_revision: str | None = None,
        model_impl: str = "auto",
        **kwargs,
    ) -> tuple[dict[str, Any], "ModelArchitectureConfig"]:
        """Parse the HF config and create ModelArchitectureConfig."""

        is_gguf = kwargs.get("is_gguf", False)
        if is_gguf:
            kwargs["gguf_file"] = Path(model).name
            model = Path(model).parent

        kwargs["local_files_only"] = huggingface_hub.constants.HF_HUB_OFFLINE

        config_dict, _ = PretrainedConfig.get_config_dict(
            model,
            revision=revision,
            code_revision=code_revision,
            token=_get_hf_token(),
            **kwargs,
        )
        # Use custom model class if it's in our registry
        model_type = config_dict.get("model_type", "")

        if model_type in _CONFIG_REGISTRY:
            # TODO: check if need to write new config class that
            # inherient ModelArchitectureTextConfig for each of those models
            raise NotImplementedError
        else:
            # We use AutoConfig.from_pretrained to leverage some existing
            # standardization in PretrainedConfig
            try:
                kwargs = _maybe_update_auto_config_kwargs(kwargs, model_type=model_type)
                # https://github.com/huggingface/transformers/blob/e8a6eb3304033fdd9346fe3b3293309fe50de238/src/transformers/models/auto/configuration_auto.py#L1238
                config_dict = AutoConfig.from_pretrained(
                    model,
                    trust_remote_code=trust_remote_code,
                    revision=revision,
                    code_revision=code_revision,
                    token=_get_hf_token(),
                    **kwargs,
                ).to_dict()
            except ValueError as e:
                if (
                    not trust_remote_code
                    and "requires you to execute the configuration file" in str(e)
                ):
                    err_msg = (
                        "Failed to load the model config. If the model "
                        "is a custom model not yet available in the "
                        "HuggingFace transformers library, consider setting "
                        "`trust_remote_code=True` in LLM or using the "
                        "`--trust-remote-code` flag in the CLI."
                    )
                    raise RuntimeError(err_msg) from e
                else:
                    raise e

        architectures = config_dict.pop("architectures", [])
        quantization_config = get_quantization_config(model, revision, config_dict)
        torch_dtype = get_torch_dtype(config_dict)

        standard_fields, text_config_dict = extract_standard_text_config_field(
            config_dict
        )
        # Ensure no overlap between standard fields and remaining text config
        overlap = set(standard_fields.keys()) & set(text_config_dict.keys())
        assert len(overlap) == 0, (
            f"Standard fields and text config dict should not overlap, got {overlap}"
        )
        # Extract text config fields
        text_config = ModelArchitectureTextConfig(**standard_fields, **text_config_dict)

        # Special architecture mapping check for GGUF models
        if is_gguf:
            if model_type not in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
                raise RuntimeError(f"Can't get gguf config for {model_type}.")
            model_type = MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[model_type]
            architectures = [model_type]

        # Architecture mapping for models without explicit architectures field
        if not architectures:
            if model_type not in MODEL_MAPPING_NAMES:
                logger.warning(
                    "Model config does not have a top-level "
                    "'architectures' field: expecting "
                    "`model_arch_overrides={'architectures': ['...']}` "
                    "to be passed in engine args."
                )
            else:
                model_type = MODEL_MAPPING_NAMES[model_type]
                architectures = [model_type]

        vision_config_dict = config_dict.get("vision_config", {})
        audio_config_dict = config_dict.get("audio_config", {})

        per_layer_attention_cls = get_per_layer_attention_cls(
            architectures, model_impl, text_config
        )

        # Create ModelArchitectureConfig
        vision_config = (
            ModelArchitectureVisionConfig(**vision_config_dict)
            if vision_config_dict
            else None
        )
        audio_config = (
            ModelArchitectureAudioConfig(**audio_config_dict)
            if audio_config_dict
            else None
        )

        arch_config = ModelArchitectureConfig(
            architectures=architectures,
            model_type=model_type,
            quantization_config=quantization_config,
            torch_dtype=torch_dtype,
            per_layer_attention_cls=per_layer_attention_cls,
            text_config=text_config,
            vision=vision_config,
            audio=audio_config,
        )

        return config_dict, arch_config
