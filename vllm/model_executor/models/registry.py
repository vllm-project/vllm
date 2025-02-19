# SPDX-License-Identifier: Apache-2.0
"""
Whenever you add an architecture to this page, please also update
`tests/models/registry.py` with example HuggingFace models for it.
"""
import importlib
import os
import pickle
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import lru_cache
from typing import (AbstractSet, Callable, Dict, List, Optional, Tuple, Type,
                    TypeVar, Union)

import cloudpickle
import torch.nn as nn

from vllm.logger import init_logger

from .interfaces import (has_inner_state, is_attention_free, is_hybrid,
                         supports_cross_encoding, supports_multimodal,
                         supports_pp, supports_transcription)
from .interfaces_base import is_text_generation_model

logger = init_logger(__name__)

# yapf: disable
_TEXT_GENERATION_MODELS = {
    # [Decoder-only]
    "AquilaModel": ("llama", "LlamaForCausalLM"),
    "AquilaForCausalLM": ("llama", "LlamaForCausalLM"),  # AquilaChat2
    "ArcticForCausalLM": ("arctic", "ArcticForCausalLM"),
    # baichuan-7b, upper case 'C' in the class name
    "BaiChuanForCausalLM": ("baichuan", "BaiChuanForCausalLM"),
    # baichuan-13b, lower case 'c' in the class name
    "BaichuanForCausalLM": ("baichuan", "BaichuanForCausalLM"),
    "BambaForCausalLM": ("bamba", "BambaForCausalLM"),
    "BloomForCausalLM": ("bloom", "BloomForCausalLM"),
    "ChatGLMModel": ("chatglm", "ChatGLMForCausalLM"),
    "CohereForCausalLM": ("commandr", "CohereForCausalLM"),
    "Cohere2ForCausalLM": ("commandr", "CohereForCausalLM"),
    "DbrxForCausalLM": ("dbrx", "DbrxForCausalLM"),
    "DeciLMForCausalLM": ("decilm", "DeciLMForCausalLM"),
    "DeepseekForCausalLM": ("deepseek", "DeepseekForCausalLM"),
    "DeepseekV2ForCausalLM": ("deepseek_v2", "DeepseekV2ForCausalLM"),
    "DeepseekV3ForCausalLM": ("deepseek_v2", "DeepseekV3ForCausalLM"),
    "ExaoneForCausalLM": ("exaone", "ExaoneForCausalLM"),
    "FalconForCausalLM": ("falcon", "FalconForCausalLM"),
    "Fairseq2LlamaForCausalLM": ("fairseq2_llama", "Fairseq2LlamaForCausalLM"),
    "GemmaForCausalLM": ("gemma", "GemmaForCausalLM"),
    "Gemma2ForCausalLM": ("gemma2", "Gemma2ForCausalLM"),
    "GlmForCausalLM": ("glm", "GlmForCausalLM"),
    "GPT2LMHeadModel": ("gpt2", "GPT2LMHeadModel"),
    "GPTBigCodeForCausalLM": ("gpt_bigcode", "GPTBigCodeForCausalLM"),
    "GPTJForCausalLM": ("gpt_j", "GPTJForCausalLM"),
    "GPTNeoXForCausalLM": ("gpt_neox", "GPTNeoXForCausalLM"),
    "GraniteForCausalLM": ("granite", "GraniteForCausalLM"),
    "GraniteMoeForCausalLM": ("granitemoe", "GraniteMoeForCausalLM"),
    "GritLM": ("gritlm", "GritLM"),
    "InternLMForCausalLM": ("llama", "LlamaForCausalLM"),
    "InternLM2ForCausalLM": ("internlm2", "InternLM2ForCausalLM"),
    "InternLM2VEForCausalLM": ("internlm2_ve", "InternLM2VEForCausalLM"),
    "InternLM3ForCausalLM": ("llama", "LlamaForCausalLM"),
    "JAISLMHeadModel": ("jais", "JAISLMHeadModel"),
    "JambaForCausalLM": ("jamba", "JambaForCausalLM"),
    "LlamaForCausalLM": ("llama", "LlamaForCausalLM"),
    # For decapoda-research/llama-*
    "LLaMAForCausalLM": ("llama", "LlamaForCausalLM"),
    "MambaForCausalLM": ("mamba", "MambaForCausalLM"),
    "FalconMambaForCausalLM": ("mamba", "MambaForCausalLM"),
    "Mamba2ForCausalLM": ("mamba2", "Mamba2ForCausalLM"),
    "MiniCPMForCausalLM": ("minicpm", "MiniCPMForCausalLM"),
    "MiniCPM3ForCausalLM": ("minicpm3", "MiniCPM3ForCausalLM"),
    "MistralForCausalLM": ("llama", "LlamaForCausalLM"),
    "MixtralForCausalLM": ("mixtral", "MixtralForCausalLM"),
    "QuantMixtralForCausalLM": ("mixtral_quant", "MixtralForCausalLM"),
    # transformers's mpt class has lower case
    "MptForCausalLM": ("mpt", "MPTForCausalLM"),
    "MPTForCausalLM": ("mpt", "MPTForCausalLM"),
    "NemotronForCausalLM": ("nemotron", "NemotronForCausalLM"),
    "OlmoForCausalLM": ("olmo", "OlmoForCausalLM"),
    "Olmo2ForCausalLM": ("olmo2", "Olmo2ForCausalLM"),
    "OlmoeForCausalLM": ("olmoe", "OlmoeForCausalLM"),
    "OPTForCausalLM": ("opt", "OPTForCausalLM"),
    "OrionForCausalLM": ("orion", "OrionForCausalLM"),
    "PersimmonForCausalLM": ("persimmon", "PersimmonForCausalLM"),
    "PhiForCausalLM": ("phi", "PhiForCausalLM"),
    "Phi3ForCausalLM": ("phi3", "Phi3ForCausalLM"),
    "Phi3SmallForCausalLM": ("phi3_small", "Phi3SmallForCausalLM"),
    "PhiMoEForCausalLM": ("phimoe", "PhiMoEForCausalLM"),
    "QWenLMHeadModel": ("qwen", "QWenLMHeadModel"),
    "Qwen2ForCausalLM": ("qwen2", "Qwen2ForCausalLM"),
    "Qwen2MoeForCausalLM": ("qwen2_moe", "Qwen2MoeForCausalLM"),
    "RWForCausalLM": ("falcon", "FalconForCausalLM"),
    "StableLMEpochForCausalLM": ("stablelm", "StablelmForCausalLM"),
    "StableLmForCausalLM": ("stablelm", "StablelmForCausalLM"),
    "Starcoder2ForCausalLM": ("starcoder2", "Starcoder2ForCausalLM"),
    "SolarForCausalLM": ("solar", "SolarForCausalLM"),
    "TeleChat2ForCausalLM": ("telechat2", "TeleChat2ForCausalLM"),
    "XverseForCausalLM": ("llama", "LlamaForCausalLM"),
    # [Encoder-decoder]
    "BartModel": ("bart", "BartForConditionalGeneration"),
    "BartForConditionalGeneration": ("bart", "BartForConditionalGeneration"),
    "Florence2ForConditionalGeneration": ("florence2", "Florence2ForConditionalGeneration"),  # noqa: E501
}

_EMBEDDING_MODELS = {
    # [Text-only]
    "BertModel": ("bert", "BertEmbeddingModel"),
    "RobertaModel": ("roberta", "RobertaEmbeddingModel"),
    "RobertaForMaskedLM": ("roberta", "RobertaEmbeddingModel"),
    "XLMRobertaModel": ("roberta", "RobertaEmbeddingModel"),
    "DeciLMForCausalLM": ("decilm", "DeciLMForCausalLM"),
    "Gemma2Model": ("gemma2", "Gemma2ForCausalLM"),
    "GlmForCausalLM": ("glm", "GlmForCausalLM"),
    "GritLM": ("gritlm", "GritLM"),
    "InternLM2ForRewardModel": ("internlm2", "InternLM2ForRewardModel"),
    "JambaForSequenceClassification": ("jamba", "JambaForSequenceClassification"),  # noqa: E501
    "LlamaModel": ("llama", "LlamaForCausalLM"),
    **{
        # Multiple models share the same architecture, so we include them all
        k: (mod, arch) for k, (mod, arch) in _TEXT_GENERATION_MODELS.items()
        if arch == "LlamaForCausalLM"
    },
    "MistralModel": ("llama", "LlamaForCausalLM"),
    "Phi3ForCausalLM": ("phi3", "Phi3ForCausalLM"),
    "Qwen2Model": ("qwen2", "Qwen2EmbeddingModel"),
    "Qwen2ForCausalLM": ("qwen2", "Qwen2ForCausalLM"),
    "Qwen2ForRewardModel": ("qwen2_rm", "Qwen2ForRewardModel"),
    "Qwen2ForProcessRewardModel": ("qwen2_rm", "Qwen2ForProcessRewardModel"),
    "TeleChat2ForCausalLM": ("telechat2", "TeleChat2ForCausalLM"),
    # [Multimodal]
    "LlavaNextForConditionalGeneration": ("llava_next", "LlavaNextForConditionalGeneration"),  # noqa: E501
    "Phi3VForCausalLM": ("phi3v", "Phi3VForCausalLM"),
    "Qwen2VLForConditionalGeneration": ("qwen2_vl", "Qwen2VLForConditionalGeneration"),  # noqa: E501
    # [Auto-converted (see adapters.py)]
    "Qwen2ForSequenceClassification": ("qwen2", "Qwen2ForCausalLM"),
    # Technically PrithviGeoSpatialMAE is a model that works on images, both in
    # input and output. I am adding it here because it piggy-backs on embedding
    # models for the time being.
    "PrithviGeoSpatialMAE": ("prithvi_geospatial_mae", "PrithviGeoSpatialMAE"),
}

_CROSS_ENCODER_MODELS = {
    "BertForSequenceClassification": ("bert", "BertForSequenceClassification"),
    "RobertaForSequenceClassification": ("roberta",
                                         "RobertaForSequenceClassification"),
    "XLMRobertaForSequenceClassification": ("roberta",
                                            "RobertaForSequenceClassification"),
}

_MULTIMODAL_MODELS = {
    # [Decoder-only]
    "AriaForConditionalGeneration": ("aria", "AriaForConditionalGeneration"),
    "Blip2ForConditionalGeneration": ("blip2", "Blip2ForConditionalGeneration"),
    "ChameleonForConditionalGeneration": ("chameleon", "ChameleonForConditionalGeneration"),  # noqa: E501
    "DeepseekVLV2ForCausalLM": ("deepseek_vl2", "DeepseekVLV2ForCausalLM"),
    "FuyuForCausalLM": ("fuyu", "FuyuForCausalLM"),
    "GLM4VForCausalLM": ("glm4v", "GLM4VForCausalLM"),
    "H2OVLChatModel": ("h2ovl", "H2OVLChatModel"),
    "InternVLChatModel": ("internvl", "InternVLChatModel"),
    "Idefics3ForConditionalGeneration":("idefics3","Idefics3ForConditionalGeneration"),
    "LlavaForConditionalGeneration": ("llava", "LlavaForConditionalGeneration"),
    "LlavaNextForConditionalGeneration": ("llava_next", "LlavaNextForConditionalGeneration"),  # noqa: E501
    "LlavaNextVideoForConditionalGeneration": ("llava_next_video", "LlavaNextVideoForConditionalGeneration"),  # noqa: E501
    "LlavaOnevisionForConditionalGeneration": ("llava_onevision", "LlavaOnevisionForConditionalGeneration"),  # noqa: E501
    "MantisForConditionalGeneration": ("llava", "MantisForConditionalGeneration"),  # noqa: E501
    "MiniCPMO": ("minicpmo", "MiniCPMO"),
    "MiniCPMV": ("minicpmv", "MiniCPMV"),
    "MolmoForCausalLM": ("molmo", "MolmoForCausalLM"),
    "NVLM_D": ("nvlm_d", "NVLM_D_Model"),
    "PaliGemmaForConditionalGeneration": ("paligemma", "PaliGemmaForConditionalGeneration"),  # noqa: E501
    "Phi3VForCausalLM": ("phi3v", "Phi3VForCausalLM"),
    "PixtralForConditionalGeneration": ("pixtral", "PixtralForConditionalGeneration"),  # noqa: E501
    "QwenVLForConditionalGeneration": ("qwen_vl", "QwenVLForConditionalGeneration"),  # noqa: E501
    "Qwen2VLForConditionalGeneration": ("qwen2_vl", "Qwen2VLForConditionalGeneration"),  # noqa: E501
    "Qwen2_5_VLForConditionalGeneration": ("qwen2_5_vl", "Qwen2_5_VLForConditionalGeneration"),  # noqa: E501
    "Qwen2AudioForConditionalGeneration": ("qwen2_audio", "Qwen2AudioForConditionalGeneration"),  # noqa: E501
    "UltravoxModel": ("ultravox", "UltravoxModel"),
    # [Encoder-decoder]
    "MllamaForConditionalGeneration": ("mllama", "MllamaForConditionalGeneration"),  # noqa: E501
    "WhisperForConditionalGeneration": ("whisper", "WhisperForConditionalGeneration"),  # noqa: E501
}

_SPECULATIVE_DECODING_MODELS = {
    "EAGLEModel": ("eagle", "EAGLE"),
    "DeepSeekMTPModel": ("deepseek_mtp", "DeepSeekMTP"),
    "MedusaModel": ("medusa", "Medusa"),
    "MLPSpeculatorPreTrainedModel": ("mlp_speculator", "MLPSpeculator"),
}

_FALLBACK_MODEL = {
    "TransformersModel": ("transformers", "TransformersModel"),
}
# yapf: enable

_VLLM_MODELS = {
    **_TEXT_GENERATION_MODELS,
    **_EMBEDDING_MODELS,
    **_CROSS_ENCODER_MODELS,
    **_MULTIMODAL_MODELS,
    **_SPECULATIVE_DECODING_MODELS,
    **_FALLBACK_MODEL,
}

# This variable is used as the args for subprocess.run(). We
# can modify  this variable to alter the args if needed. e.g.
# when we use par format to pack things together, sys.executable
# might not be the target we want to run.
_SUBPROCESS_COMMAND = [
    sys.executable, "-m", "vllm.model_executor.models.registry"
]


@dataclass(frozen=True)
class _ModelInfo:
    architecture: str
    is_text_generation_model: bool
    is_pooling_model: bool
    supports_cross_encoding: bool
    supports_multimodal: bool
    supports_pp: bool
    has_inner_state: bool
    is_attention_free: bool
    is_hybrid: bool
    supports_transcription: bool

    @staticmethod
    def from_model_cls(model: Type[nn.Module]) -> "_ModelInfo":
        return _ModelInfo(
            architecture=model.__name__,
            is_text_generation_model=is_text_generation_model(model),
            is_pooling_model=True,  # Can convert any model into a pooling model
            supports_cross_encoding=supports_cross_encoding(model),
            supports_multimodal=supports_multimodal(model),
            supports_pp=supports_pp(model),
            has_inner_state=has_inner_state(model),
            is_attention_free=is_attention_free(model),
            is_hybrid=is_hybrid(model),
            supports_transcription=supports_transcription(model))


class _BaseRegisteredModel(ABC):

    @abstractmethod
    def inspect_model_cls(self) -> _ModelInfo:
        raise NotImplementedError

    @abstractmethod
    def load_model_cls(self) -> Type[nn.Module]:
        raise NotImplementedError


@dataclass(frozen=True)
class _RegisteredModel(_BaseRegisteredModel):
    """
    Represents a model that has already been imported in the main process.
    """

    interfaces: _ModelInfo
    model_cls: Type[nn.Module]

    @staticmethod
    def from_model_cls(model_cls: Type[nn.Module]):
        return _RegisteredModel(
            interfaces=_ModelInfo.from_model_cls(model_cls),
            model_cls=model_cls,
        )

    def inspect_model_cls(self) -> _ModelInfo:
        return self.interfaces

    def load_model_cls(self) -> Type[nn.Module]:
        return self.model_cls


@dataclass(frozen=True)
class _LazyRegisteredModel(_BaseRegisteredModel):
    """
    Represents a model that has not been imported in the main process.
    """
    module_name: str
    class_name: str

    # Performed in another process to avoid initializing CUDA
    def inspect_model_cls(self) -> _ModelInfo:
        return _run_in_subprocess(
            lambda: _ModelInfo.from_model_cls(self.load_model_cls()))

    def load_model_cls(self) -> Type[nn.Module]:
        mod = importlib.import_module(self.module_name)
        return getattr(mod, self.class_name)


@lru_cache(maxsize=128)
def _try_load_model_cls(
    model_arch: str,
    model: _BaseRegisteredModel,
) -> Optional[Type[nn.Module]]:
    from vllm.platforms import current_platform
    current_platform.verify_model_arch(model_arch)
    try:
        return model.load_model_cls()
    except Exception:
        logger.exception("Error in loading model architecture '%s'",
                         model_arch)
        return None


@lru_cache(maxsize=128)
def _try_inspect_model_cls(
    model_arch: str,
    model: _BaseRegisteredModel,
) -> Optional[_ModelInfo]:
    try:
        return model.inspect_model_cls()
    except Exception:
        logger.exception("Error in inspecting model architecture '%s'",
                         model_arch)
        return None


@dataclass
class _ModelRegistry:
    # Keyed by model_arch
    models: Dict[str, _BaseRegisteredModel] = field(default_factory=dict)

    def get_supported_archs(self) -> AbstractSet[str]:
        return self.models.keys()

    def register_model(
        self,
        model_arch: str,
        model_cls: Union[Type[nn.Module], str],
    ) -> None:
        """
        Register an external model to be used in vLLM.

        :code:`model_cls` can be either:

        - A :class:`torch.nn.Module` class directly referencing the model.
        - A string in the format :code:`<module>:<class>` which can be used to
          lazily import the model. This is useful to avoid initializing CUDA
          when importing the model and thus the related error
          :code:`RuntimeError: Cannot re-initialize CUDA in forked subprocess`.
        """
        if model_arch in self.models:
            logger.warning(
                "Model architecture %s is already registered, and will be "
                "overwritten by the new model class %s.", model_arch,
                model_cls)

        if isinstance(model_cls, str):
            split_str = model_cls.split(":")
            if len(split_str) != 2:
                msg = "Expected a string in the format `<module>:<class>`"
                raise ValueError(msg)

            model = _LazyRegisteredModel(*split_str)
        else:
            model = _RegisteredModel.from_model_cls(model_cls)

        self.models[model_arch] = model

    def _raise_for_unsupported(self, architectures: List[str]):
        all_supported_archs = self.get_supported_archs()

        if any(arch in all_supported_archs for arch in architectures):
            raise ValueError(
                f"Model architectures {architectures} failed "
                "to be inspected. Please check the logs for more details.")

        raise ValueError(
            f"Model architectures {architectures} are not supported for now. "
            f"Supported architectures: {all_supported_archs}")

    def _try_load_model_cls(self,
                            model_arch: str) -> Optional[Type[nn.Module]]:
        if model_arch not in self.models:
            return None

        return _try_load_model_cls(model_arch, self.models[model_arch])

    def _try_inspect_model_cls(self, model_arch: str) -> Optional[_ModelInfo]:
        if model_arch not in self.models:
            return None

        return _try_inspect_model_cls(model_arch, self.models[model_arch])

    def _normalize_archs(
        self,
        architectures: Union[str, List[str]],
    ) -> List[str]:
        if isinstance(architectures, str):
            architectures = [architectures]
        if not architectures:
            logger.warning("No model architectures are specified")

        normalized_arch = []
        for model in architectures:
            if model not in self.models:
                model = "TransformersModel"
            normalized_arch.append(model)
        return normalized_arch

    def inspect_model_cls(
        self,
        architectures: Union[str, List[str]],
    ) -> Tuple[_ModelInfo, str]:
        architectures = self._normalize_archs(architectures)

        for arch in architectures:
            model_info = self._try_inspect_model_cls(arch)
            if model_info is not None:
                return (model_info, arch)

        return self._raise_for_unsupported(architectures)

    def resolve_model_cls(
        self,
        architectures: Union[str, List[str]],
    ) -> Tuple[Type[nn.Module], str]:
        architectures = self._normalize_archs(architectures)

        for arch in architectures:
            model_cls = self._try_load_model_cls(arch)
            if model_cls is not None:
                return (model_cls, arch)

        return self._raise_for_unsupported(architectures)

    def is_text_generation_model(
        self,
        architectures: Union[str, List[str]],
    ) -> bool:
        model_cls, _ = self.inspect_model_cls(architectures)
        return model_cls.is_text_generation_model

    def is_pooling_model(
        self,
        architectures: Union[str, List[str]],
    ) -> bool:
        model_cls, _ = self.inspect_model_cls(architectures)
        return model_cls.is_pooling_model

    def is_cross_encoder_model(
        self,
        architectures: Union[str, List[str]],
    ) -> bool:
        model_cls, _ = self.inspect_model_cls(architectures)
        return model_cls.supports_cross_encoding

    def is_multimodal_model(
        self,
        architectures: Union[str, List[str]],
    ) -> bool:
        model_cls, _ = self.inspect_model_cls(architectures)
        return model_cls.supports_multimodal

    def is_pp_supported_model(
        self,
        architectures: Union[str, List[str]],
    ) -> bool:
        model_cls, _ = self.inspect_model_cls(architectures)
        return model_cls.supports_pp

    def model_has_inner_state(
        self,
        architectures: Union[str, List[str]],
    ) -> bool:
        model_cls, _ = self.inspect_model_cls(architectures)
        return model_cls.has_inner_state

    def is_attention_free_model(
        self,
        architectures: Union[str, List[str]],
    ) -> bool:
        model_cls, _ = self.inspect_model_cls(architectures)
        return model_cls.is_attention_free

    def is_hybrid_model(
        self,
        architectures: Union[str, List[str]],
    ) -> bool:
        model_cls, _ = self.inspect_model_cls(architectures)
        return model_cls.is_hybrid

    def is_transcription_model(
        self,
        architectures: Union[str, List[str]],
    ) -> bool:
        model_cls, _ = self.inspect_model_cls(architectures)
        return model_cls.supports_transcription


ModelRegistry = _ModelRegistry({
    model_arch:
    _LazyRegisteredModel(
        module_name=f"vllm.model_executor.models.{mod_relname}",
        class_name=cls_name,
    )
    for model_arch, (mod_relname, cls_name) in _VLLM_MODELS.items()
})

_T = TypeVar("_T")


def _run_in_subprocess(fn: Callable[[], _T]) -> _T:
    # NOTE: We use a temporary directory instead of a temporary file to avoid
    # issues like https://stackoverflow.com/questions/23212435/permission-denied-to-write-to-my-temporary-file
    with tempfile.TemporaryDirectory() as tempdir:
        output_filepath = os.path.join(tempdir, "registry_output.tmp")

        # `cloudpickle` allows pickling lambda functions directly
        input_bytes = cloudpickle.dumps((fn, output_filepath))

        # cannot use `sys.executable __file__` here because the script
        # contains relative imports
        returned = subprocess.run(_SUBPROCESS_COMMAND,
                                  input=input_bytes,
                                  capture_output=True)

        # check if the subprocess is successful
        try:
            returned.check_returncode()
        except Exception as e:
            # wrap raised exception to provide more information
            raise RuntimeError(f"Error raised in subprocess:\n"
                               f"{returned.stderr.decode()}") from e

        with open(output_filepath, "rb") as f:
            return pickle.load(f)


def _run() -> None:
    # Setup plugins
    from vllm.plugins import load_general_plugins
    load_general_plugins()

    fn, output_file = pickle.loads(sys.stdin.buffer.read())

    result = fn()

    with open(output_file, "wb") as f:
        f.write(pickle.dumps(result))


if __name__ == "__main__":
    _run()
