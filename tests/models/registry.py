# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import AbstractSet, Any, Literal, Mapping, Optional

import pytest
from packaging.version import Version
from transformers import __version__ as TRANSFORMERS_VERSION


@dataclass(frozen=True)
class _HfExamplesInfo:
    default: str
    """The default model to use for testing this architecture."""

    extras: Mapping[str, str] = field(default_factory=dict)
    """Extra models to use for testing this architecture."""

    tokenizer: Optional[str] = None
    """Set the tokenizer to load for this architecture."""

    tokenizer_mode: str = "auto"
    """Set the tokenizer type for this architecture."""

    speculative_model: Optional[str] = None
    """
    The default model to use for testing this architecture, which is only used
    for speculative decoding.
    """

    min_transformers_version: Optional[str] = None
    """
    The minimum version of HF Transformers that is required to run this model.
    """

    is_available_online: bool = True
    """
    Set this to ``False`` if the name of this architecture no longer exists on
    the HF repo. To maintain backwards compatibility, we have not removed them
    from the main model registry, so without this flag the registry tests will
    fail.
    """

    trust_remote_code: bool = False
    """The ``trust_remote_code`` level required to load the model."""

    hf_overrides: dict[str, Any] = field(default_factory=dict)
    """The ``hf_overrides`` required to load the model."""

    def check_transformers_version(
        self,
        *,
        on_fail: Literal["error", "skip"],
    ) -> None:
        """
        If the installed transformers version does not meet the requirements,
        perform the given action.
        """
        if self.min_transformers_version is None:
            return

        current_version = TRANSFORMERS_VERSION
        required_version = self.min_transformers_version
        if Version(current_version) < Version(required_version):
            msg = (
                f"You have `transformers=={current_version}` installed, but "
                f"`transformers>={required_version}` is required to run this "
                "model")

            if on_fail == "error":
                raise RuntimeError(msg)
            else:
                pytest.skip(msg)

    def check_available_online(
        self,
        *,
        on_fail: Literal["error", "skip"],
    ) -> None:
        """
        If the model is not available online, perform the given action.
        """
        if not self.is_available_online:
            msg = "Model is not available online"

            if on_fail == "error":
                raise RuntimeError(msg)
            else:
                pytest.skip(msg)


# yapf: disable
_TEXT_GENERATION_EXAMPLE_MODELS = {
    # [Decoder-only]
    "AquilaModel": _HfExamplesInfo("BAAI/AquilaChat-7B",
                                   trust_remote_code=True),
    "AquilaForCausalLM": _HfExamplesInfo("BAAI/AquilaChat2-7B",
                                         trust_remote_code=True),
    "ArcticForCausalLM": _HfExamplesInfo("Snowflake/snowflake-arctic-instruct",
                                         trust_remote_code=True),
    "BaiChuanForCausalLM": _HfExamplesInfo("baichuan-inc/Baichuan-7B",
                                         trust_remote_code=True),
    "BaichuanForCausalLM": _HfExamplesInfo("baichuan-inc/Baichuan2-7B-chat",
                                         trust_remote_code=True),
    "BambaForCausalLM": _HfExamplesInfo("ibm-ai-platform/Bamba-9B"),
    "BloomForCausalLM": _HfExamplesInfo("bigscience/bloomz-1b1"),
    "ChatGLMModel": _HfExamplesInfo("THUDM/chatglm3-6b",
                                    trust_remote_code=True),
    "CohereForCausalLM": _HfExamplesInfo("CohereForAI/c4ai-command-r-v01",
                                         trust_remote_code=True),
    "Cohere2ForCausalLM": _HfExamplesInfo("CohereForAI/c4ai-command-r7b-12-2024", # noqa: E501
                                         trust_remote_code=True),
    "DbrxForCausalLM": _HfExamplesInfo("databricks/dbrx-instruct"),
    "DeciLMForCausalLM": _HfExamplesInfo("Deci/DeciLM-7B-instruct",
                                         trust_remote_code=True),
    "DeepseekForCausalLM": _HfExamplesInfo("deepseek-ai/deepseek-llm-7b-chat"),
    "DeepseekV2ForCausalLM": _HfExamplesInfo("deepseek-ai/DeepSeek-V2-Lite-Chat",  # noqa: E501
                                         trust_remote_code=True),
    "DeepseekV3ForCausalLM": _HfExamplesInfo("deepseek-ai/DeepSeek-V3",  # noqa: E501
                                         trust_remote_code=True),
    "ExaoneForCausalLM": _HfExamplesInfo("LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct"),  # noqa: E501
    "Fairseq2LlamaForCausalLM": _HfExamplesInfo("mgleize/fairseq2-dummy-Llama-3.2-1B"),  # noqa: E501
    "FalconForCausalLM": _HfExamplesInfo("tiiuae/falcon-7b"),
    "GemmaForCausalLM": _HfExamplesInfo("google/gemma-2b"),
    "Gemma2ForCausalLM": _HfExamplesInfo("google/gemma-2-9b"),
    "GlmForCausalLM": _HfExamplesInfo("THUDM/glm-4-9b-chat-hf"),
    "GPT2LMHeadModel": _HfExamplesInfo("gpt2"),
    "GPTBigCodeForCausalLM": _HfExamplesInfo("bigcode/starcoder"),
    "GPTJForCausalLM": _HfExamplesInfo("EleutherAI/gpt-j-6b"),
    "GPTNeoXForCausalLM": _HfExamplesInfo("EleutherAI/pythia-160m"),
    "GraniteForCausalLM": _HfExamplesInfo("ibm/PowerLM-3b"),
    "GraniteMoeForCausalLM": _HfExamplesInfo("ibm/PowerMoE-3b"),
    "Grok1ModelForCausalLM": _HfExamplesInfo("hpcai-tech/grok-1",
                                             trust_remote_code=True),
    "InternLMForCausalLM": _HfExamplesInfo("internlm/internlm-chat-7b",
                                           trust_remote_code=True),
    "InternLM2ForCausalLM": _HfExamplesInfo("internlm/internlm2-chat-7b",
                                            trust_remote_code=True),
    "InternLM2VEForCausalLM": _HfExamplesInfo("OpenGVLab/Mono-InternVL-2B",
                                              trust_remote_code=True),
    "InternLM3ForCausalLM": _HfExamplesInfo("internlm/internlm3-8b-instruct",
                                            trust_remote_code=True),
    "JAISLMHeadModel": _HfExamplesInfo("inceptionai/jais-13b-chat"),
    "JambaForCausalLM": _HfExamplesInfo("ai21labs/AI21-Jamba-1.5-Mini",
                                        extras={"tiny": "ai21labs/Jamba-tiny-dev"}),  # noqa: E501
    "LlamaForCausalLM": _HfExamplesInfo("meta-llama/Llama-3.2-1B-Instruct"),
    "LLaMAForCausalLM": _HfExamplesInfo("decapoda-research/llama-7b-hf",
                                        is_available_online=False),
    "MambaForCausalLM": _HfExamplesInfo("state-spaces/mamba-130m-hf"),
    "Mamba2ForCausalLM": _HfExamplesInfo("mistralai/Mamba-Codestral-7B-v0.1",
                                         is_available_online=False),
    "FalconMambaForCausalLM": _HfExamplesInfo("tiiuae/falcon-mamba-7b-instruct"),  # noqa: E501
    "MiniCPMForCausalLM": _HfExamplesInfo("openbmb/MiniCPM-2B-sft-bf16",
                                         trust_remote_code=True),
    "MiniCPM3ForCausalLM": _HfExamplesInfo("openbmb/MiniCPM3-4B",
                                         trust_remote_code=True),
    "MistralForCausalLM": _HfExamplesInfo("mistralai/Mistral-7B-Instruct-v0.1"),
    "MixtralForCausalLM": _HfExamplesInfo("mistralai/Mixtral-8x7B-Instruct-v0.1"),  # noqa: E501
    "QuantMixtralForCausalLM": _HfExamplesInfo("mistral-community/Mixtral-8x22B-v0.1-AWQ"),  # noqa: E501
    "MptForCausalLM": _HfExamplesInfo("mpt", is_available_online=False),
    "MPTForCausalLM": _HfExamplesInfo("mosaicml/mpt-7b"),
    "NemotronForCausalLM": _HfExamplesInfo("nvidia/Minitron-8B-Base"),
    "OlmoForCausalLM": _HfExamplesInfo("allenai/OLMo-1B-hf"),
    "Olmo2ForCausalLM": _HfExamplesInfo("shanearora/OLMo-7B-1124-hf"),
    "OlmoeForCausalLM": _HfExamplesInfo("allenai/OLMoE-1B-7B-0924-Instruct"),
    "OPTForCausalLM": _HfExamplesInfo("facebook/opt-iml-max-1.3b"),
    "OrionForCausalLM": _HfExamplesInfo("OrionStarAI/Orion-14B-Chat",
                                        trust_remote_code=True),
    "PersimmonForCausalLM": _HfExamplesInfo("adept/persimmon-8b-chat"),
    "PhiForCausalLM": _HfExamplesInfo("microsoft/phi-2"),
    "Phi3ForCausalLM": _HfExamplesInfo("microsoft/Phi-3-mini-4k-instruct"),
    "Phi3SmallForCausalLM": _HfExamplesInfo("microsoft/Phi-3-small-8k-instruct",
                                            trust_remote_code=True),
    "PhiMoEForCausalLM": _HfExamplesInfo("microsoft/Phi-3.5-MoE-instruct",
                                         trust_remote_code=True),
    "QWenLMHeadModel": _HfExamplesInfo("Qwen/Qwen-7B-Chat",
                                       trust_remote_code=True),
    "Qwen2ForCausalLM": _HfExamplesInfo("Qwen/Qwen2-7B-Instruct",
                                        extras={"2.5": "Qwen/Qwen2.5-7B-Instruct"}), # noqa: E501
    "Qwen2MoeForCausalLM": _HfExamplesInfo("Qwen/Qwen1.5-MoE-A2.7B-Chat"),
    "RWForCausalLM": _HfExamplesInfo("tiiuae/falcon-40b",
                                     is_available_online=False),
    "StableLMEpochForCausalLM": _HfExamplesInfo("stabilityai/stablelm-zephyr-3b",  # noqa: E501
                                                is_available_online=False),
    "StableLmForCausalLM": _HfExamplesInfo("stabilityai/stablelm-3b-4e1t"),
    "Starcoder2ForCausalLM": _HfExamplesInfo("bigcode/starcoder2-3b"),
    "SolarForCausalLM": _HfExamplesInfo("upstage/solar-pro-preview-instruct"),
    "TeleChat2ForCausalLM": _HfExamplesInfo("Tele-AI/TeleChat2-3B",
                                            trust_remote_code=True),
    "XverseForCausalLM": _HfExamplesInfo("xverse/XVERSE-7B-Chat",
                                         is_available_online=False,
                                         trust_remote_code=True),
    # [Encoder-decoder]
    "BartModel": _HfExamplesInfo("facebook/bart-base"),
    "BartForConditionalGeneration": _HfExamplesInfo("facebook/bart-large-cnn"),
}

_EMBEDDING_EXAMPLE_MODELS = {
    # [Text-only]
    "BertModel": _HfExamplesInfo("BAAI/bge-base-en-v1.5"),
    "Gemma2Model": _HfExamplesInfo("BAAI/bge-multilingual-gemma2"),
    "GritLM": _HfExamplesInfo("parasail-ai/GritLM-7B-vllm"),
    "InternLM2ForRewardModel": _HfExamplesInfo("internlm/internlm2-1_8b-reward",
                                               trust_remote_code=True),
    "JambaForSequenceClassification": _HfExamplesInfo("ai21labs/Jamba-tiny-reward-dev"),  # noqa: E501
    "LlamaModel": _HfExamplesInfo("llama", is_available_online=False),
    "MistralModel": _HfExamplesInfo("intfloat/e5-mistral-7b-instruct"),
    "Qwen2Model": _HfExamplesInfo("ssmits/Qwen2-7B-Instruct-embed-base"),
    "Qwen2ForRewardModel": _HfExamplesInfo("Qwen/Qwen2.5-Math-RM-72B"),
    "Qwen2ForProcessRewardModel": _HfExamplesInfo("Qwen/Qwen2.5-Math-PRM-7B"),
    "Qwen2ForSequenceClassification": _HfExamplesInfo("jason9693/Qwen2.5-1.5B-apeach"),  # noqa: E501
    "RobertaModel": _HfExamplesInfo("sentence-transformers/stsb-roberta-base-v2"),  # noqa: E501
    "RobertaForMaskedLM": _HfExamplesInfo("sentence-transformers/all-roberta-large-v1"),  # noqa: E501
    "XLMRobertaModel": _HfExamplesInfo("intfloat/multilingual-e5-small"),
    # [Multimodal]
    "LlavaNextForConditionalGeneration": _HfExamplesInfo("royokong/e5-v"),
    "Phi3VForCausalLM": _HfExamplesInfo("TIGER-Lab/VLM2Vec-Full",
                                         trust_remote_code=True),
    "Qwen2VLForConditionalGeneration": _HfExamplesInfo("MrLight/dse-qwen2-2b-mrl-v1"), # noqa: E501
    # The model on Huggingface is currently being updated,
    # hence I temporarily mark it as not available online
    "PrithviGeoSpatialMAE": _HfExamplesInfo("ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11",  # noqa: E501
                                            is_available_online=False),
}

_CROSS_ENCODER_EXAMPLE_MODELS = {
    # [Text-only]
    "BertForSequenceClassification": _HfExamplesInfo("cross-encoder/ms-marco-MiniLM-L-6-v2"),  # noqa: E501
    "RobertaForSequenceClassification": _HfExamplesInfo("cross-encoder/quora-roberta-base"),  # noqa: E501
    "XLMRobertaForSequenceClassification": _HfExamplesInfo("BAAI/bge-reranker-v2-m3"),  # noqa: E501
}

_MULTIMODAL_EXAMPLE_MODELS = {
    # [Decoder-only]
    "AriaForConditionalGeneration": _HfExamplesInfo("rhymes-ai/Aria"),
    "Blip2ForConditionalGeneration": _HfExamplesInfo("Salesforce/blip2-opt-2.7b"),  # noqa: E501
    "ChameleonForConditionalGeneration": _HfExamplesInfo("facebook/chameleon-7b"),  # noqa: E501
    "DeepseekVLV2ForCausalLM": _HfExamplesInfo("deepseek-ai/deepseek-vl2-tiny",  # noqa: E501
                                               hf_overrides={"architectures": ["DeepseekVLV2ForCausalLM"]}),  # noqa: E501
    "FuyuForCausalLM": _HfExamplesInfo("adept/fuyu-8b"),
    "GLM4VForCausalLM": _HfExamplesInfo("THUDM/glm-4v-9b",
                                        trust_remote_code=True,
                                        hf_overrides={"architectures": ["GLM4VForCausalLM"]}),  # noqa: E501
    "H2OVLChatModel": _HfExamplesInfo("h2oai/h2ovl-mississippi-800m",
                                      extras={"2b": "h2oai/h2ovl-mississippi-2b"}),  # noqa: E501
    "InternVLChatModel": _HfExamplesInfo("OpenGVLab/InternVL2-1B",
                                         extras={"2B": "OpenGVLab/InternVL2-2B"},  # noqa: E501
                                         trust_remote_code=True),
    "Idefics3ForConditionalGeneration": _HfExamplesInfo("HuggingFaceM4/Idefics3-8B-Llama3",  # noqa: E501
                                                        {"tiny": "HuggingFaceTB/SmolVLM-256M-Instruct"}),  # noqa: E501
    "LlavaForConditionalGeneration": _HfExamplesInfo("llava-hf/llava-1.5-7b-hf",
                                                     extras={"mistral": "mistral-community/pixtral-12b"}),  # noqa: E501
    "LlavaNextForConditionalGeneration": _HfExamplesInfo("llava-hf/llava-v1.6-mistral-7b-hf"),  # noqa: E501
    "LlavaNextVideoForConditionalGeneration": _HfExamplesInfo("llava-hf/LLaVA-NeXT-Video-7B-hf"),  # noqa: E501
    "LlavaOnevisionForConditionalGeneration": _HfExamplesInfo("llava-hf/llava-onevision-qwen2-0.5b-ov-hf"),  # noqa: E501
    "MantisForConditionalGeneration": _HfExamplesInfo("TIGER-Lab/Mantis-8B-siglip-llama3",  # noqa: E501
                                                      hf_overrides={"architectures": ["MantisForConditionalGeneration"]}),  # noqa: E501
    "MiniCPMO": _HfExamplesInfo("openbmb/MiniCPM-o-2_6",
                                trust_remote_code=True),
    "MiniCPMV": _HfExamplesInfo("openbmb/MiniCPM-Llama3-V-2_5",
                                extras={"2.6": "openbmb/MiniCPM-V-2_6"},  # noqa: E501
                                trust_remote_code=True),
    "MolmoForCausalLM": _HfExamplesInfo("allenai/Molmo-7B-D-0924",
                                        extras={"olmo": "allenai/Molmo-7B-O-0924"},  # noqa: E501
                                        trust_remote_code=True),
    "NVLM_D": _HfExamplesInfo("nvidia/NVLM-D-72B",
                              trust_remote_code=True),
    "PaliGemmaForConditionalGeneration": _HfExamplesInfo("google/paligemma-3b-mix-224",  # noqa: E501
                                                         extras={"v2": "google/paligemma2-3b-ft-docci-448"}),  # noqa: E501
    "Phi3VForCausalLM": _HfExamplesInfo("microsoft/Phi-3-vision-128k-instruct",
                                        trust_remote_code=True),
    "PixtralForConditionalGeneration": _HfExamplesInfo("mistralai/Pixtral-12B-2409",  # noqa: E501
                                                       tokenizer_mode="mistral"),
    "QwenVLForConditionalGeneration": _HfExamplesInfo("Qwen/Qwen-VL",
                                                      extras={"chat": "Qwen/Qwen-VL-Chat"},  # noqa: E501
                                                      trust_remote_code=True,
                                                      hf_overrides={"architectures": ["QwenVLForConditionalGeneration"]}),  # noqa: E501
    "Qwen2AudioForConditionalGeneration": _HfExamplesInfo("Qwen/Qwen2-Audio-7B-Instruct"),  # noqa: E501
    "Qwen2VLForConditionalGeneration": _HfExamplesInfo("Qwen/Qwen2-VL-2B-Instruct"),  # noqa: E501
    "Qwen2_5_VLForConditionalGeneration": _HfExamplesInfo("Qwen/Qwen2.5-VL-3B-Instruct",  # noqa: E501
                                                          min_transformers_version="4.49"),  # noqa: E501
    "UltravoxModel": _HfExamplesInfo("fixie-ai/ultravox-v0_4",
                                     extras={"v0.5": "fixie-ai/ultravox-v0_5-llama-3_2-1b"},  # noqa: E501
                                     trust_remote_code=True),
    # [Encoder-decoder]
    # Florence-2 uses BartFastTokenizer which can't be loaded from AutoTokenizer
    # Therefore, we borrow the BartTokenizer from the original Bart model
    "Florence2ForConditionalGeneration": _HfExamplesInfo("microsoft/Florence-2-base",  # noqa: E501
                                                         tokenizer="facebook/bart-base",
                                                         trust_remote_code=True),  # noqa: E501
    "MllamaForConditionalGeneration": _HfExamplesInfo("meta-llama/Llama-3.2-11B-Vision-Instruct"),  # noqa: E501
    "WhisperForConditionalGeneration": _HfExamplesInfo("openai/whisper-large-v3"),  # noqa: E501
}

_SPECULATIVE_DECODING_EXAMPLE_MODELS = {
    "EAGLEModel": _HfExamplesInfo("JackFram/llama-68m",
                                  speculative_model="abhigoyal/vllm-eagle-llama-68m-random"),  # noqa: E501
    "MedusaModel": _HfExamplesInfo("JackFram/llama-68m",
                                   speculative_model="abhigoyal/vllm-medusa-llama-68m-random"),  # noqa: E501
    "MLPSpeculatorPreTrainedModel": _HfExamplesInfo("JackFram/llama-160m",
                                                    speculative_model="ibm-ai-platform/llama-160m-accelerator"),  # noqa: E501
    "DeepSeekMTPModel": _HfExamplesInfo("luccafong/deepseek_mtp_main_random",
                                        speculative_model="luccafong/deepseek_mtp_draft_random",  # noqa: E501
                                        trust_remote_code=True),
}

_FALLBACK_MODEL = {
    "TransformersModel": _HfExamplesInfo("ArthurZ/Ilama-3.2-1B", trust_remote_code=True),  # noqa: E501
}

_EXAMPLE_MODELS = {
    **_TEXT_GENERATION_EXAMPLE_MODELS,
    **_EMBEDDING_EXAMPLE_MODELS,
    **_CROSS_ENCODER_EXAMPLE_MODELS,
    **_MULTIMODAL_EXAMPLE_MODELS,
    **_SPECULATIVE_DECODING_EXAMPLE_MODELS,
    **_FALLBACK_MODEL,
}


class HfExampleModels:
    def __init__(self, hf_models: Mapping[str, _HfExamplesInfo]) -> None:
        super().__init__()

        self.hf_models = hf_models

    def get_supported_archs(self) -> AbstractSet[str]:
        return self.hf_models.keys()

    def get_hf_info(self, model_arch: str) -> _HfExamplesInfo:
        return self.hf_models[model_arch]

    def find_hf_info(self, model_id: str) -> _HfExamplesInfo:
        for info in self.hf_models.values():
            if info.default == model_id:
                return info

        # Fallback to extras
        for info in self.hf_models.values():
            if any(extra == model_id for extra in info.extras.values()):
                return info

        raise ValueError(f"No example model defined for {model_id}")


HF_EXAMPLE_MODELS = HfExampleModels(_EXAMPLE_MODELS)
