from dataclasses import dataclass, field
from typing import AbstractSet, Mapping, Optional


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

    is_available_online: bool = True
    """
    Set this to ``False`` if the name of this architecture no longer exists on
    the HF repo. To maintain backwards compatibility, we have not removed them
    from the main model registry, so without this flag the registry tests will
    fail.
    """

    trust_remote_code: bool = False
    """The ``trust_remote_code`` level required to load the model."""


# yapf: disable
_TEXT_GENERATION_EXAMPLE_MODELS = {
    # [Decoder-only]
    "AquilaModel": _HfExamplesInfo("BAAI/AquilaChat-7B",
                                   trust_remote_code=True),
    "AquilaForCausalLM": _HfExamplesInfo("BAAI/AquilaChat2-7B",
                                         trust_remote_code=True),
    "ArcticForCausalLM": _HfExamplesInfo("Snowflake/snowflake-arctic-instruct",
                                         trust_remote_code=True),
    "AriaForConditionalGeneration": _HfExamplesInfo("rhymes-ai/Aria",
                                                    trust_remote_code=True),
    "BaiChuanForCausalLM": _HfExamplesInfo("baichuan-inc/Baichuan-7B",
                                         trust_remote_code=True),
    "BaichuanForCausalLM": _HfExamplesInfo("baichuan-inc/Baichuan2-7B-chat",
                                         trust_remote_code=True),
    "BloomForCausalLM": _HfExamplesInfo("bigscience/bloomz-1b1"),
    # ChatGLMModel supports multimodal
    "CohereForCausalLM": _HfExamplesInfo("CohereForAI/c4ai-command-r-v01",
                                         trust_remote_code=True),
    "DbrxForCausalLM": _HfExamplesInfo("databricks/dbrx-instruct"),
    "DeciLMForCausalLM": _HfExamplesInfo("Deci/DeciLM-7B-instruct",
                                         trust_remote_code=True),
    "DeepseekForCausalLM": _HfExamplesInfo("deepseek-ai/deepseek-llm-7b-chat"),
    "DeepseekV2ForCausalLM": _HfExamplesInfo("deepseek-ai/DeepSeek-V2-Lite-Chat",  # noqa: E501
                                         trust_remote_code=True),
    "ExaoneForCausalLM": _HfExamplesInfo("LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct"),  # noqa: E501
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
    "InternLMForCausalLM": _HfExamplesInfo("internlm/internlm-chat-7b",
                                           trust_remote_code=True),
    "InternLM2ForCausalLM": _HfExamplesInfo("internlm/internlm2-chat-7b",
                                            trust_remote_code=True),
    "InternLM2VEForCausalLM": _HfExamplesInfo("OpenGVLab/Mono-InternVL-2B",
                                              trust_remote_code=True),
    "JAISLMHeadModel": _HfExamplesInfo("inceptionai/jais-13b-chat"),
    "JambaForCausalLM": _HfExamplesInfo("ai21labs/AI21-Jamba-1.5-Mini"),
    "LlamaForCausalLM": _HfExamplesInfo("meta-llama/Meta-Llama-3-8B"),
    "LLaMAForCausalLM": _HfExamplesInfo("decapoda-research/llama-7b-hf",
                                        is_available_online=False),
    "MambaForCausalLM": _HfExamplesInfo("state-spaces/mamba-130m-hf"),
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
    # QWenLMHeadModel supports multimodal
    "Qwen2ForCausalLM": _HfExamplesInfo("Qwen/Qwen2-7B-Instruct"),
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
    # Florence-2 uses BartFastTokenizer which can't be loaded from AutoTokenizer
    # Therefore, we borrow the BartTokenizer from the original Bart model
    "Florence2ForConditionalGeneration": _HfExamplesInfo("microsoft/Florence-2-base",  # noqa: E501
                                                         tokenizer="facebook/bart-base",
                                                         trust_remote_code=True),  # noqa: E501
}

_EMBEDDING_EXAMPLE_MODELS = {
    # [Text-only]
    "BertModel": _HfExamplesInfo("BAAI/bge-base-en-v1.5"),
    "Gemma2Model": _HfExamplesInfo("BAAI/bge-multilingual-gemma2"),
    "LlamaModel": _HfExamplesInfo("llama", is_available_online=False),
    "MistralModel": _HfExamplesInfo("intfloat/e5-mistral-7b-instruct"),
    "Qwen2Model": _HfExamplesInfo("ssmits/Qwen2-7B-Instruct-embed-base"),
    "Qwen2ForRewardModel": _HfExamplesInfo("Qwen/Qwen2.5-Math-RM-72B"),
    "Qwen2ForSequenceClassification": _HfExamplesInfo("jason9693/Qwen2.5-1.5B-apeach"),  # noqa: E501
    "RobertaModel": _HfExamplesInfo("sentence-transformers/stsb-roberta-base-v2"),  # noqa: E501
    "RobertaForMaskedLM": _HfExamplesInfo("sentence-transformers/all-roberta-large-v1"),  # noqa: E501
    "XLMRobertaModel": _HfExamplesInfo("intfloat/multilingual-e5-large"),
    # [Multimodal]
    "LlavaNextForConditionalGeneration": _HfExamplesInfo("royokong/e5-v"),
    "Phi3VForCausalLM": _HfExamplesInfo("TIGER-Lab/VLM2Vec-Full",
                                         trust_remote_code=True),
    "Qwen2VLForConditionalGeneration": _HfExamplesInfo("MrLight/dse-qwen2-2b-mrl-v1"), # noqa: E501
}

_CROSS_ENCODER_EXAMPLE_MODELS = {
    # [Text-only]
    "BertForSequenceClassification": _HfExamplesInfo("cross-encoder/ms-marco-MiniLM-L-6-v2"),  # noqa: E501
    "RobertaForSequenceClassification": _HfExamplesInfo("cross-encoder/quora-roberta-base"),  # noqa: E501
    "XLMRobertaForSequenceClassification": _HfExamplesInfo("BAAI/bge-reranker-v2-m3"),  # noqa: E501
}

_MULTIMODAL_EXAMPLE_MODELS = {
    # [Decoder-only]
    "Blip2ForConditionalGeneration": _HfExamplesInfo("Salesforce/blip2-opt-2.7b"),  # noqa: E501
    "ChameleonForConditionalGeneration": _HfExamplesInfo("facebook/chameleon-7b"),  # noqa: E501
    "ChatGLMModel": _HfExamplesInfo("THUDM/glm-4v-9b",
                                    extras={"text_only": "THUDM/chatglm3-6b"},
                                    trust_remote_code=True),
    "ChatGLMForConditionalGeneration": _HfExamplesInfo("chatglm2-6b",
                                                       is_available_online=False),
    "FuyuForCausalLM": _HfExamplesInfo("adept/fuyu-8b"),
    "H2OVLChatModel": _HfExamplesInfo("h2oai/h2ovl-mississippi-800m"),
    "InternVLChatModel": _HfExamplesInfo("OpenGVLab/InternVL2-1B",
                                         trust_remote_code=True),
    "Idefics3ForConditionalGeneration": _HfExamplesInfo("HuggingFaceM4/Idefics3-8B-Llama3"),  # noqa: E501
    "LlavaForConditionalGeneration": _HfExamplesInfo("llava-hf/llava-1.5-7b-hf",
                                                     extras={"mistral": "mistral-community/pixtral-12b"}),  # noqa: E501
    "LlavaNextForConditionalGeneration": _HfExamplesInfo("llava-hf/llava-v1.6-mistral-7b-hf"),  # noqa: E501
    "LlavaNextVideoForConditionalGeneration": _HfExamplesInfo("llava-hf/LLaVA-NeXT-Video-7B-hf"),  # noqa: E501
    "LlavaOnevisionForConditionalGeneration": _HfExamplesInfo("llava-hf/llava-onevision-qwen2-0.5b-ov-hf"),  # noqa: E501
    "MantisForConditionalGeneration": _HfExamplesInfo("TIGER-Lab/Mantis-8B-siglip-llama3"),  # noqa: E501
    "MiniCPMV": _HfExamplesInfo("openbmb/MiniCPM-Llama3-V-2_5",
                                trust_remote_code=True),
    "MolmoForCausalLM": _HfExamplesInfo("allenai/Molmo-7B-D-0924",
                                        trust_remote_code=True),
    "NVLM_D": _HfExamplesInfo("nvidia/NVLM-D-72B",
                              trust_remote_code=True),
    "PaliGemmaForConditionalGeneration": _HfExamplesInfo("google/paligemma-3b-pt-224"),  # noqa: E501
    "Phi3VForCausalLM": _HfExamplesInfo("microsoft/Phi-3-vision-128k-instruct",
                                        trust_remote_code=True),
    "PixtralForConditionalGeneration": _HfExamplesInfo("mistralai/Pixtral-12B-2409",  # noqa: E501
                                                       tokenizer_mode="mistral"),
    "QWenLMHeadModel": _HfExamplesInfo("Qwen/Qwen-VL-Chat",
                                       extras={"text_only": "Qwen/Qwen-7B-Chat"},  # noqa: E501
                                       trust_remote_code=True),
    "Qwen2AudioForConditionalGeneration": _HfExamplesInfo("Qwen/Qwen2-Audio-7B-Instruct"),  # noqa: E501
    "Qwen2VLForConditionalGeneration": _HfExamplesInfo("Qwen/Qwen2-VL-2B-Instruct"),  # noqa: E501
    "UltravoxModel": _HfExamplesInfo("fixie-ai/ultravox-v0_3"),
    # [Encoder-decoder]
    "MllamaForConditionalGeneration": _HfExamplesInfo("meta-llama/Llama-3.2-11B-Vision-Instruct"),  # noqa: E501
}

_SPECULATIVE_DECODING_EXAMPLE_MODELS = {
    "EAGLEModel": _HfExamplesInfo("JackFram/llama-68m",
                                  speculative_model="abhigoyal/vllm-eagle-llama-68m-random"),  # noqa: E501
    "MedusaModel": _HfExamplesInfo("JackFram/llama-68m",
                                   speculative_model="abhigoyal/vllm-medusa-llama-68m-random"),  # noqa: E501
    "MLPSpeculatorPreTrainedModel": _HfExamplesInfo("JackFram/llama-160m",
                                                    speculative_model="ibm-fms/llama-160m-accelerator"),  # noqa: E501
}

_EXAMPLE_MODELS = {
    **_TEXT_GENERATION_EXAMPLE_MODELS,
    **_EMBEDDING_EXAMPLE_MODELS,
    **_CROSS_ENCODER_EXAMPLE_MODELS,
    **_MULTIMODAL_EXAMPLE_MODELS,
    **_SPECULATIVE_DECODING_EXAMPLE_MODELS,
}


class HfExampleModels:
    def __init__(self, hf_models: Mapping[str, _HfExamplesInfo]) -> None:
        super().__init__()

        self.hf_models = hf_models

    def get_supported_archs(self) -> AbstractSet[str]:
        return self.hf_models.keys()

    def get_hf_info(self, model_arch: str) -> _HfExamplesInfo:
        return self.hf_models[model_arch]


HF_EXAMPLE_MODELS = HfExampleModels(_EXAMPLE_MODELS)
