# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Mapping, Set
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import pytest
from packaging.version import Version
from transformers import __version__ as TRANSFORMERS_VERSION

from vllm.config import TokenizerMode


@dataclass(frozen=True)
class _HfExamplesInfo:
    default: str
    """The default model to use for testing this architecture."""

    extras: Mapping[str, str] = field(default_factory=dict)
    """Extra models to use for testing this architecture."""

    tokenizer: Optional[str] = None
    """Set the tokenizer to load for this architecture."""

    tokenizer_mode: TokenizerMode = "auto"
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

    max_transformers_version: Optional[str] = None
    """
    The maximum version of HF Transformers that this model runs on.
    """

    transformers_version_reason: Optional[str] = None
    """
    The reason for the minimum/maximum version requirement.
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

    v0_only: bool = False
    """The model is only available with the vLLM V0 engine."""

    hf_overrides: dict[str, Any] = field(default_factory=dict)
    """The ``hf_overrides`` required to load the model."""

    max_model_len: Optional[int] = None
    """
    The maximum model length to use for this model. Some models default to a
    length that is too large to fit into memory in CI.
    """

    revision: Optional[str] = None
    """
    The specific revision (commit hash, tag, or branch) to use for the model.
    If not specified, the default revision will be used.
    """

    def check_transformers_version(
        self,
        *,
        on_fail: Literal["error", "skip", "return"],
        check_min_version: bool = True,
        check_max_version: bool = True,
    ) -> Optional[str]:
        """
        If the installed transformers version does not meet the requirements,
        perform the given action.
        """
        if (self.min_transformers_version is None
                and self.max_transformers_version is None):
            return None

        current_version = TRANSFORMERS_VERSION
        cur_base_version = Version(current_version).base_version
        min_version = self.min_transformers_version
        max_version = self.max_transformers_version
        msg = f"`transformers=={current_version}` installed, but `transformers"
        # Only check the base version for the min/max version, otherwise preview
        # models cannot be run because `x.yy.0.dev0`<`x.yy.0`
        if (check_min_version and min_version
                and Version(cur_base_version) < Version(min_version)):
            msg += f">={min_version}` is required to run this model."
        elif (check_max_version and max_version
              and Version(cur_base_version) > Version(max_version)):
            msg += f"<={max_version}` is required to run this model."
        else:
            return None

        if self.transformers_version_reason:
            msg += f" Reason: {self.transformers_version_reason}"

        if on_fail == "error":
            raise RuntimeError(msg)
        elif on_fail == "skip":
            pytest.skip(msg)

        return msg

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
    "ArceeForCausalLM": _HfExamplesInfo("arcee-ai/AFM-4.5B-Base"),
    "ArcticForCausalLM": _HfExamplesInfo("Snowflake/snowflake-arctic-instruct",
                                         trust_remote_code=True),
    "BaiChuanForCausalLM": _HfExamplesInfo("baichuan-inc/Baichuan-7B",
                                         trust_remote_code=True),
    "BaichuanForCausalLM": _HfExamplesInfo("baichuan-inc/Baichuan2-7B-chat",
                                         trust_remote_code=True),
    "BailingMoeForCausalLM": _HfExamplesInfo("inclusionAI/Ling-lite-1.5",
                                         trust_remote_code=True),
    "BambaForCausalLM": _HfExamplesInfo("ibm-ai-platform/Bamba-9B-v1",
                                        min_transformers_version="4.55.1",
                                        extras={"tiny": "hmellor/tiny-random-BambaForCausalLM"}),  # noqa: E501
    "BloomForCausalLM": _HfExamplesInfo("bigscience/bloom-560m",
                                        {"1b": "bigscience/bloomz-1b1"}),
    "ChatGLMModel": _HfExamplesInfo("zai-org/chatglm3-6b",
                                    trust_remote_code=True,
                                    max_transformers_version="4.48"),
    "ChatGLMForConditionalGeneration": _HfExamplesInfo("thu-coai/ShieldLM-6B-chatglm3",  # noqa: E501
                                                       trust_remote_code=True),
    "CohereForCausalLM": _HfExamplesInfo("CohereForAI/c4ai-command-r-v01",
                                         trust_remote_code=True),
    "Cohere2ForCausalLM": _HfExamplesInfo("CohereForAI/c4ai-command-r7b-12-2024", # noqa: E501
                                         trust_remote_code=True),
    "DbrxForCausalLM": _HfExamplesInfo("databricks/dbrx-instruct"),
    "DeciLMForCausalLM": _HfExamplesInfo("nvidia/Llama-3_3-Nemotron-Super-49B-v1", # noqa: E501
                                         trust_remote_code=True),
    "DeepseekForCausalLM": _HfExamplesInfo("deepseek-ai/deepseek-llm-7b-chat"),
    "DeepseekV2ForCausalLM": _HfExamplesInfo("deepseek-ai/DeepSeek-V2-Lite-Chat",  # noqa: E501
                                         trust_remote_code=True),
    "DeepseekV3ForCausalLM": _HfExamplesInfo("deepseek-ai/DeepSeek-V3",  # noqa: E501
                                         trust_remote_code=True),
    "Ernie4_5ForCausalLM": _HfExamplesInfo("baidu/ERNIE-4.5-0.3B-PT",
                                            min_transformers_version="4.54"),
    "Ernie4_5_MoeForCausalLM": _HfExamplesInfo("baidu/ERNIE-4.5-21B-A3B-PT",
                                               min_transformers_version="4.54"),
    "ExaoneForCausalLM": _HfExamplesInfo("LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct",
                                         trust_remote_code=True),
    "Exaone4ForCausalLM": _HfExamplesInfo("LGAI-EXAONE/EXAONE-4.0-32B",
                                          min_transformers_version="4.54"),
    "Fairseq2LlamaForCausalLM": _HfExamplesInfo("mgleize/fairseq2-dummy-Llama-3.2-1B"),  # noqa: E501
    "FalconForCausalLM": _HfExamplesInfo("tiiuae/falcon-7b"),
    "FalconH1ForCausalLM":_HfExamplesInfo("tiiuae/Falcon-H1-0.5B-Base"),
    "GemmaForCausalLM": _HfExamplesInfo("google/gemma-1.1-2b-it"),
    "Gemma2ForCausalLM": _HfExamplesInfo("google/gemma-2-9b"),
    "Gemma3ForCausalLM": _HfExamplesInfo("google/gemma-3-1b-it"),
    "Gemma3nForCausalLM": _HfExamplesInfo("google/gemma-3n-E2B-it",
                                          min_transformers_version="4.53"),
    "GlmForCausalLM": _HfExamplesInfo("zai-org/glm-4-9b-chat-hf"),
    "Glm4ForCausalLM": _HfExamplesInfo("zai-org/GLM-4-9B-0414"),
    "Glm4MoeForCausalLM": _HfExamplesInfo("zai-org/GLM-4.5",
                                          min_transformers_version="4.54"),   # noqa: E501
    "GPT2LMHeadModel": _HfExamplesInfo("openai-community/gpt2",
                                       {"alias": "gpt2"}),
    "GPTBigCodeForCausalLM": _HfExamplesInfo("bigcode/starcoder",
                                             {"tiny": "bigcode/tiny_starcoder_py"}),  # noqa: E501
    "GPTJForCausalLM": _HfExamplesInfo("Milos/slovak-gpt-j-405M",
                                       {"6b": "EleutherAI/gpt-j-6b"}),
    "GPTNeoXForCausalLM": _HfExamplesInfo("EleutherAI/pythia-70m",
                                          {"1b": "EleutherAI/pythia-1.4b"}),
    "GptOssForCausalLM": _HfExamplesInfo("lmsys/gpt-oss-20b-bf16"),
    "GraniteForCausalLM": _HfExamplesInfo("ibm/PowerLM-3b"),
    "GraniteMoeForCausalLM": _HfExamplesInfo("ibm/PowerMoE-3b"),
    "GraniteMoeHybridForCausalLM": _HfExamplesInfo("ibm-granite/granite-4.0-tiny-preview"),  # noqa: E501
    "GraniteMoeSharedForCausalLM": _HfExamplesInfo("ibm-research/moe-7b-1b-active-shared-experts"),  # noqa: E501
    "Grok1ModelForCausalLM": _HfExamplesInfo("hpcai-tech/grok-1",
                                             trust_remote_code=True),
    "HunYuanMoEV1ForCausalLM": _HfExamplesInfo("tencent/Hunyuan-A13B-Instruct",
                                               trust_remote_code=True),
    # TODO: Remove is_available_online once their config.json is fixed
    "HunYuanDenseV1ForCausalLM":_HfExamplesInfo("tencent/Hunyuan-7B-Instruct-0124",
                                                trust_remote_code=True,
                                                is_available_online=False),
    "HCXVisionForCausalLM": _HfExamplesInfo(
        "naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B",
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
                                        min_transformers_version="4.55.1",
                                        extras={
                                            "tiny": "ai21labs/Jamba-tiny-dev",
                                            "random": "ai21labs/Jamba-tiny-random",  # noqa: E501
                                        }),
    "LlamaForCausalLM": _HfExamplesInfo("meta-llama/Llama-3.2-1B-Instruct",
                                        extras={"guard": "meta-llama/Llama-Guard-3-1B",  # noqa: E501
                                                "hermes": "NousResearch/Hermes-3-Llama-3.1-8B", # noqa: E501
                                                "fp8": "RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8"}),  # noqa: E501
    "LLaMAForCausalLM": _HfExamplesInfo("decapoda-research/llama-7b-hf",
                                        is_available_online=False),
    "Llama4ForCausalLM": _HfExamplesInfo("meta-llama/Llama-4-Scout-17B-16E-Instruct", # noqa: E501
                                         is_available_online=False),
    "MambaForCausalLM": _HfExamplesInfo("state-spaces/mamba-130m-hf"),
    "Mamba2ForCausalLM": _HfExamplesInfo("mistralai/Mamba-Codestral-7B-v0.1"),
    "FalconMambaForCausalLM": _HfExamplesInfo("tiiuae/falcon-mamba-7b-instruct"),  # noqa: E501
    "MiniCPMForCausalLM": _HfExamplesInfo("openbmb/MiniCPM-2B-sft-bf16",
                                         trust_remote_code=True),
    "MiniCPM3ForCausalLM": _HfExamplesInfo("openbmb/MiniCPM3-4B",
                                         trust_remote_code=True),
    "MiniMaxForCausalLM": _HfExamplesInfo("MiniMaxAI/MiniMax-Text-01-hf"),
    "MiniMaxText01ForCausalLM": _HfExamplesInfo("MiniMaxAI/MiniMax-Text-01",
                                                trust_remote_code=True,
                                                revision="a59aa9cbc53b9fb8742ca4e9e1531b9802b6fdc3"),  # noqa: E501
    "MiniMaxM1ForCausalLM": _HfExamplesInfo("MiniMaxAI/MiniMax-M1-40k",
                                            trust_remote_code=True),
    "MistralForCausalLM": _HfExamplesInfo("mistralai/Mistral-7B-Instruct-v0.1"),
    "MixtralForCausalLM": _HfExamplesInfo("mistralai/Mixtral-8x7B-Instruct-v0.1",  # noqa: E501
                                          {"tiny": "TitanML/tiny-mixtral"}),  # noqa: E501
    "QuantMixtralForCausalLM": _HfExamplesInfo("mistral-community/Mixtral-8x22B-v0.1-AWQ"),  # noqa: E501
    "MptForCausalLM": _HfExamplesInfo("mpt", is_available_online=False),
    "MPTForCausalLM": _HfExamplesInfo("mosaicml/mpt-7b"),
    "NemotronForCausalLM": _HfExamplesInfo("nvidia/Minitron-8B-Base"),
    "NemotronHForCausalLM": _HfExamplesInfo("nvidia/Nemotron-H-8B-Base-8K",
                                            trust_remote_code=True),
    "OlmoForCausalLM": _HfExamplesInfo("allenai/OLMo-1B-hf"),
    "Olmo2ForCausalLM": _HfExamplesInfo("allenai/OLMo-2-0425-1B"),
    "OlmoeForCausalLM": _HfExamplesInfo("allenai/OLMoE-1B-7B-0924-Instruct"),
    "OPTForCausalLM": _HfExamplesInfo("facebook/opt-125m",
                                      {"1b": "facebook/opt-iml-max-1.3b"}),
    "OrionForCausalLM": _HfExamplesInfo("OrionStarAI/Orion-14B-Chat",
                                        trust_remote_code=True),
    "PersimmonForCausalLM": _HfExamplesInfo("adept/persimmon-8b-chat"),
    "PhiForCausalLM": _HfExamplesInfo("microsoft/phi-2"),
    "Phi3ForCausalLM": _HfExamplesInfo("microsoft/Phi-3-mini-4k-instruct"),
    "Phi4FlashForCausalLM": _HfExamplesInfo("microsoft/Phi-4-mini-flash-reasoning", # noqa: E501
                                        trust_remote_code=True,
                                        v0_only=True,
                                        max_model_len=10240),
    "PhiMoEForCausalLM": _HfExamplesInfo("microsoft/Phi-3.5-MoE-instruct",
                                         trust_remote_code=True),
    "Plamo2ForCausalLM": _HfExamplesInfo("pfnet/plamo-2-1b",
                                         max_transformers_version="4.53",
                                         transformers_version_reason="vLLM impl inherits PreTrainedModel and clashes with get_input_embeddings",  # noqa: E501
                                        trust_remote_code=True),
    "QWenLMHeadModel": _HfExamplesInfo("Qwen/Qwen-7B-Chat",
                                       max_transformers_version="4.53",
                                       transformers_version_reason="HF model uses remote code that is not compatible with latest Transformers",  # noqa: E501
                                       trust_remote_code=True),
    "Qwen2ForCausalLM": _HfExamplesInfo("Qwen/Qwen2-0.5B-Instruct",
                                        extras={"2.5": "Qwen/Qwen2.5-0.5B-Instruct"}), # noqa: E501
    "Qwen2MoeForCausalLM": _HfExamplesInfo("Qwen/Qwen1.5-MoE-A2.7B-Chat"),
    "Qwen3ForCausalLM": _HfExamplesInfo("Qwen/Qwen3-8B"),
    "Qwen3MoeForCausalLM": _HfExamplesInfo("Qwen/Qwen3-30B-A3B"),
    "RWForCausalLM": _HfExamplesInfo("tiiuae/falcon-40b"),
    "StableLMEpochForCausalLM": _HfExamplesInfo("stabilityai/stablelm-zephyr-3b"),  # noqa: E501
    "StableLmForCausalLM": _HfExamplesInfo("stabilityai/stablelm-3b-4e1t"),
    "Starcoder2ForCausalLM": _HfExamplesInfo("bigcode/starcoder2-3b"),
    "Step3TextForCausalLM": _HfExamplesInfo("stepfun-ai/step3",
                                            trust_remote_code=True,
                                            is_available_online=False),
    "SolarForCausalLM": _HfExamplesInfo("upstage/solar-pro-preview-instruct",
                                        trust_remote_code=True),
    "TeleChat2ForCausalLM": _HfExamplesInfo("Tele-AI/TeleChat2-3B",
                                            trust_remote_code=True),
    "TeleFLMForCausalLM": _HfExamplesInfo("CofeAI/FLM-2-52B-Instruct-2407",
                                            trust_remote_code=True),
    "XverseForCausalLM": _HfExamplesInfo("xverse/XVERSE-7B-Chat",
                                         tokenizer="meta-llama/Llama-2-7b",
                                         trust_remote_code=True),
    "Zamba2ForCausalLM": _HfExamplesInfo("Zyphra/Zamba2-7B-instruct"),
    "MiMoForCausalLM": _HfExamplesInfo("XiaomiMiMo/MiMo-7B-RL",
                                        trust_remote_code=True),
    "Dots1ForCausalLM": _HfExamplesInfo("rednote-hilab/dots.llm1.inst"),
    # [Encoder-decoder]
    "BartModel": _HfExamplesInfo("facebook/bart-base"),
    "BartForConditionalGeneration": _HfExamplesInfo("facebook/bart-large-cnn"),
}

_EMBEDDING_EXAMPLE_MODELS = {
    # [Text-only]
    "BertModel": _HfExamplesInfo("BAAI/bge-base-en-v1.5", v0_only=True),
    "Gemma2Model": _HfExamplesInfo("BAAI/bge-multilingual-gemma2", v0_only=True),  # noqa: E501
    "GritLM": _HfExamplesInfo("parasail-ai/GritLM-7B-vllm"),
    "GteModel": _HfExamplesInfo("Snowflake/snowflake-arctic-embed-m-v2.0",
                                               trust_remote_code=True),
    "GteNewModel": _HfExamplesInfo("Alibaba-NLP/gte-base-en-v1.5",
                                   trust_remote_code=True,
                                   hf_overrides={"architectures": ["GteNewModel"]}),  # noqa: E501
    "InternLM2ForRewardModel": _HfExamplesInfo("internlm/internlm2-1_8b-reward",
                                               trust_remote_code=True),
    "JambaForSequenceClassification": _HfExamplesInfo("ai21labs/Jamba-tiny-reward-dev"),  # noqa: E501
    "LlamaModel": _HfExamplesInfo("llama", is_available_online=False),
    "MistralModel": _HfExamplesInfo("intfloat/e5-mistral-7b-instruct"),
    "ModernBertModel": _HfExamplesInfo("Alibaba-NLP/gte-modernbert-base",
                                trust_remote_code=True, v0_only=True),
    "NomicBertModel": _HfExamplesInfo("nomic-ai/nomic-embed-text-v2-moe",
                                               trust_remote_code=True, v0_only=True),  # noqa: E501
    "Qwen2Model": _HfExamplesInfo("ssmits/Qwen2-7B-Instruct-embed-base"),
    "Qwen2ForRewardModel": _HfExamplesInfo("Qwen/Qwen2.5-Math-RM-72B",
                                           max_transformers_version="4.53",
                                           transformers_version_reason="HF model uses remote code that is not compatible with latest Transformers"),  # noqa: E501
    "Qwen2ForProcessRewardModel": _HfExamplesInfo("Qwen/Qwen2.5-Math-PRM-7B",
                                                  max_transformers_version="4.53",
                                                  transformers_version_reason="HF model uses remote code that is not compatible with latest Transformers"),  # noqa: E501
    "RobertaModel": _HfExamplesInfo("sentence-transformers/stsb-roberta-base-v2", v0_only=True),  # noqa: E501
    "RobertaForMaskedLM": _HfExamplesInfo("sentence-transformers/all-roberta-large-v1", v0_only=True),  # noqa: E501
    "XLMRobertaModel": _HfExamplesInfo("intfloat/multilingual-e5-small", v0_only=True),  # noqa: E501
    # [Multimodal]
    "LlavaNextForConditionalGeneration": _HfExamplesInfo("royokong/e5-v"),
    "Phi3VForCausalLM": _HfExamplesInfo("TIGER-Lab/VLM2Vec-Full",
                                         trust_remote_code=True),
    "Qwen2VLForConditionalGeneration": _HfExamplesInfo("MrLight/dse-qwen2-2b-mrl-v1"), # noqa: E501
    "PrithviGeoSpatialMAE": _HfExamplesInfo("ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11", # noqa: E501
                                            is_available_online=False),  # noqa: E501
}

_SEQUENCE_CLASSIFICATION_EXAMPLE_MODELS = {
    # [Decoder-only]
    "GPT2ForSequenceClassification": _HfExamplesInfo("nie3e/sentiment-polish-gpt2-small"),  # noqa: E501

    # [Cross-encoder]
    "BertForSequenceClassification": _HfExamplesInfo("cross-encoder/ms-marco-MiniLM-L-6-v2", v0_only=True),  # noqa: E501
    "ModernBertForSequenceClassification": _HfExamplesInfo("Alibaba-NLP/gte-reranker-modernbert-base", v0_only=True), # noqa: E501
    "RobertaForSequenceClassification": _HfExamplesInfo("cross-encoder/quora-roberta-base", v0_only=True),  # noqa: E501
    "XLMRobertaForSequenceClassification": _HfExamplesInfo("BAAI/bge-reranker-v2-m3", v0_only=True),  # noqa: E501
}

_AUTOMATIC_CONVERTED_MODELS = {
    # Use as_seq_cls_model for automatic conversion
    "GemmaForSequenceClassification": _HfExamplesInfo("BAAI/bge-reranker-v2-gemma",  # noqa: E501
                                                      v0_only=True,
                                                      hf_overrides={"architectures": ["GemmaForSequenceClassification"], # noqa: E501
                                                                    "classifier_from_token": ["Yes"],  # noqa: E501
                                                                    "method": "no_post_processing"}),  # noqa: E501
    "LlamaForSequenceClassification": _HfExamplesInfo("Skywork/Skywork-Reward-V2-Llama-3.2-1B"),  # noqa: E501
    "Qwen2ForSequenceClassification": _HfExamplesInfo("jason9693/Qwen2.5-1.5B-apeach"),  # noqa: E501
    "Qwen3ForSequenceClassification": _HfExamplesInfo("tomaarsen/Qwen3-Reranker-0.6B-seq-cls"),  # noqa: E501
}

_MULTIMODAL_EXAMPLE_MODELS = {
    # [Decoder-only]
    "AriaForConditionalGeneration": _HfExamplesInfo("rhymes-ai/Aria"),
    "AyaVisionForConditionalGeneration": _HfExamplesInfo("CohereForAI/aya-vision-8b"), # noqa: E501
    "Blip2ForConditionalGeneration": _HfExamplesInfo("Salesforce/blip2-opt-2.7b",  # noqa: E501
                                                     extras={"6b": "Salesforce/blip2-opt-6.7b"}),  # noqa: E501
    "ChameleonForConditionalGeneration": _HfExamplesInfo("facebook/chameleon-7b"),  # noqa: E501
    "DeepseekVLV2ForCausalLM": _HfExamplesInfo("deepseek-ai/deepseek-vl2-tiny",  # noqa: E501
                                                extras={"fork": "Isotr0py/deepseek-vl2-tiny"},  # noqa: E501
                                                max_transformers_version="4.48",  # noqa: E501
                                                transformers_version_reason="HF model is not compatible.",  # noqa: E501
                                                hf_overrides={"architectures": ["DeepseekVLV2ForCausalLM"]}),  # noqa: E501
    "Emu3ForConditionalGeneration": _HfExamplesInfo("BAAI/Emu3-Chat-hf"),
    "FuyuForCausalLM": _HfExamplesInfo("adept/fuyu-8b"),
    "Gemma3ForConditionalGeneration": _HfExamplesInfo("google/gemma-3-4b-it"),
    "Gemma3nForConditionalGeneration": _HfExamplesInfo("google/gemma-3n-E2B-it",    # noqa: E501
                                        min_transformers_version="4.53"),
    "GraniteSpeechForConditionalGeneration": _HfExamplesInfo("ibm-granite/granite-speech-3.3-2b"),  # noqa: E501
    "GLM4VForCausalLM": _HfExamplesInfo("zai-org/glm-4v-9b",
                                        trust_remote_code=True,
                                        hf_overrides={"architectures": ["GLM4VForCausalLM"]}),  # noqa: E501
    "Glm4vForConditionalGeneration": _HfExamplesInfo("zai-org/GLM-4.1V-9B-Thinking"),  # noqa: E501
    "Glm4vMoeForConditionalGeneration": _HfExamplesInfo("zai-org/GLM-4.5V",
                                          is_available_online=False),   # noqa: E501
    "H2OVLChatModel": _HfExamplesInfo("h2oai/h2ovl-mississippi-800m",
                                      trust_remote_code=True,
                                      extras={"2b": "h2oai/h2ovl-mississippi-2b"},  # noqa: E501
                                      max_transformers_version="4.48",  # noqa: E501
                                      transformers_version_reason="HF model is not compatible."),  # noqa: E501
    "InternVLChatModel": _HfExamplesInfo("OpenGVLab/InternVL2-1B",
                                         extras={"2B": "OpenGVLab/InternVL2-2B",
                                                 "3.0": "OpenGVLab/InternVL3-1B"},  # noqa: E501
                                         trust_remote_code=True),
    "InternS1ForConditionalGeneration": _HfExamplesInfo("internlm/Intern-S1",
                                         trust_remote_code=True),
    "Idefics3ForConditionalGeneration": _HfExamplesInfo("HuggingFaceM4/Idefics3-8B-Llama3",  # noqa: E501
                                                        {"tiny": "HuggingFaceTB/SmolVLM-256M-Instruct"}),  # noqa: E501
    "KeyeForConditionalGeneration": _HfExamplesInfo("Kwai-Keye/Keye-VL-8B-Preview", # noqa: E501
                                                    trust_remote_code=True),
    "KimiVLForConditionalGeneration": _HfExamplesInfo("moonshotai/Kimi-VL-A3B-Instruct",  # noqa: E501
                                                      extras={"thinking": "moonshotai/Kimi-VL-A3B-Thinking"},  # noqa: E501
                                                      trust_remote_code=True),
    "Llama4ForConditionalGeneration": _HfExamplesInfo("meta-llama/Llama-4-Scout-17B-16E-Instruct",   # noqa: E501
                                                      max_model_len=10240,
                                                      extras={"llama-guard-4": "meta-llama/Llama-Guard-4-12B"}, # noqa: E501
                                                      ),
    "LlavaForConditionalGeneration": _HfExamplesInfo("llava-hf/llava-1.5-7b-hf",
                                                     extras={"mistral": "mistral-community/pixtral-12b", # noqa: E501
                                                             "mistral-fp8": "nm-testing/pixtral-12b-FP8-dynamic"}),  # noqa: E501
    "LlavaNextForConditionalGeneration": _HfExamplesInfo("llava-hf/llava-v1.6-mistral-7b-hf"),  # noqa: E501
    "LlavaNextVideoForConditionalGeneration": _HfExamplesInfo("llava-hf/LLaVA-NeXT-Video-7B-hf"),  # noqa: E501
    "LlavaOnevisionForConditionalGeneration": _HfExamplesInfo("llava-hf/llava-onevision-qwen2-0.5b-ov-hf"),  # noqa: E501
    "MantisForConditionalGeneration": _HfExamplesInfo("TIGER-Lab/Mantis-8B-siglip-llama3",  # noqa: E501
                                                      max_transformers_version="4.48",  # noqa: E501
                                                      transformers_version_reason="HF model is not compatible.",  # noqa: E501
                                                      hf_overrides={"architectures": ["MantisForConditionalGeneration"]}),  # noqa: E501
    "MiniCPMO": _HfExamplesInfo("openbmb/MiniCPM-o-2_6",
                                trust_remote_code=True),
    "MiniCPMV": _HfExamplesInfo("openbmb/MiniCPM-Llama3-V-2_5",
                                extras={"2.6": "openbmb/MiniCPM-V-2_6", "4.0": "openbmb/MiniCPM-V-4"},  # noqa: E501
                                trust_remote_code=True),
    "MiniMaxVL01ForConditionalGeneration": _HfExamplesInfo("MiniMaxAI/MiniMax-VL-01", # noqa: E501
                                              trust_remote_code=True,
                                              v0_only=True),
    "Mistral3ForConditionalGeneration": _HfExamplesInfo("mistralai/Mistral-Small-3.1-24B-Instruct-2503",  # noqa: E501
                                                        extras={"fp8": "nm-testing/Mistral-Small-3.1-24B-Instruct-2503-FP8-dynamic"}),  # noqa: E501
    "MolmoForCausalLM": _HfExamplesInfo("allenai/Molmo-7B-D-0924",
                                        max_transformers_version="4.48",
                                        transformers_version_reason="Incorrectly-detected `tensorflow` import.",  # noqa: E501
                                        extras={"olmo": "allenai/Molmo-7B-O-0924"},  # noqa: E501
                                        trust_remote_code=True),
    "NVLM_D": _HfExamplesInfo("nvidia/NVLM-D-72B",
                              trust_remote_code=True),
    "Llama_Nemotron_Nano_VL" : _HfExamplesInfo("nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1", # noqa: E501
                                                     trust_remote_code=True),
    "Ovis": _HfExamplesInfo("AIDC-AI/Ovis2-1B", trust_remote_code=True,
                            extras={"1.6-llama": "AIDC-AI/Ovis1.6-Llama3.2-3B",
                                    "1.6-gemma": "AIDC-AI/Ovis1.6-Gemma2-9B"}),  # noqa: E501
    "PaliGemmaForConditionalGeneration": _HfExamplesInfo("google/paligemma-3b-mix-224",  # noqa: E501
                                                         extras={"v2": "google/paligemma2-3b-ft-docci-448"}),  # noqa: E501
    "Phi3VForCausalLM": _HfExamplesInfo("microsoft/Phi-3-vision-128k-instruct",
                                        trust_remote_code=True,
                                        max_transformers_version="4.48",
                                        transformers_version_reason="Use of deprecated imports which have been removed.",  # noqa: E501
                                        extras={"phi3.5": "microsoft/Phi-3.5-vision-instruct"}),  # noqa: E501
    "Phi4MMForCausalLM": _HfExamplesInfo("microsoft/Phi-4-multimodal-instruct",
                                        trust_remote_code=True),
    "Phi4MultimodalForCausalLM": _HfExamplesInfo("microsoft/Phi-4-multimodal-instruct",  # noqa: E501
                                                 revision="refs/pr/70"),
    "PixtralForConditionalGeneration": _HfExamplesInfo("mistralai/Pixtral-12B-2409",  # noqa: E501
                                                       tokenizer_mode="mistral"),
    "QwenVLForConditionalGeneration": _HfExamplesInfo("Qwen/Qwen-VL",
                                                      extras={"chat": "Qwen/Qwen-VL-Chat"},  # noqa: E501
                                                      trust_remote_code=True,
                                                      hf_overrides={"architectures": ["QwenVLForConditionalGeneration"]}),  # noqa: E501
    "Qwen2AudioForConditionalGeneration": _HfExamplesInfo("Qwen/Qwen2-Audio-7B-Instruct"),  # noqa: E501
    "Qwen2VLForConditionalGeneration": _HfExamplesInfo("Qwen/Qwen2-VL-2B-Instruct"),  # noqa: E501
    "Qwen2_5_VLForConditionalGeneration": _HfExamplesInfo("Qwen/Qwen2.5-VL-3B-Instruct", # noqa: E501
                                                          max_model_len=4096),
    "Qwen2_5OmniModel": _HfExamplesInfo("Qwen/Qwen2.5-Omni-3B"),
    "Qwen2_5OmniForConditionalGeneration": _HfExamplesInfo("Qwen/Qwen2.5-Omni-7B-AWQ"),  # noqa: E501
    "SkyworkR1VChatModel": _HfExamplesInfo("Skywork/Skywork-R1V-38B",
                                           trust_remote_code=True),
    "SmolVLMForConditionalGeneration": _HfExamplesInfo("HuggingFaceTB/SmolVLM2-2.2B-Instruct"),  # noqa: E501
    "Step3VLForConditionalGeneration": _HfExamplesInfo("stepfun-ai/step3",
                                                        trust_remote_code=True,
                                                        is_available_online=False),
    "UltravoxModel": _HfExamplesInfo("fixie-ai/ultravox-v0_5-llama-3_2-1b",  # noqa: E501
                                     trust_remote_code=True),
    "TarsierForConditionalGeneration": _HfExamplesInfo("omni-research/Tarsier-7b"),  # noqa: E501
    "Tarsier2ForConditionalGeneration": _HfExamplesInfo("omni-research/Tarsier2-Recap-7b",  # noqa: E501
                                                        hf_overrides={"architectures": ["Tarsier2ForConditionalGeneration"]}),  # noqa: E501
    "VoxtralForConditionalGeneration": _HfExamplesInfo(
        "mistralai/Voxtral-Mini-3B-2507",
        min_transformers_version="4.54",
        # disable this temporarily until we support HF format
        is_available_online=False,
    ),
    # [Encoder-decoder]
    # Florence-2 uses BartFastTokenizer which can't be loaded from AutoTokenizer
    # Therefore, we borrow the BartTokenizer from the original Bart model
    "Florence2ForConditionalGeneration": _HfExamplesInfo("microsoft/Florence-2-base",  # noqa: E501
                                                         tokenizer="Isotr0py/Florence-2-tokenizer",  # noqa: E501
                                                         trust_remote_code=True),  # noqa: E501
    "MllamaForConditionalGeneration": _HfExamplesInfo("meta-llama/Llama-3.2-11B-Vision-Instruct"),  # noqa: E501
    "WhisperForConditionalGeneration": _HfExamplesInfo("openai/whisper-large-v3"),  # noqa: E501
    # [Cross-encoder]
    "JinaVLForRanking": _HfExamplesInfo("jinaai/jina-reranker-m0"),   # noqa: E501
}


_SPECULATIVE_DECODING_EXAMPLE_MODELS = {
    "MedusaModel": _HfExamplesInfo("JackFram/llama-68m",
                                   speculative_model="abhigoyal/vllm-medusa-llama-68m-random"),  # noqa: E501
    # Temporarily disabled.
    # TODO(woosuk): Re-enable this once the MLP Speculator is supported in V1.
    # "MLPSpeculatorPreTrainedModel": _HfExamplesInfo("JackFram/llama-160m",
    #                                                 speculative_model="ibm-ai-platform/llama-160m-accelerator"),  # noqa: E501
    "DeepSeekMTPModel": _HfExamplesInfo("luccafong/deepseek_mtp_main_random",
                                        speculative_model="luccafong/deepseek_mtp_draft_random",  # noqa: E501
                                        trust_remote_code=True),
    "EagleLlamaForCausalLM": _HfExamplesInfo("yuhuili/EAGLE-LLaMA3-Instruct-8B",
                                             trust_remote_code=True,
                                             speculative_model="yuhuili/EAGLE-LLaMA3-Instruct-8B",
                                             tokenizer="meta-llama/Meta-Llama-3-8B-Instruct"),  # noqa: E501
    "Eagle3LlamaForCausalLM": _HfExamplesInfo("yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",  # noqa: E501
                                            trust_remote_code=True,
                                            speculative_model="yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
                                            tokenizer="meta-llama/Llama-3.1-8B-Instruct"),
    # TODO: Re-enable this once tests/models/test_initialization.py is fixed, see PR #22333 #22611   # noqa: E501
    # "LlamaForCausalLMEagle3": _HfExamplesInfo("AngelSlim/Qwen3-8B_eagle3",  # noqa: E501
    #                                         trust_remote_code=True,
    #                                         speculative_model="AngelSlim/Qwen3-8B_eagle3",   # noqa: E501
    #                                         tokenizer="Qwen/Qwen3-8B"),
    "EagleLlama4ForCausalLM": _HfExamplesInfo(
        "morgendave/EAGLE-Llama-4-Scout-17B-16E-Instruct",
        trust_remote_code=True,
        speculative_model="morgendave/EAGLE-Llama-4-Scout-17B-16E-Instruct",
        tokenizer="meta-llama/Llama-4-Scout-17B-16E-Instruct"),  # noqa: E501
    "EagleMiniCPMForCausalLM": _HfExamplesInfo("openbmb/MiniCPM-1B-sft-bf16",
                                            trust_remote_code=True,
                                            is_available_online=False,
                                            speculative_model="openbmb/MiniCPM-2B-sft-bf16",
                                            tokenizer="openbmb/MiniCPM-2B-sft-bf16"),
    "Glm4MoeMTPModel": _HfExamplesInfo("zai-org/GLM-4.5",
                                        speculative_model="zai-org/GLM-4.5",
                                        min_transformers_version="4.54",
                                        is_available_online=False),
    "MiMoMTPModel": _HfExamplesInfo("XiaomiMiMo/MiMo-7B-RL",
                                    trust_remote_code=True,
                                    speculative_model="XiaomiMiMo/MiMo-7B-RL")
}

_TRANSFORMERS_BACKEND_MODELS = {
    "TransformersModel": _HfExamplesInfo("Qwen/Qwen3-Embedding-0.6B"),
    "TransformersForCausalLM": _HfExamplesInfo("hmellor/Ilama-3.2-1B", trust_remote_code=True),  # noqa: E501
    "TransformersForMultimodalLM": _HfExamplesInfo("OpenGVLab/InternVL3-1B-hf"),
}

_EXAMPLE_MODELS = {
    **_TEXT_GENERATION_EXAMPLE_MODELS,
    **_EMBEDDING_EXAMPLE_MODELS,
    **_SEQUENCE_CLASSIFICATION_EXAMPLE_MODELS,
    **_MULTIMODAL_EXAMPLE_MODELS,
    **_SPECULATIVE_DECODING_EXAMPLE_MODELS,
    **_TRANSFORMERS_BACKEND_MODELS,
}


class HfExampleModels:
    def __init__(self, hf_models: Mapping[str, _HfExamplesInfo]) -> None:
        super().__init__()

        self.hf_models = hf_models

    def get_supported_archs(self) -> Set[str]:
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
AUTO_EXAMPLE_MODELS = HfExampleModels(_AUTOMATIC_CONVERTED_MODELS)
