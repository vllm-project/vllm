# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from dataclasses import dataclass, field
from functools import partial

import pytest
from mistral_common.protocol.instruct.chunk import ImageChunk, TextChunk
from mistral_common.protocol.instruct.messages import AssistantMessage, UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest

from vllm.multimodal.video import sample_frames_from_video
from vllm.platforms import current_platform
from vllm.tokenizers.mistral import MistralTokenizer

from ....conftest import IMAGE_ASSETS, VIDEO_ASSETS
from ...utils import dummy_hf_overrides
from .vlm_utils.builders import sample_frames_with_video_metadata


@dataclass(slots=True)
class VitCudagraphTestConfig:
    model: str
    modalities: list[str] = field(default_factory=lambda: ["image", "video"])
    image_prompt: str | None = None
    video_prompt: str | None = None
    dtype: str = "bfloat16"
    max_model_len: int = 4096
    max_tokens: int = 64
    max_num_seqs: int = 2
    num_video_frames: int = 16
    needs_video_metadata: bool = False
    vllm_runner_kwargs: dict = field(default_factory=dict)
    compilation_config_overrides: dict = field(default_factory=dict)
    # Mistral-format models need the image token grid inserted by the
    # tokenizer's chat encoding, so `image_prompt` is used as the text chunk.
    mistral_chat_prompt: bool = False
    marks: list = field(default_factory=list)
    skip: bool = False


def params_with_marks(
    configs: dict[str, VitCudagraphTestConfig],
) -> list[object]:
    return [
        pytest.param(model_id, marks=cfg.marks) for model_id, cfg in configs.items()
    ]


def qwen_vl_chat_template(content: str) -> str:
    return f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"


def internvl_chat_template(content: str) -> str:
    return f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"


def kimi_vl_chat_template(content: str) -> str:
    return (
        f"<|im_user|>user<|im_middle|>{content}<|im_end|>"
        "<|im_assistant|>assistant<|im_middle|>"
    )


def deepseek_v41_chat_template(content: str) -> str:
    return f"<｜begin▁of▁sentence｜><｜User｜>{content}<｜Assistant｜></think>"


def step3_vl_chat_template(content: str) -> str:
    return (
        "<｜begin▁of▁sentence｜> You are a helpful assistant.<|BOT|>user\n "
        f"<im_patch>{content} <|EOT|><|BOT|>assistant\n"
    )


def gemma3_chat_template(content: str) -> str:
    return f"<bos><start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n"


def deepseek_vl2_chat_template(content: str) -> str:
    return f"<|User|>: {content}\n\n<|Assistant|>:"


def ernie45_vl_chat_template(content: str) -> str:
    return (
        f"<|begin_of_sentence|>User: {content}"
        "Picture 1:<|IMAGE_START|><|image@placeholder|><|IMAGE_END|>\n"
        "Assistant: <think></think>"
    )


def minicpmv_25_chat_template(content: str) -> str:
    """Llama3-style chat template used by MiniCPM-V 2.5."""
    return (
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{content}"
        f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def minicpmv_chat_template(content: str) -> str:
    """ChatML template used by MiniCPM-V 2.6 / 4.0 / 4.5."""
    return f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"


def lfm3_vl_chat_template(content: str) -> str:
    """ChatML template used by LFM3-VL."""
    return (
        f"<|startoftext|><|im_start|>user\n{content}<|im_end|>\n"
        "<|im_start|>assistant\n<think></think>"
    )


def _lfm3_vl_dummy_hf_overrides(hf_config):
    """Shrink LFM3-VL for dummy tests while keeping conv and attention layers."""
    # `dummy_hf_overrides` sets num_hidden_layers=1 and slices vision
    # layer_types[:1] (`["conv"]`), so restore both conv and full_attention.
    hf_config = dummy_hf_overrides(
        hf_config,
        model_arch="Lfm3VlForConditionalGeneration",
        num_hidden_layers=3,
    )
    if hasattr(hf_config, "vision_config"):
        hf_config.vision_config.update(
            {
                "num_layers": 2,
                "num_hidden_layers": 2,
                "layer_types": ["conv", "full_attention"],
            }
        )
    return hf_config


MODEL_CONFIGS: dict[str, VitCudagraphTestConfig] = {
    "pixtral": VitCudagraphTestConfig(
        model="mistralai/Ministral-3-3B-Instruct-2512",
        modalities=["image"],
        image_prompt="What is in this image?",
        compilation_config_overrides={
            "encoder_cudagraph_token_budgets": [1024, 4096],
        },
        vllm_runner_kwargs={
            "load_format": "dummy",
            "tokenizer_mode": "mistral",
            "config_format": "mistral",
        },
        mistral_chat_prompt=True,
        marks=[pytest.mark.core_model],
    ),
    "mistral3": VitCudagraphTestConfig(
        model="mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        modalities=["image"],
        image_prompt="<s>[INST][IMG]What is in this image?[/INST]",
        compilation_config_overrides={
            "encoder_cudagraph_token_budgets": [1024, 4096],
        },
        vllm_runner_kwargs={
            "load_format": "dummy",
            # The repo also ships a Mistral-format config, which resolves to
            # PixtralForConditionalGeneration.
            "config_format": "hf",
            "tokenizer_mode": "hf",
            "hf_overrides": partial(
                dummy_hf_overrides,
                model_arch="Mistral3ForConditionalGeneration",
            ),
        },
        marks=[pytest.mark.core_model],
    ),
    "gemma3": VitCudagraphTestConfig(
        model="google/gemma-3-4b-it",
        modalities=["image"],
        image_prompt=gemma3_chat_template("<start_of_image>What is in this image?"),
        compilation_config_overrides={
            "encoder_cudagraph_token_budgets": [512],
        },
        dtype="bfloat16",
        max_model_len=4096,
    ),
    "llama4": VitCudagraphTestConfig(
        model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        modalities=["image"],
        image_prompt=(
            "<|begin_of_text|><|header_start|>user<|header_end|>\n\n"
            "<|image|>What is in this image?<|eot|>"
            "<|header_start|>assistant<|header_end|>\n\n"
        ),
        max_model_len=4096,
        max_tokens=32,
        max_num_seqs=2,
        vllm_runner_kwargs={
            "load_format": "dummy",
            "hf_overrides": partial(
                dummy_hf_overrides,
                model_arch="Llama4ForConditionalGeneration",
            ),
        },
        marks=[pytest.mark.core_model],
    ),
    "qwen2_vl": VitCudagraphTestConfig(
        model="Qwen/Qwen2-VL-2B-Instruct",
        image_prompt=qwen_vl_chat_template(
            "<|vision_start|><|image_pad|><|vision_end|>What is in this image?"
        ),
        video_prompt=qwen_vl_chat_template(
            "<|vision_start|><|video_pad|><|vision_end|>"
            "Describe this video in one sentence."
        ),
        needs_video_metadata=False,
        marks=[pytest.mark.core_model],
    ),
    "qwen2_5_vl": VitCudagraphTestConfig(
        model="Qwen/Qwen2.5-VL-3B-Instruct",
        image_prompt=qwen_vl_chat_template(
            "<|vision_start|><|image_pad|><|vision_end|>What is in this image?"
        ),
        video_prompt=qwen_vl_chat_template(
            "<|vision_start|><|video_pad|><|vision_end|>"
            "Describe this video in one sentence."
        ),
        needs_video_metadata=True,
        marks=[pytest.mark.core_model],
    ),
    "kimi_vl": VitCudagraphTestConfig(
        model="moonshotai/Kimi-VL-A3B-Instruct",
        modalities=["image"],
        image_prompt=kimi_vl_chat_template(
            "<|media_start|>image<|media_content|><|media_pad|><|media_end|>"
            "What is in this image?"
        ),
        needs_video_metadata=False,
        # Single bucket sized to cover the test images' output tokens.
        # The default auto-inferred range fans out into multiple power-of-2
        # buckets, each holding a full ViT capture pool.
        compilation_config_overrides={
            "encoder_cudagraph_token_budgets": [1024],
        },
        # Shrink to 1 text + 1 vision layer with random weights so the
        # test runs on any CI GPU (incl. L4) and skips the multi-GiB
        # weight download. The test only validates that encoder CG
        # capture/replay functions correctly, not output quality.
        vllm_runner_kwargs={
            "trust_remote_code": True,
            "load_format": "dummy",
            "hf_overrides": partial(
                dummy_hf_overrides,
                model_arch="KimiVLForConditionalGeneration",
            ),
        },
        marks=[pytest.mark.core_model],
    ),
    "kimi_k3": VitCudagraphTestConfig(
        model="moonshotai/Kimi-K3",
        modalities=["image"],
        image_prompt="<|kimi_image_placeholder|>What is in this image?",
        compilation_config_overrides={
            "encoder_cudagraph_token_budgets": [512],
        },
        # 1 text + 1 vision layer with random weights, as for kimi_vl.
        vllm_runner_kwargs={
            "trust_remote_code": True,
            "enable_chunked_prefill": True,
            "load_format": "dummy",
            "hf_overrides": partial(
                dummy_hf_overrides,
                model_arch="KimiK3ForConditionalGeneration",
            ),
        },
        marks=[pytest.mark.core_model],
    ),
    "minimax_m3": VitCudagraphTestConfig(
        model="MiniMaxAI/MiniMax-M3",
        modalities=["image"],
        image_prompt=("]~b]user\n]<]image[>[What is in this image?[e~[\n]~b]ai\n"),
        compilation_config_overrides={
            "encoder_cudagraph_token_budgets": [512],
        },
        vllm_runner_kwargs={
            "trust_remote_code": True,
            "load_format": "dummy",
            "hf_overrides": partial(
                dummy_hf_overrides,
                model_arch="MiniMaxM3SparseForConditionalGeneration",
            ),
            "mm_processor_kwargs": {"max_long_side_pixel": 336},
        },
        marks=[pytest.mark.core_model],
    ),
    "qwen3_vl": VitCudagraphTestConfig(
        model="Qwen/Qwen3-VL-2B-Instruct",
        image_prompt=qwen_vl_chat_template(
            "<|vision_start|><|image_pad|><|vision_end|>What is in this image?"
        ),
        video_prompt=qwen_vl_chat_template(
            "<|vision_start|><|video_pad|><|vision_end|>"
            "Describe this video in one sentence."
        ),
        needs_video_metadata=True,
        marks=[pytest.mark.core_model],
    ),
    "qwen3_5": VitCudagraphTestConfig(
        model="Qwen/Qwen3.5-0.8B",
        image_prompt=qwen_vl_chat_template(
            "<|vision_start|><|image_pad|><|vision_end|>What is in this image?"
        ),
        video_prompt=qwen_vl_chat_template(
            "<|vision_start|><|video_pad|><|vision_end|>"
            "Describe this video in one sentence."
        ),
        needs_video_metadata=True,
        vllm_runner_kwargs={"enable_chunked_prefill": True},
        marks=[pytest.mark.core_model],
    ),
    "internvl": VitCudagraphTestConfig(
        model="OpenGVLab/InternVL3-1B",
        num_video_frames=8,
        image_prompt=internvl_chat_template("<image>\nWhat is in this image?"),
        video_prompt=internvl_chat_template(
            "<video>\nDescribe this video in one sentence."
        ),
        needs_video_metadata=False,
        vllm_runner_kwargs={"trust_remote_code": True},
        marks=[pytest.mark.core_model],
    ),
    "idefics3": VitCudagraphTestConfig(
        model="HuggingFaceTB/SmolVLM-256M-Instruct",
        modalities=["image"],
        image_prompt=(
            "<|begin_of_text|>User:<image>What is in this image?"
            "<end_of_utterance>\nAssistant:"
        ),
        max_model_len=4096,
        compilation_config_overrides={
            "encoder_cudagraph_token_budgets": [4096],
        },
        vllm_runner_kwargs={"gpu_memory_utilization": 0.80},
        marks=[pytest.mark.core_model],
    ),
    "deepseek_v41": VitCudagraphTestConfig(
        model="deepseek-ai/DeepSeek-V4.1-Flash",
        modalities=["image"],
        image_prompt=deepseek_v41_chat_template(
            "<｜deepseek_image｜>\n\nWhat is in this image?"
        ),
        compilation_config_overrides={
            "encoder_cudagraph_token_budgets": [1024],
        },
        vllm_runner_kwargs={
            "load_format": "dummy",
            "attention_backend": (
                "ROCM_FLASHMLA_SPARSE_DSV41"
                if current_platform.is_rocm()
                else "FLASHMLA_SPARSE_DSV41"
            ),
            "hf_overrides": partial(
                dummy_hf_overrides,
                model_arch="DeepseekV41ForCausalLM",
                exist_overrides={"vision_n_layers": 1},
            ),
        },
    ),
    "step3_vl": VitCudagraphTestConfig(
        model="stepfun-ai/Step3-VL-10B",
        modalities=["image"],
        image_prompt=step3_vl_chat_template("What is in this image?"),
        # Single bucket sized to cover the largest test image's output
        # tokens (1152 > 1141 for cherry_blossom). The default auto-
        # inferred range fans out into multiple power-of-2 buckets, each
        # holding a full ViT capture pool.
        compilation_config_overrides={
            "encoder_cudagraph_token_budgets": [1152],
        },
        # Shrink to 1 text + 1 vision layer with random weights so the
        # test runs on any CI GPU (incl. L4) and skips the 20 GiB weight
        # download. The test only validates that encoder CG capture/
        # replay functions correctly, not output quality.
        vllm_runner_kwargs={
            "trust_remote_code": True,
            "load_format": "dummy",
            "hf_overrides": partial(
                dummy_hf_overrides,
                model_arch="StepVLForConditionalGeneration",
            ),
        },
    ),
    "ernie45_vl": VitCudagraphTestConfig(
        model="baidu/ERNIE-4.5-VL-28B-A3B-PT",
        # Image only: Ernie's resampler applies a temporal conv for video
        # that changes the output token count, so video uses the eager path.
        modalities=["image"],
        image_prompt=ernie45_vl_chat_template("What is in this image?"),
        # Ernie4_5_VLMoeModel is deliberately not torch-compiled, since its
        # split of text and vision experts breaks compilation, so piecewise
        # cudagraphs have nothing to partition. Only the encoder graphs this
        # test covers are captured.
        compilation_config_overrides={
            "cudagraph_mode": 2,
        },
        # Shrink to 1 text + 1 vision layer with random weights so the test
        # runs on any CI GPU and skips the ~56 GiB weight download. The test
        # only validates encoder CG capture/replay, not output quality.
        vllm_runner_kwargs={
            "load_format": "dummy",
            "trust_remote_code": True,
            "revision": "refs/pr/17",
            "hf_overrides": partial(
                dummy_hf_overrides,
                model_arch="Ernie4_5_VLMoeForConditionalGeneration",
            ),
        },
    ),
    "glm4_1v": VitCudagraphTestConfig(
        model="zai-org/GLM-4.1V-9B-Thinking",
        image_prompt=(
            "[gMASK]<sop><|system|>\nYou are a helpful assistant.<|user|>\n"
            "<|begin_of_image|><|image|><|end_of_image|>"
            "What is in this image?<|assistant|>assistant\n"
        ),
        video_prompt=(
            "[gMASK]<sop><|system|>\nYou are a helpful assistant.<|user|>\n"
            "<|begin_of_video|><|video|><|end_of_video|>"
            "Describe this video in one sentence<|assistant|>assistant\n"
        ),
        needs_video_metadata=True,
        marks=[pytest.mark.core_model],
        vllm_runner_kwargs={
            "load_format": "dummy",
            "hf_overrides": partial(
                dummy_hf_overrides,
                model_arch="Glm4vForConditionalGeneration",
            ),
        },
    ),
    "deepseek_ocr": VitCudagraphTestConfig(
        model="deepseek-ai/DeepSeek-OCR",
        modalities=["image"],
        image_prompt="<image>\nWhat is in this image?",
        marks=[pytest.mark.core_model],
        compilation_config_overrides={
            "encoder_cudagraph_token_budgets": [272],
            "mode": 0,
            "cudagraph_mode": 2,
        },
        vllm_runner_kwargs={
            "load_format": "dummy",
            "hf_overrides": partial(
                dummy_hf_overrides,
                model_arch="DeepseekOCRForCausalLM",
            ),
        },
        skip=True,  # TODO: Re-enable this once OOM issues are resolved on CI.
    ),
    "deepseek_ocr2": VitCudagraphTestConfig(
        model=os.environ.get("DEEPSEEK_OCR2_TEST_MODEL", "deepseek-ai/DeepSeek-OCR-2"),
        modalities=["image"],
        image_prompt="<image>\nFree OCR.",
        marks=[pytest.mark.core_model],
        max_model_len=2048,
        compilation_config_overrides={
            "encoder_cudagraph_token_budgets": [288],
            "mode": 0,
            "cudagraph_mode": 0,
        },
        vllm_runner_kwargs={
            "load_format": "dummy",
            "gpu_memory_utilization": 0.7,
        },
    ),
    "deepseek_vl2": VitCudagraphTestConfig(
        model="deepseek-ai/deepseek-vl2-tiny",
        modalities=["image"],
        image_prompt=deepseek_vl2_chat_template("<image>\nWhat is in this image?"),
        vllm_runner_kwargs={
            "hf_overrides": {"architectures": ["DeepseekVLV2ForCausalLM"]},
        },
        marks=[pytest.mark.core_model],
    ),
    "gemma4": VitCudagraphTestConfig(
        model="google/gemma-4-E2B-it",
        image_prompt=(
            "<bos><start_of_turn>user\n<|image|>\nWhat is in this image?<end_of_turn>\n"
            "<start_of_turn>model\n"
        ),
        video_prompt=(
            "<bos><start_of_turn>user\n<|video|>\nDescribe this video in one sentence."
            "<end_of_turn>\n<start_of_turn>model\n"
        ),
        # The 16-frame test video produces 1056 vision tokens. Capture only
        # the smallest supported bucket that covers it instead of all default
        # buckets through max_model_len, which adds unrelated memory pressure.
        compilation_config_overrides={
            "encoder_cudagraph_token_budgets": [1120],
        },
        needs_video_metadata=True,
        marks=[pytest.mark.core_model],
    ),
    "minicpmv_25": VitCudagraphTestConfig(
        model="openbmb/MiniCPM-Llama3-V-2_5",
        modalities=["image"],
        image_prompt=minicpmv_25_chat_template(
            "(<image>./</image>)\nWhat is in this image?"
        ),
        # CI runs on 35GB MIG slices: every budget captures one graph per
        # patch-grid bucket, so the default budgets through max_model_len
        # OOM there. A small budget set covers the test images (each item
        # is at most (max_slice_num + 1) * query_num = 640 tokens).
        compilation_config_overrides={
            "encoder_cudagraph_token_budgets": [64, 1024],
        },
        vllm_runner_kwargs={"trust_remote_code": True},
        marks=[pytest.mark.core_model],
    ),
    "minicpmv_26": VitCudagraphTestConfig(
        model="openbmb/MiniCPM-V-2_6",
        image_prompt=minicpmv_chat_template(
            "(<image>./</image>)\nWhat is in this image?"
        ),
        video_prompt=minicpmv_chat_template(
            "(<video>./</video>)\nDescribe this video in one sentence."
        ),
        max_model_len=2048,
        # Fewer frames keep the video item within the 1024 budget.
        num_video_frames=4,
        compilation_config_overrides={
            "encoder_cudagraph_token_budgets": [64, 1024],
            "encoder_cudagraph_max_frames_per_batch": 4,
        },
        vllm_runner_kwargs={"trust_remote_code": True},
        marks=[pytest.mark.core_model],
    ),
    "minicpmv_40": VitCudagraphTestConfig(
        model="openbmb/MiniCPM-V-4",
        image_prompt=minicpmv_chat_template(
            "(<image>./</image>)\nWhat is in this image?"
        ),
        video_prompt=minicpmv_chat_template(
            "(<video>./</video>)\nDescribe this video in one sentence."
        ),
        max_model_len=2048,
        num_video_frames=4,
        compilation_config_overrides={
            "encoder_cudagraph_token_budgets": [64, 1024],
            "encoder_cudagraph_max_frames_per_batch": 4,
        },
        vllm_runner_kwargs={"trust_remote_code": True},
        marks=[pytest.mark.core_model],
    ),
    "lfm3_vl": VitCudagraphTestConfig(
        model="LiquidAI/LFM3-VL-2B",
        image_prompt=lfm3_vl_chat_template(
            "<|vision_start|><|image|><|vision_end|>What is in this image?"
        ),
        video_prompt=lfm3_vl_chat_template(
            "<|vision_start|><|video|><|vision_end|>"
            "Describe this video in one sentence."
        ),
        needs_video_metadata=True,
        compilation_config_overrides={
            "encoder_cudagraph_token_budgets": [1152, 2048],
        },
        vllm_runner_kwargs={
            "load_format": "dummy",
            "hf_overrides": _lfm3_vl_dummy_hf_overrides,
            "enable_chunked_prefill": True,
            "gpu_memory_utilization": 0.80,
        },
        marks=[pytest.mark.core_model],
    ),
}


def get_compilation_config(config: VitCudagraphTestConfig):
    return {
        "cudagraph_mm_encoder": True,
        "encoder_cudagraph_max_vision_items_per_batch": 1,
        "encoder_cudagraph_max_frames_per_batch": 16,
        **config.compilation_config_overrides,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_id", params_with_marks(MODEL_CONFIGS))
@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="Skip if not cuda or rocm"
)
def test_vit_cudagraph_image(model_id, vllm_runner, image_assets):
    config = MODEL_CONFIGS[model_id]

    if config.skip:
        pytest.skip(f"{model_id} is marked to be skipped.")

    if "image" not in config.modalities:
        pytest.skip(f"{model_id} does not support the image modality.")

    images = [[asset.pil_image] for asset in image_assets]

    with vllm_runner(
        config.model,
        dtype=config.dtype,
        max_model_len=config.max_model_len,
        max_num_seqs=config.max_num_seqs,
        limit_mm_per_prompt={"image": 1},
        compilation_config=get_compilation_config(config),
        **config.vllm_runner_kwargs,
    ) as vllm_model:
        if config.mistral_chat_prompt:
            assert config.image_prompt is not None
            tokenizer = vllm_model.llm.get_tokenizer()
            assert isinstance(tokenizer, MistralTokenizer)
            image_prompts = [
                tokenizer.mistral.encode_chat_completion(
                    ChatCompletionRequest(
                        messages=[
                            UserMessage(
                                content=[
                                    ImageChunk(image=asset.pil_image),
                                    TextChunk(text=config.image_prompt),
                                ]
                            ),
                            # Dummy weights tend to repeat the last prompt token;
                            # end on text rather than the [/INST] control token
                            # so the output does not detokenize to "".
                            AssistantMessage(content="The image shows", prefix=True),
                        ]
                    )
                ).tokens
                for asset in image_assets
            ]
        else:
            image_prompts = IMAGE_ASSETS.prompts(
                {
                    "stop_sign": config.image_prompt,  # type: ignore[typeddict-item]
                    "cherry_blossom": config.image_prompt,  # type: ignore[typeddict-item]
                }
            )
        outputs = vllm_model.generate_greedy(
            image_prompts, config.max_tokens, images=images
        )

        # Basic validation that we got a response
        assert len(outputs) == 2
        output_ids, output_text = outputs[0]

        # Ensure we got some output
        assert len(output_ids) > 0
        assert len(output_text) > 0

        # Ensure the output is a string
        assert isinstance(output_text, str)


@pytest.mark.parametrize("model_id", params_with_marks(MODEL_CONFIGS))
@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="Skip if not cuda or rocm"
)
def test_vit_cudagraph_video(model_id, vllm_runner, video_assets):
    config = MODEL_CONFIGS[model_id]

    if config.skip:
        pytest.skip(f"{model_id} is marked to be skipped.")

    if "video" not in config.modalities:
        pytest.skip(f"{model_id} does not support the video modality")

    video_prompts = VIDEO_ASSETS.prompts(
        {
            "baby_reading": config.video_prompt,  # type: ignore[typeddict-item]
        }
    )
    sampled_vids: list
    if config.needs_video_metadata:
        sampled_vids = [
            sample_frames_with_video_metadata(
                (asset.np_ndarrays, asset.metadata), config.num_video_frames
            )
            for asset in video_assets
        ]
    else:
        sampled_vids = [
            sample_frames_from_video(asset.np_ndarrays, config.num_video_frames)
            for asset in video_assets
        ]
    videos = [sampled_vids[0]]

    with vllm_runner(
        config.model,
        dtype=config.dtype,
        max_model_len=config.max_model_len,
        max_num_seqs=config.max_num_seqs,
        limit_mm_per_prompt={"video": 1},
        compilation_config=get_compilation_config(config),
        **config.vllm_runner_kwargs,
    ) as vllm_model:
        outputs = vllm_model.generate_greedy(
            video_prompts, config.max_tokens, videos=videos
        )

        # Basic validation that we got a response
        assert len(outputs) == 1
        output_ids, output_text = outputs[0]

        # Ensure we got some output
        assert len(output_ids) > 0
        assert len(output_text) > 0

        # Ensure the output is a string
        assert isinstance(output_text, str)


def _check_eonly_encoder_outputs(worker, batches):
    """Compare real E-only replay with eager, retaining outputs across replay."""
    import torch

    from vllm.v1.worker.mm_encoder_model_runner import MMEncoderModelRunner

    runner = worker.model_runner
    assert isinstance(runner, MMEncoderModelRunner)
    manager = runner.model_state.encoder_runner.cudagraph_manager
    assert manager is not None and manager.is_captured()
    retained: list[tuple[torch.Tensor, torch.Tensor]] = []
    max_error = 0.0
    stats = []
    with torch.inference_mode():
        for batch in batches:
            kwargs = {key: value.to(runner.device) for key, value in batch.items()}
            actual = manager.execute(kwargs)
            expected = runner.model.embed_multimodal(**kwargs)
            assert len(actual) == len(expected)
            for output, reference in zip(actual, expected):
                assert torch.isfinite(output).all()
                # Two BF16 ulps at unit scale, fixed before model validation.
                torch.testing.assert_close(output, reference, rtol=0.016, atol=0.016)
                max_error = max(max_error, (output - reference).abs().max().item())
            for output, snapshot in retained:
                torch.testing.assert_close(output, snapshot, rtol=0, atol=0)
            retained.extend((output, output.clone()) for output in actual)
            stats.append(manager.get_cumulative_stats())
    assert stats[0]["graph_hits"] > 0
    assert stats[1]["graph_hits"] > stats[0]["graph_hits"]
    assert stats[2]["graph_misses"] > stats[1]["graph_misses"]
    assert stats[3]["graph_hits"] > stats[2]["graph_hits"]
    assert stats[4]["graph_hits"] > stats[3]["graph_hits"]
    assert stats[5]["graph_hits"] > stats[4]["graph_hits"]
    assert stats[6]["graph_hits"] > stats[5]["graph_hits"]
    return {"max_abs_error": max_error, "stats": stats}


def _check_deepseek_ocr2_encoder_outputs(worker, batches):
    import torch

    from vllm.v1.worker.mm_encoder_model_runner import MMEncoderModelRunner

    runner = worker.model_runner
    assert isinstance(runner, MMEncoderModelRunner)
    manager = runner.model_state.encoder_runner.cudagraph_manager
    assert manager is not None and manager.is_captured()
    assert manager.path_token_budgets == {
        "global": [256],
        "local": [0, 144, 288],
    }

    hits = [manager.get_cumulative_stats()["graph_hits"]]
    with torch.inference_mode():
        for batch in batches:
            kwargs = {
                key: value.to(
                    device=runner.device,
                    dtype=runner.dtype
                    if key in ("pixel_values", "images_crop")
                    else value.dtype,
                )
                for key, value in batch.items()
            }
            actual = manager.execute(kwargs)
            reference = runner.model.embed_multimodal(**kwargs)
            assert len(actual) == len(reference) == 1
            assert actual[0].shape == reference[0].shape
            torch.testing.assert_close(actual[0], reference[0], rtol=0.016, atol=0.016)
            hits.append(manager.get_cumulative_stats()["graph_hits"])

    assert [b - a for a, b in zip(hits[:-1], hits[1:])] == [1, 2, 1, 2]


@pytest.mark.core_model
@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA graphs")
def test_deepseek_ocr2_encoder_cudagraph_paths(vllm_runner, monkeypatch):
    """Match eager embeddings for untiled, tiled, and local-fallback images."""
    from PIL import Image
    from transformers import AutoTokenizer

    from vllm.transformers_utils.processors.deepseek_ocr import DeepseekOCRProcessor

    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    model_name = os.environ.get(
        "DEEPSEEK_OCR2_TEST_MODEL", "deepseek-ai/DeepSeek-OCR-2"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
    processor = DeepseekOCRProcessor(
        tokenizer=tokenizer, image_size=768, base_size=1024, strategy="v2"
    )
    sizes = [(600, 600), (1200, 600), (1200, 1200), (1200, 600)]
    batches = []
    for i, size in enumerate(sizes):
        image = Image.new("RGB", size, color=(20 + i * 40, 80, 160))
        processed = processor(prompt="<image>\nFree OCR.", images=[image])
        batches.append(
            {
                key: processed[key]
                for key in ("pixel_values", "images_crop", "images_spatial_crop")
            }
        )
    batches[1]["images_crop"].zero_()

    with vllm_runner(
        model_name,
        mm_encoder_only=True,
        dtype="bfloat16",
        max_model_len=4096,
        max_num_seqs=1,
        limit_mm_per_prompt={"image": 1},
        compilation_config={
            "cudagraph_mode": "NONE",
            "cudagraph_mm_encoder": True,
            "encoder_cudagraph_token_budgets": [288],
            "encoder_cudagraph_max_vision_items_per_batch": 1,
        },
    ) as model:
        model.llm.collective_rpc(
            partial(_check_deepseek_ocr2_encoder_outputs, batches=batches)
        )


@pytest.mark.parametrize(
    "model_id",
    params_with_marks(
        {key: MODEL_CONFIGS[key] for key in ("qwen3_vl", "qwen3_5", "glm4_1v")}
    ),
)
@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA graphs")
def test_eonly_vit_cudagraph_outputs(
    model_id, vllm_runner, image_assets, video_assets, monkeypatch
):
    """Exercise startup capture, image/video replay, fallback and output ownership."""
    import numpy as np
    from PIL import Image
    from transformers import AutoProcessor

    # collective_rpc below sends a `partial` function, which needs pickle
    # fallback since msgpack can't serialize it.
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    config = MODEL_CONFIGS[model_id]
    token_budgets = [64, 256, 2048]
    processor = AutoProcessor.from_pretrained(config.model)
    first, second = [asset.pil_image for asset in image_assets]
    batches = [
        [first.resize((224, 224))],
        [second.resize((448, 224)), first.resize((224, 224))],
        [second.resize((1792, 1792))],
        [second.resize((224, 224))],
        [first.resize((1280, 720))],
        [first.resize((224, 224))],
    ]
    frames = sample_frames_from_video(video_assets[0].np_ndarrays, 2)
    video = np.stack(
        [np.asarray(Image.fromarray(frame).resize((224, 224))) for frame in frames]
    )
    with vllm_runner(
        config.model,
        dtype=config.dtype,
        mm_encoder_only=True,
        # FA padding NaNs are tracked separately in #57136.
        mm_encoder_attn_backend="FLASHINFER",
        enable_prefix_caching=False,
        max_model_len=4096,
        max_num_seqs=2,
        limit_mm_per_prompt={"image": 2, "video": 1},
        compilation_config={
            "cudagraph_mode": "NONE",
            "cudagraph_mm_encoder": True,
            "encoder_cudagraph_token_budgets": token_budgets,
            "encoder_cudagraph_max_vision_items_per_batch": 2,
            "encoder_cudagraph_max_frames_per_batch": 2,
        },
    ) as model:
        mm_config = model.llm.llm_engine.vllm_config.model_config.multimodal_config
        processor_kwargs = mm_config.merge_mm_processor_kwargs({})
        inputs = [
            dict(
                processor.image_processor(
                    images=images, return_tensors="pt", **processor_kwargs
                )
            )
            for images in batches
        ]
        inputs.append(
            dict(
                processor.video_processor(
                    videos=[video],
                    do_sample_frames=False,
                    return_tensors="pt",
                    **processor_kwargs,
                )
            )
        )
        results = model.llm.collective_rpc(
            partial(_check_eonly_encoder_outputs, batches=inputs)
        )
        for result in results:
            assert result["stats"][0]["token_budgets"] == token_budgets
        print(f"E-only {model_id}: {results}")


def _check_k3_replay_and_threshold(worker, grids, hits, misses):
    """Run one image batch through the real manager; compare with eager.

    Args:
        grids: grid_thw of each image in the batch.
        hits: expected graph hits for the batch.
        misses: expected eager fallbacks for the batch.

    """
    import torch

    from vllm.utils.torch_utils import set_random_seed

    runner = worker.model_runner
    manager = (
        runner.model_state.encoder_runner.cudagraph_manager
        if hasattr(runner, "model_state")
        else runner.encoder_cudagraph_manager
    )
    assert manager is not None and manager.is_captured()
    model = runner.model
    patch = runner.model_config.hf_config.vision_config.patch_size
    set_random_seed(0)
    with torch.inference_mode():
        num_patches = sum(t * h * w for t, h, w in grids)
        kwargs = {
            # The input norm takes raw 8-bit pixels.
            "pixel_values": torch.randint(
                0,
                256,
                (num_patches, 3, patch, patch),
                device=runner.device,
                dtype=torch.uint8,
            ),
            "grid_thws": grids,
        }
        before = manager.get_cumulative_stats()
        actual = torch.cat(manager.execute(kwargs))
        after = manager.get_cumulative_stats()
        assert after["graph_hits"] - before["graph_hits"] == hits
        assert after["graph_misses"] - before["graph_misses"] == misses
        expected = model.encoder_eager_forward(kwargs)
        assert actual.shape == expected.shape
        assert torch.isfinite(actual).all() and expected.abs().max() > 0
        # Two bf16 ulps at the output scale: replay runs padded GEMMs.
        bound = 2 * expected.abs().max().item() * 2**-7
        diff = (actual.float() - expected.float()).abs().max().item()
        assert diff <= bound, f"max abs diff {diff} > {bound}"
    return manager.token_budgets


@pytest.mark.core_model
@pytest.mark.skipif(not current_platform.is_rocm(), reason="AMD Kimi-K3 encoder graphs")
@pytest.mark.parametrize(
    ("grids", "hits", "misses"),
    [
        pytest.param([[1, 16, 16]], 1, 0, id="fills_smallest_budget_64"),
        pytest.param([[1, 28, 28]], 1, 0, id="padded_budget_196"),
        pytest.param([[1, 32, 64]], 1, 0, id="fills_largest_budget_512"),
        pytest.param([[1, 34, 64]], 0, 1, id="above_largest_budget_544"),
        pytest.param([[1, 28, 28]] * 2, 2, 0, id="two_images_share_replay"),
        pytest.param(
            [[1, 28, 28], [1, 34, 64]], 0, 2, id="batch_over_budget_all_eager"
        ),
    ],
)
def test_kimi_k3_vit_cudagraph_replay_and_threshold(
    grids, hits, misses, vllm_runner, monkeypatch
):
    """Replay matches eager; a batch whose total output tokens fit the largest
    budget (512 by default) replays once, and a larger batch runs eagerly."""
    # collective_rpc below sends a `partial` function, which needs pickle
    # fallback since msgpack can't serialize it.
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    config = MODEL_CONFIGS["kimi_k3"]
    compilation_config = get_compilation_config(config)
    compilation_config.pop("encoder_cudagraph_token_budgets")
    compilation_config["encoder_cudagraph_max_vision_items_per_batch"] = 2
    with vllm_runner(
        config.model,
        dtype=config.dtype,
        max_model_len=config.max_model_len,
        max_num_seqs=config.max_num_seqs,
        limit_mm_per_prompt={"image": 1},
        compilation_config=compilation_config,
        **config.vllm_runner_kwargs,
    ) as vllm_model:
        results = vllm_model.llm.collective_rpc(
            partial(
                _check_k3_replay_and_threshold,
                grids=grids,
                hits=hits,
                misses=misses,
            )
        )
    for token_budgets in results:
        assert token_budgets[-1] == 512


@pytest.mark.core_model
@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="Skip if not cuda or rocm"
)
def test_lfm3_vl_conv2d_gather_isolates_frames_and_tail_padding(dist_init):
    """Verify `conv_indices` gather kernels match eager and ignore NaN tails."""
    import torch

    from vllm.model_executor.models.lfm3_vl import (
        Lfm3VlVisionConv2d,
        Lfm3VlVisionModel,
        _native_conv2d_gather,
        _triton_conv2d_gather,
    )
    from vllm.transformers_utils.configs.lfm3_vl import Lfm3VlVisionConfig
    from vllm.triton_utils import HAS_TRITON
    from vllm.utils.torch_utils import set_default_torch_dtype

    with torch.inference_mode():
        torch.manual_seed(0)
        device = torch.device(current_platform.device_type, 0)
        cfg = Lfm3VlVisionConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            layer_types=["conv", "full_attention"],
        )
        with set_default_torch_dtype(torch.bfloat16):
            vision = Lfm3VlVisionModel(cfg).to(device).eval()
        conv = vision.layers[0].conv
        assert isinstance(conv, Lfm3VlVisionConv2d)
        for param in conv.parameters():
            param.normal_(std=0.05)

        grid_thw = [[1, 16, 32], [2, 8, 8]]
        frame_shapes = [(h, w) for t, h, w in grid_thw for _ in range(t)]
        n_valid = sum(h * w for h, w in frame_shapes)
        n_buf = n_valid + 128

        x = torch.randn(n_valid, cfg.hidden_size, device=device, dtype=torch.bfloat16)
        eager = conv(x, frame_shapes=frame_shapes)

        conv_indices = vision._build_conv_indices(grid_thw, device=device)
        padded_idx = torch.zeros(
            n_buf, cfg.conv_kernel_size**2, device=device, dtype=torch.int64
        )
        padded_idx[:n_valid].copy_(conv_indices)

        padded_x = torch.full(
            (n_buf, cfg.hidden_size),
            float("nan"),
            device=device,
            dtype=torch.bfloat16,
        )
        padded_x[:n_valid].copy_(x)

        # Exercise both Triton and PyTorch fallback paths directly under NaN tails.
        bcx, _ = conv.in_proj(x)
        b_gate, c_gate, x_in = bcx.chunk(3, dim=-1)
        padded_gated = torch.full(
            (n_buf, b_gate.shape[1]),
            float("nan"),
            device=device,
            dtype=torch.bfloat16,
        )
        padded_gated[:n_valid].copy_(b_gate * x_in)

        native_conv = _native_conv2d_gather(
            padded_gated, conv.spatial_conv.weight, padded_idx
        )
        native_out, _ = conv.out_proj(c_gate * native_conv[:n_valid])
        assert torch.isfinite(native_out).all()
        torch.testing.assert_close(native_out, eager, rtol=1e-3, atol=1e-3)

        if HAS_TRITON:
            triton_conv = _triton_conv2d_gather(
                padded_gated, conv.spatial_conv.weight, padded_idx
            )
            triton_out, _ = conv.out_proj(c_gate * triton_conv[:n_valid])
            assert torch.isfinite(triton_out).all()
            torch.testing.assert_close(triton_out, eager, rtol=1e-3, atol=1e-3)

        graph_out = conv(padded_x, conv_indices=padded_idx)[:n_valid]
        assert torch.isfinite(graph_out).all()
        torch.testing.assert_close(graph_out, eager, rtol=1e-3, atol=1e-3)

        swapped_grid = [[1, 32, 16], [2, 8, 8]]
        padded_idx[:n_valid].copy_(
            vision._build_conv_indices(swapped_grid, device=device)
        )
        swapped_eager = conv(x, frame_shapes=[(32, 16), (8, 8), (8, 8)])
        swapped_out = conv(padded_x, conv_indices=padded_idx)[:n_valid]
        torch.testing.assert_close(swapped_out, swapped_eager, rtol=1e-3, atol=1e-3)
        assert not torch.allclose(swapped_out, eager, rtol=1e-2, atol=1e-2)
