# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field
from functools import partial

import pytest

from vllm.multimodal.video import sample_frames_from_video
from vllm.platforms import current_platform

from ....conftest import IMAGE_ASSETS, VIDEO_ASSETS
from ...utils import dummy_hf_overrides
from .vlm_utils.builders import sample_frames_with_video_metadata


@dataclass
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
    marks: list = field(default_factory=list)
    skip: bool = False


def params_with_marks(
    configs: dict[str, VitCudagraphTestConfig],
) -> list[pytest.param]:
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


MODEL_CONFIGS: dict[str, VitCudagraphTestConfig] = {
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
        needs_video_metadata=False,
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
                "ROCM_FLASHMLA_SPARSE_DSV4"
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

    image_prompts = IMAGE_ASSETS.prompts(
        {
            "stop_sign": config.image_prompt,  # type: ignore[typeddict-item]
            "cherry_blossom": config.image_prompt,  # type: ignore[typeddict-item]
        }
    )
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


@pytest.mark.parametrize(
    "model_id",
    params_with_marks({key: MODEL_CONFIGS[key] for key in ("qwen3_vl", "qwen3_5")}),
)
@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA graphs")
def test_eonly_vit_cudagraph_outputs(
    model_id, vllm_runner, image_assets, video_assets, monkeypatch
):
    """Exercise startup capture, image/video replay, fallback and output ownership."""
    import numpy as np
    from PIL import Image
    from transformers import AutoProcessor

    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
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
