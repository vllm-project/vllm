# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import weakref

import pytest

from tests.utils import create_new_process_for_each_test
from vllm import LLM, SamplingParams
from vllm.exceptions import VLLMValidationError

MODEL_NAME = "distilbert/distilgpt2"

PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

TOKEN_IDS = [
    [0],
    [0, 1],
    [0, 2, 1],
    [0, 3, 1, 2],
]


@pytest.fixture(scope="module")
def llm(vllm_runner):
    with vllm_runner(
        MODEL_NAME,
        max_num_batched_tokens=4096,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.10,
        enforce_eager=True,
    ) as runner:
        # pytest caches yielded fixtures until after teardown, so use a proxy to
        # avoid retaining the LLM while VllmRunner.__exit__ releases ROCm memory.
        yield weakref.proxy(runner.llm)


@pytest.mark.skip_global_cleanup
def test_multiple_sampling_params(llm: LLM):
    sampling_params = [
        SamplingParams(temperature=0.01, top_p=0.95),
        SamplingParams(temperature=0.3, top_p=0.95),
        SamplingParams(temperature=0.7, top_p=0.95),
        SamplingParams(temperature=0.99, top_p=0.95),
    ]

    # Multiple SamplingParams should be matched with each prompt
    outputs = llm.generate(PROMPTS, sampling_params=sampling_params)
    assert len(PROMPTS) == len(outputs)

    # Exception raised, if the size of params does not match the size of prompts
    with pytest.raises(VLLMValidationError):
        outputs = llm.generate(PROMPTS, sampling_params=sampling_params[:3])

    # Single SamplingParams should be applied to every prompt
    single_sampling_params = SamplingParams(temperature=0.3, top_p=0.95)
    outputs = llm.generate(PROMPTS, sampling_params=single_sampling_params)
    assert len(PROMPTS) == len(outputs)

    # sampling_params is None, default params should be applied
    outputs = llm.generate(PROMPTS, sampling_params=None)
    assert len(PROMPTS) == len(outputs)


def test_multiple_priority(llm: LLM):
    # Generate works when priority is None
    outputs = llm.generate(PROMPTS, sampling_params=None, priority=None)
    assert len(PROMPTS) == len(outputs)

    # Generate works when length of priority is same as the len(PROMPTS)
    outputs = llm.generate(PROMPTS, sampling_params=None, priority=[0] * len(PROMPTS))
    assert len(PROMPTS) == len(outputs)

    # Exception raised, if the length of priority does not match the length of prompts
    with pytest.raises(VLLMValidationError):
        outputs = llm.generate(
            PROMPTS, sampling_params=None, priority=[0] * (len(PROMPTS) - 1)
        )

    # Exception raised, if the priority list is empty
    with pytest.raises(VLLMValidationError):
        outputs = llm.generate(PROMPTS, sampling_params=None, priority=[])


def test_single_prompt_priority(llm: LLM):
    # Single string prompts should be normalized to one request.
    outputs = llm.generate(PROMPTS[0], sampling_params=None, priority=[0])
    assert len(outputs) == 1


def test_max_model_len(vllm_runner):
    max_model_len = 20
    with vllm_runner(
        MODEL_NAME,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.10,
        enforce_eager=True,  # reduce test time
    ) as runner:
        sampling_params = SamplingParams(max_tokens=max_model_len + 10)
        outputs = runner.llm.generate(PROMPTS, sampling_params)
        for output in outputs:
            num_total_tokens = len(output.prompt_token_ids) + len(
                output.outputs[0].token_ids
            )
            # Total tokens must not exceed max_model_len.
            # It can be less if generation finishes due to other reasons (e.g., EOS)
            # before reaching the absolute model length limit.
            assert num_total_tokens <= max_model_len


def test_log_stats(vllm_runner):
    with vllm_runner(
        MODEL_NAME,
        disable_log_stats=False,
        gpu_memory_utilization=0.10,
        enforce_eager=True,  # reduce test time
    ) as runner:
        outputs = runner.llm.generate(PROMPTS, sampling_params=None)

        # disable_log_stats is False, every output should have metrics
        assert all(output.metrics is not None for output in outputs)


@create_new_process_for_each_test()
def test_inline_qwen35_mixed_generation_and_prefix_hits(tmp_path, monkeypatch):
    """One real model serves long generation and isolated final-position vectors."""
    import torch
    from transformers import AutoConfig

    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    model = "Qwen/Qwen3.5-4B"
    revision = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    config = AutoConfig.from_pretrained(model, revision=revision).get_text_config()
    llm = LLM(
        model=model,
        revision=revision,
        tokenizer_revision=revision,
        max_model_len=1024,
        max_num_batched_tokens=128,
        gpu_memory_utilization=0.3,
        enable_prefix_caching=True,
        enforce_eager=True,
    )
    ordinary = SamplingParams(max_tokens=32, temperature=0, ignore_eos=True)
    inline = SamplingParams(
        max_tokens=1,
        temperature=0,
        extra_args={
            "kv_transfer_params": {"return_inline": True},
        },
    )
    prompts = ["A forest is", "A forest is " + "full of trees " * 200]
    before = llm.generate(prompts[0], ordinary)[0]
    mixed = llm.generate([prompts[0], *prompts], [ordinary, inline, inline])
    repeated = llm.generate(prompts, inline)
    after = llm.generate(prompts[0], ordinary)[0]
    assert repeated[1].num_cached_tokens is not None
    assert repeated[1].num_cached_tokens > 0
    for normal in (before, mixed[0], after):
        assert normal.kv_transfer_params is None
        assert len(normal.outputs[0].token_ids) == 32
        assert normal.outputs[0].token_ids == before.outputs[0].token_ids
    for first, cached in zip(mixed[1:], repeated):
        payload = first.kv_transfer_params
        assert payload["representation"] == "post_final_norm"
        assert payload["layer_id"] == config.num_hidden_layers
        assert payload["token_position"] == len(first.prompt_token_ids) - 1
        vector = torch.tensor(payload["hidden_states"])
        assert vector.shape == (config.hidden_size,)
        assert torch.isfinite(vector).all()
        # Prefix reuse should retain the same representation, allowing bf16
        # arithmetic to differ with batching (not a reference-model parity test).
        torch.testing.assert_close(
            vector,
            torch.tensor(cached.kv_transfer_params["hidden_states"]),
            atol=0.125,
            rtol=0.02,
        )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("with_image", [False, True])
@create_new_process_for_each_test()
def test_inline_qwen35_against_reference_final_hidden_state(with_image, monkeypatch):
    """Compare the returned final prompt vector with Transformers output."""
    import gc

    import torch
    from PIL import Image, ImageDraw
    from transformers import AutoModelForImageTextToText, AutoProcessor

    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    model_id = "Qwen/Qwen3.5-4B"
    revision = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    processor = AutoProcessor.from_pretrained(model_id, revision=revision)
    image = Image.new("RGB", (256, 256), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((24, 48, 112, 208), fill="blue")
    draw.ellipse((136, 80, 232, 176), fill="orange")
    content = [{"type": "text", "text": "Describe the shapes and colors."}]
    if with_image:
        content.insert(0, {"type": "image", "image": image})
    else:
        content.append(
            {
                "type": "text",
                "text": (
                    "A blue rectangle and an orange circle on a white background."
                ),
            }
        )
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = processor(
        text=[prompt], images=[image] if with_image else None, return_tensors="pt"
    ).to("cuda")
    reference = (
        AutoModelForImageTextToText.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .eval()
        .to("cuda")
    )
    with torch.inference_mode():
        ref = reference(**inputs, output_hidden_states=True, use_cache=False)
    normalized = ref.hidden_states[-1][0, -1].float().cpu()
    next_token = ref.logits[0, -1].argmax().item()
    ids = inputs.input_ids[0].tolist()
    del reference, ref, inputs
    gc.collect()
    torch.accelerator.empty_cache()

    llm = LLM(
        model=model_id,
        revision=revision,
        tokenizer_revision=revision,
        dtype="bfloat16",
        max_model_len=1024,
        enforce_eager=True,
    )
    engine_prompt = {"prompt": prompt}
    if with_image:
        engine_prompt["multi_modal_data"] = {"image": image}
    result = llm.generate(
        engine_prompt,
        SamplingParams(
            max_tokens=1,
            temperature=0,
            extra_args={"kv_transfer_params": {"return_inline": True}},
        ),
    )[0]
    assert result.prompt_token_ids == ids
    assert result.outputs[0].token_ids == [next_token]
    vector = torch.tensor(result.kv_transfer_params["hidden_states"])
    # Fixed gates for bf16 backend differences: test magnitudes as well as
    # direction. These are acceptance thresholds, not measured error claims.
    assert torch.isfinite(vector).all()
    relative_rms = (vector - normalized).norm() / normalized.norm()
    scaled_max = (vector - normalized).abs().max() / normalized.abs().max()
    cosine = torch.nn.functional.cosine_similarity(vector, normalized, dim=0)
    assert relative_rms < 0.01, f"relative RMS error: {relative_rms}"
    assert scaled_max < 0.05, f"scaled maximum error: {scaled_max}"
    assert cosine > 0.999, f"cosine: {cosine}"
