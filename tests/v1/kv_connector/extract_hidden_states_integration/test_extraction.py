# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import tempfile
from dataclasses import dataclass
from typing import Any

import pytest
import torch

from tests.models.multimodal.generation.test_nemotron_parse_35 import (
    MODEL as NEMOTRON_PARSE_MODEL,
)
from tests.models.multimodal.generation.test_nemotron_parse_35 import (
    PROMPT as NEMOTRON_PARSE_PROMPT,
)
from tests.utils import create_new_process_for_each_test, multi_gpu_test
from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    example_hidden_states_connector,
)
from vllm.inputs import TextPrompt
from vllm.platforms import current_platform


def get_and_check_output(output, expected_shape):
    assert output.kv_transfer_params is not None
    hidden_states_path = output.kv_transfer_params.get("hidden_states_path")
    assert hidden_states_path is not None

    obj = example_hidden_states_connector.load_hidden_states(hidden_states_path)
    token_ids = obj["token_ids"]
    hidden_states = obj["hidden_states"]

    prompt_token_ids = output.prompt_token_ids
    assert torch.equal(token_ids, torch.tensor(prompt_token_ids))

    assert hidden_states.shape == expected_shape

    # Verify hidden_states are not all zeros (i.e., they were actually computed)
    assert not torch.allclose(hidden_states, torch.zeros_like(hidden_states))

    return token_ids, hidden_states


@pytest.fixture
def predictable_llama_config_path(tmp_path_factory):
    """Create a minimal LlamaConfig for PredictableLlamaForCausalLM."""
    from transformers import LlamaConfig, LlamaTokenizerFast

    config_dir = tmp_path_factory.mktemp("predictable_llama")

    # Create a minimal Llama config with small dimensions
    config = LlamaConfig(
        vocab_size=32000,  # Covers the TinyLlama tokenizer below.
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=24,  # Enough layers to test various layer_ids
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=1024,
        architectures=["PredictableLlamaForCausalLM"],
    )

    # Save config
    config.save_pretrained(config_dir)

    # Create a simple tokenizer
    tokenizer = LlamaTokenizerFast.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        cache_dir=os.path.expanduser("~/.cache/huggingface"),
    )
    tokenizer.save_pretrained(config_dir)

    return str(config_dir)


# Need for Python 3.14 compatibility, vLLM with force spawn due to CUDA
# reinitialization, and register will not persist.
PREDICTABLE_LLAMA_MODEL_CLS = (
    "tests.v1.kv_connector.extract_hidden_states_integration"
    ".predictable_llama:PredictableLlamaForCausalLM"
)


@create_new_process_for_each_test()
def test_extract_hidden_states_with_predictable_dummy_model(
    predictable_llama_config_path, tmp_path
):
    """Test hidden-state extraction with a predictable dummy model.

    Tests 3 scenarios:

    1. **Basic extraction**: unsorted layer selection, multiple prompts
       of varying length — verifies correct layer association and
       deterministic values.
    2. **Chunked prefill**: max_num_batched_tokens=128 with ~500-token
       prompts so each is split across multiple scheduler iterations —
       verifies hidden states are reassembled correctly.
    3. **Per-request options**: custom hidden_states_path and
       include_output_tokens — verifies per-request kv_transfer_params
       plumbing.
    """
    layer_ids = [5, 2, 10]
    expected_layer_ids = sorted(layer_ids)
    num_layers = len(layer_ids)
    max_num_batched_tokens = 128

    llm = LLM(
        model=predictable_llama_config_path,
        model_class_overrides={
            "PredictableLlamaForCausalLM": PREDICTABLE_LLAMA_MODEL_CLS,
        },
        speculative_config={
            "method": "extract_hidden_states",
            "num_speculative_tokens": 1,
            "draft_model_config": {
                "hf_config": {"eagle_aux_hidden_state_layer_ids": layer_ids}
            },
        },
        kv_transfer_config={
            "kv_connector": "ExampleHiddenStatesConnector",
            "kv_role": "kv_producer",
            "kv_connector_extra_config": {
                "shared_storage_path": tmp_path,
                "allow_custom_save_path": True,
            },
        },
        max_model_len=1024,
        max_num_batched_tokens=max_num_batched_tokens,
        enforce_eager=True,
        trust_remote_code=True,
        load_format="dummy",
    )

    hidden_size = llm.llm_engine.model_config.get_hidden_size()

    # --- Scenario 1: basic extraction with non-sequential layers ----------
    prompts = [
        "Short",
        "Medium length",
        "Much longer prompt with many tokens",
        "Much longer prompt with many tokens",  # repeated prompt
    ]
    sampling_params = SamplingParams(max_tokens=1, temperature=0.0)
    outputs = llm.generate(prompts, sampling_params)

    assert len(outputs) == len(prompts)
    for output in outputs:
        assert output.prompt_token_ids is not None
        expected_shape = (
            len(output.prompt_token_ids),
            num_layers,
            hidden_size,
        )
        _token_ids, hidden_states = get_and_check_output(output, expected_shape)

        for idx, layer_id in enumerate(expected_layer_ids):
            layer_hidden = hidden_states[:, idx, :]
            assert torch.allclose(
                layer_hidden,
                torch.full_like(layer_hidden, layer_id),
                atol=1e-5,
            ), (
                f"Layer {layer_id} at position {idx} should output "
                f"{float(layer_id)}, but got mean="
                f"{layer_hidden.mean():.3f}, min="
                f"{layer_hidden.min():.3f}, max={layer_hidden.max():.3f}"
            )

    # --- Scenario 2: chunked prefill with long prompts --------------------
    long_prompt = " ".join(["word"] * 500)
    chunked_prompts = [
        long_prompt,
        long_prompt + " extra tokens here",
        "Short",
    ]
    outputs = llm.generate(chunked_prompts, sampling_params)

    assert len(outputs) == len(chunked_prompts)
    for output in outputs:
        assert output.prompt_token_ids is not None
        prompt_len = len(output.prompt_token_ids)
        expected_shape = (prompt_len, num_layers, hidden_size)
        _token_ids, hidden_states = get_and_check_output(output, expected_shape)

        for idx, layer_id in enumerate(expected_layer_ids):
            layer_hidden = hidden_states[:, idx, :]
            assert torch.allclose(
                layer_hidden,
                torch.full_like(layer_hidden, layer_id),
                atol=1e-5,
            ), (
                f"Layer {layer_id} at position {idx} should output "
                f"{float(layer_id)}, but got mean="
                f"{layer_hidden.mean():.3f}, min="
                f"{layer_hidden.min():.3f}, max="
                f"{layer_hidden.max():.3f}. "
                f"prompt_len={prompt_len}, "
                f"max_num_batched_tokens={max_num_batched_tokens}"
            )

    # --- Scenario 3: per-request options ----------------------------------
    max_tokens = 5
    custom_path = os.path.join(tmp_path, "subdir", "custom.safetensors")

    sampling_params_list = [
        SamplingParams(max_tokens=max_tokens, temperature=0.0),
        SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0,
            extra_args={
                "kv_transfer_params": {
                    "hidden_states_path": custom_path,
                    "include_output_tokens": True,
                }
            },
        ),
    ]
    per_req_prompts = ["Short", "Medium length"]
    outputs = llm.generate(per_req_prompts, sampling_params_list)

    # First output: prompt-only hidden states, default path
    out0 = outputs[0]
    assert out0.kv_transfer_params is not None
    assert out0.prompt_token_ids is not None
    path0 = out0.kv_transfer_params["hidden_states_path"]
    assert path0 != custom_path
    obj0 = example_hidden_states_connector.load_hidden_states(path0)
    assert torch.equal(obj0["token_ids"], torch.tensor(out0.prompt_token_ids))
    assert obj0["hidden_states"].shape == (
        len(out0.prompt_token_ids),
        num_layers,
        hidden_size,
    )
    example_hidden_states_connector.cleanup_hidden_states(path0)

    # Second output: prompt + output tokens, custom path
    out1 = outputs[1]
    assert out1.kv_transfer_params is not None
    assert out1.prompt_token_ids is not None
    assert out1.kv_transfer_params["hidden_states_path"] == custom_path
    obj1 = example_hidden_states_connector.load_hidden_states(custom_path)
    token_ids = obj1["token_ids"]
    hidden_states = obj1["hidden_states"]
    # The final output token was never an input to the model, so its hidden
    # state is not in the cache — hence the -1.
    total_tokens = len(out1.prompt_token_ids) + len(out1.outputs[0].token_ids) - 1
    assert token_ids.shape[0] == total_tokens
    assert hidden_states.shape == (total_tokens, num_layers, hidden_size)

    # Verify predictable layer values hold for all tokens (prompt + output)
    for idx, layer_id in enumerate(expected_layer_ids):
        layer_hidden = hidden_states[:, idx, :]
        assert torch.allclose(
            layer_hidden,
            torch.full_like(layer_hidden, layer_id),
            atol=1e-5,
        )
    example_hidden_states_connector.cleanup_hidden_states(custom_path)


@dataclass(frozen=True)
class _VLMExtractionCase:
    model: str
    prompt: str
    images: tuple[tuple[ImageAsset, str], ...]
    layer_ids: tuple[int, ...]
    engine_kwargs: dict[str, Any]
    chunk_size: int | None = None


def _run_vlm_extraction(vllm_runner, tmp_path, case, prompt, *, chunk_size=None):
    engine_kwargs = case.engine_kwargs.copy()
    if chunk_size is not None:
        engine_kwargs.update(
            enable_chunked_prefill=True,
            disable_chunked_mm_input=True,
            max_num_batched_tokens=chunk_size,
        )

    with vllm_runner(
        case.model,
        enforce_eager=True,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.5,
        speculative_config={
            "method": "extract_hidden_states",
            "num_speculative_tokens": 1,
            "draft_model_config": {
                "hf_config": {"eagle_aux_hidden_state_layer_ids": list(case.layer_ids)}
            },
        },
        kv_transfer_config={
            "kv_connector": "ExampleHiddenStatesConnector",
            "kv_role": "kv_producer",
            "kv_connector_extra_config": {"shared_storage_path": str(tmp_path)},
        },
        **engine_kwargs,
    ) as runner:
        (rendered,) = runner.llm.renderer.render_cmpl([prompt], skip_mm_cache=True)
        assert rendered["type"] == "multimodal"
        image_spans = rendered["mm_placeholders"]["image"]
        assert len(image_spans) == len(case.images)
        assert all(span.get_num_embeds() > 100 for span in image_spans)
        prompt_token_ids = rendered["prompt_token_ids"]

        if chunk_size is not None:
            scheduler_config = runner.llm.llm_engine.vllm_config.scheduler_config
            assert scheduler_config.max_num_scheduled_tokens == chunk_size
            first, second = image_spans
            # Each image fits alone, but the first step cannot fit both images.
            assert first.offset + first.length <= chunk_size
            assert second.length <= chunk_size < second.offset + second.length

        (output,) = runner.llm.generate(
            prompt, SamplingParams(max_tokens=1, temperature=0)
        )
        assert output.prompt_token_ids == prompt_token_ids
        hidden_size = runner.llm.model_config.get_hidden_size()
        expected_shape = (len(prompt_token_ids), len(case.layer_ids), hidden_size)
        return get_and_check_output(output, expected_shape)


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            _VLMExtractionCase(
                model=NEMOTRON_PARSE_MODEL,
                prompt=f"<pad>{NEMOTRON_PARSE_PROMPT}",
                images=((ImageAsset("paper-11"), "png"),),
                layer_ids=(2, 5, 9, 10),
                engine_kwargs={"trust_remote_code": True, "max_model_len": 4096},
            ),
            id="nemotron-parse",
        ),
        pytest.param(
            _VLMExtractionCase(
                model="Qwen/Qwen3-VL-2B-Instruct",
                prompt=(
                    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
                    "<|vision_start|><|image_pad|><|vision_end|>"
                    "Describe both images.<|im_end|>\n<|im_start|>assistant\n"
                ),
                images=(
                    (ImageAsset("stop_sign"), "jpg"),
                    (ImageAsset("cherry_blossom"), "jpg"),
                ),
                layer_ids=(0, 2, 5, 9),
                engine_kwargs={
                    "max_model_len": 2048,
                    "max_num_batched_tokens": 2048,
                    "limit_mm_per_prompt": {"image": 2, "video": 0},
                    "mm_processor_kwargs": {"max_pixels": 768 * 1024},
                },
                chunk_size=1024,
            ),
            id="qwen3-vl",
        ),
    ],
)
@create_new_process_for_each_test()
def test_extract_hidden_states_vision_language_model(
    vllm_runner, tmp_path, case: _VLMExtractionCase
):
    images = []
    for asset, ext in case.images:
        with asset.pil_image_ext(ext=ext) as image:
            images.append(image.convert("RGB"))
    prompt = TextPrompt(prompt=case.prompt, multi_modal_data={"image": images})

    token_ids, hidden_states = _run_vlm_extraction(
        vllm_runner, tmp_path / "full", case, prompt
    )
    if case.chunk_size is None:
        return

    chunked_token_ids, chunked_hidden_states = _run_vlm_extraction(
        vllm_runner, tmp_path / "chunked", case, prompt, chunk_size=case.chunk_size
    )
    assert torch.equal(chunked_token_ids, token_ids)
    similarity = torch.nn.functional.cosine_similarity(
        chunked_hidden_states.float(), hidden_states.float(), dim=-1
    )
    torch.testing.assert_close(
        similarity, torch.ones_like(similarity), rtol=0, atol=1e-3
    )


@create_new_process_for_each_test()
def test_extract_hidden_states_qwen35_hybrid_smoke(tmp_path):
    """Smoke test for Qwen3.5 hybrid (mamba + full-attention) models.
    Uses load_format="dummy" to just check shape/plumbing.
    """
    layer_ids = [5, 11, 17]
    hidden_size = 1024  # Qwen/Qwen3.5-0.8B hidden_size

    llm = LLM(
        model="Qwen/Qwen3.5-0.8B",
        speculative_config={
            "method": "extract_hidden_states",
            "num_speculative_tokens": 1,
            "draft_model_config": {
                "hf_config": {"eagle_aux_hidden_state_layer_ids": layer_ids}
            },
        },
        kv_transfer_config={
            "kv_connector": "ExampleHiddenStatesConnector",
            "kv_role": "kv_producer",
            "kv_connector_extra_config": {"shared_storage_path": str(tmp_path)},
        },
        max_model_len=256,
        enforce_eager=True,
        gpu_memory_utilization=0.4,
        load_format="dummy",
    )

    prompts = ["Hello world", "Test prompt with several tokens"]
    sampling_params = SamplingParams(max_tokens=1, temperature=0.0)
    outputs = llm.generate(prompts, sampling_params)

    assert len(outputs) == len(prompts)
    for output in outputs:
        assert output.kv_transfer_params is not None
        hidden_states_path = output.kv_transfer_params.get("hidden_states_path")
        assert hidden_states_path is not None

        obj = example_hidden_states_connector.load_hidden_states(hidden_states_path)
        token_ids = obj["token_ids"]
        hidden_states = obj["hidden_states"]

        assert output.prompt_token_ids is not None
        assert torch.equal(token_ids, torch.tensor(output.prompt_token_ids))
        assert hidden_states.shape == (
            len(output.prompt_token_ids),
            len(layer_ids),
            hidden_size,
        )


@pytest.mark.timeout(
    240 if current_platform.is_rocm() or current_platform.is_xpu() else 120
)
@multi_gpu_test(num_gpus=2)
@create_new_process_for_each_test()
def test_extract_hidden_states_tp2():
    """Test that hidden states extraction works with tensor_parallel_size=2."""
    tmp_dir = tempfile.mkdtemp()
    layer_ids = [5, 11, 17]
    hidden_size = 1024  # Qwen/Qwen3-0.6B hidden_size

    llm = LLM(
        model="Qwen/Qwen3-0.6B",
        tensor_parallel_size=2,
        speculative_config={
            "method": "extract_hidden_states",
            "num_speculative_tokens": 1,
            "draft_model_config": {
                "hf_config": {"eagle_aux_hidden_state_layer_ids": layer_ids}
            },
        },
        kv_transfer_config={
            "kv_connector": "ExampleHiddenStatesConnector",
            "kv_role": "kv_producer",
            "kv_connector_extra_config": {"shared_storage_path": tmp_dir},
        },
        max_model_len=256,
        enforce_eager=True,
        gpu_memory_utilization=0.4,
        load_format="dummy",
    )

    prompts = ["Hello world", "Test prompt with several tokens"]
    sampling_params = SamplingParams(max_tokens=1, temperature=0.0)
    outputs = llm.generate(prompts, sampling_params)

    assert len(outputs) == len(prompts)
    for output in outputs:
        assert output.kv_transfer_params is not None
        hidden_states_path = output.kv_transfer_params.get("hidden_states_path")
        assert hidden_states_path is not None

        obj = example_hidden_states_connector.load_hidden_states(hidden_states_path)
        token_ids = obj["token_ids"]
        hidden_states = obj["hidden_states"]

        assert output.prompt_token_ids is not None
        assert torch.equal(token_ids, torch.tensor(output.prompt_token_ids))
        assert hidden_states.shape == (
            len(output.prompt_token_ids),
            len(layer_ids),
            hidden_size,
        )


@pytest.mark.parametrize("model_impl", ["vllm", "transformers"])
@create_new_process_for_each_test()
def test_extract_layer_zero_is_input_embeddings(tmp_path, model_impl):
    """Layer 0 is the input to the first decoder layer, i.e. the input embeddings."""
    from transformers import AutoModelForCausalLM

    model = "meta-llama/Llama-3.2-1B-Instruct"
    llm = LLM(
        model=model,
        model_impl=model_impl,
        speculative_config={
            "method": "extract_hidden_states",
            "num_speculative_tokens": 1,
            "draft_model_config": {
                "hf_config": {"eagle_aux_hidden_state_layer_ids": [0, 1]}
            },
        },
        kv_transfer_config={
            "kv_connector": "ExampleHiddenStatesConnector",
            "kv_role": "kv_producer",
            "kv_connector_extra_config": {"shared_storage_path": str(tmp_path)},
        },
        max_model_len=256,
        gpu_memory_utilization=0.4,
    )
    hidden_size = llm.llm_engine.model_config.get_hidden_size()
    sampling_params = SamplingParams(max_tokens=1, temperature=0.0)
    output = llm.generate(["Hello world"], sampling_params)[0]
    expected_shape = (len(output.prompt_token_ids), 2, hidden_size)
    token_ids, hidden_states = get_and_check_output(output, expected_shape)

    hf_model = AutoModelForCausalLM.from_pretrained(model, dtype=hidden_states.dtype)
    with torch.no_grad():
        embeddings = hf_model.get_input_embeddings()(token_ids)
    torch.testing.assert_close(hidden_states[:, 0, :], embeddings)
