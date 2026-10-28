# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.utils import (
    get_attn_backend_list_based_on_platform,
    single_gpu_only,
)
from vllm import SamplingParams
from vllm.platforms import current_platform

from .utils import _run_eagle_correctness


@single_gpu_only
@pytest.mark.skipif(
    current_platform.is_device_capability_family(100),
    reason="DeepSeek head_dim=192 not supported on SM100/SM110 (Blackwell)",
)
@pytest.mark.parametrize(
    [
        "model_setup",
        "mm_enabled",
        "enable_chunked_prefill",
        "model_impl",
        "expected_accuracy_threshold",
    ],
    [
        (
            (
                "eagle",
                "eagle618/deepseek-v3-random",
                "eagle618/eagle-deepseek-v3-random",
                1,
            ),
            False,
            False,
            "auto",
            0.0,
        ),
    ],
    ids=["deepseek_eagle"],
)
@pytest.mark.parametrize("attn_backend", get_attn_backend_list_based_on_platform())
def test_eagle_correctness_light(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    model_setup: tuple[str, str, str, int],
    mm_enabled: bool,
    expected_accuracy_threshold: float,
    enable_chunked_prefill: bool,
    model_impl: str,
    attn_backend: str,
    vllm_runner,
):
    _run_eagle_correctness(
        monkeypatch,
        sampling_config,
        model_setup,
        mm_enabled,
        expected_accuracy_threshold,
        enable_chunked_prefill,
        model_impl,
        attn_backend,
        vllm_runner,
    )


@single_gpu_only
@pytest.mark.parametrize(
    [
        "model_setup",
        "mm_enabled",
        "enable_chunked_prefill",
        "model_impl",
        "expected_accuracy_threshold",
    ],
    [
        (
            ("eagle3", "Qwen/Qwen3-8B", "AngelSlim/Qwen3-8B_eagle3", 1),
            False,
            False,
            "auto",
            0.8,
        ),
        pytest.param(
            ("eagle3", "Qwen/Qwen3-8B", "AngelSlim/Qwen3-8B_eagle3", 1),
            False,
            False,
            "transformers",
            0.8,
            # TODO(hmellor): figure out why memory usage is so high
            marks=pytest.mark.skipif(
                not current_platform.is_rocm(),
                reason="Feature is experimental and uses too much memory in CUDA CI",
            ),
        ),
        (
            (
                "eagle3",
                "Qwen/Qwen3-VL-8B-Instruct",
                "taobao-mnn/Qwen3-VL-8B-Instruct-Eagle3",
                1,
            ),
            False,
            False,
            "auto",
            0.8,
        ),
        pytest.param(
            (
                "eagle3",
                "Qwen/Qwen2.5-VL-7B-Instruct",
                "Rayzl/qwen2.5-vl-7b-eagle3-sgl",
                1,
            ),
            False,
            False,
            "auto",
            0.7,
            # TODO: Re-measure the reference threshold when re-enabling this case.
            # Text-only mode starts but loses target correctness with ROCm/TRITON;
            # full multimodal profiling currently fails during engine startup.
            marks=pytest.mark.skip(
                reason="Qwen2.5-VL Eagle3 is not yet reliable in this test setup"
            ),
        ),
        (
            (
                "eagle3",
                "meta-llama/Llama-3.1-8B-Instruct",
                "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
                1,
            ),
            False,
            False,
            "auto",
            0.7,
        ),
    ],
    ids=[
        "qwen3_eagle3",
        "qwen3_eagle3-transformers",
        "qwen3_vl_eagle3",
        "qwen2_5_vl_eagle3",
        "llama3_eagle3",
    ],
)
@pytest.mark.parametrize("attn_backend", get_attn_backend_list_based_on_platform())
def test_eagle_correctness_medium(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    model_setup: tuple[str, str, str, int],
    mm_enabled: bool,
    expected_accuracy_threshold: float,
    enable_chunked_prefill: bool,
    model_impl: str,
    attn_backend: str,
    vllm_runner,
):
    _run_eagle_correctness(
        monkeypatch,
        sampling_config,
        model_setup,
        mm_enabled,
        expected_accuracy_threshold,
        enable_chunked_prefill,
        model_impl,
        attn_backend,
        vllm_runner,
    )


@pytest.mark.parametrize("use_v2", ["0", "1"])
def test_eagle_narrow_vocabulary_preserves_target_distribution(
    tmp_path, monkeypatch, vllm_runner, use_v2: str
):
    """Loaded EAGLE weights must preserve target-only mass during verification."""
    import torch
    from safetensors.torch import save_file
    from transformers import LlamaConfig, LlamaForCausalLM

    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", use_v2)
    vocab_size, draft_vocab_size, hidden_size = 8201, 8193, 64
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=256,
        tie_word_embeddings=False,
    )
    target = LlamaForCausalLM(config)
    support = [0, 1, vocab_size - 1]
    probabilities = [0.5, 0.3, 0.2]
    with torch.no_grad():
        for parameter in target.parameters():
            parameter.zero_()
        target.model.embed_tokens.weight.fill_(1)
        target.model.norm.weight.fill_(1)
        target.lm_head.weight.fill_(-1)
        for token, probability in zip(support, probabilities):
            target.lm_head.weight[token].fill_(
                torch.log(torch.tensor(probability)).item() / hidden_size
            )
    target_path, draft_path = tmp_path / "target", tmp_path / "draft"
    target.save_pretrained(target_path)
    config.num_hidden_layers = 1
    config.vocab_size = draft_vocab_size
    config.architectures = ["EagleLlamaForCausalLM"]
    config.save_pretrained(draft_path)
    weights = {
        name.removeprefix("model."): value.contiguous()
        for name, value in target.state_dict().items()
        if name.startswith("model.layers.0.") and "input_layernorm" not in name
    }
    # The EAGLE head shares target embeddings/lm_head, but truncates its logits.
    weights["fc.weight"] = torch.cat(
        [torch.eye(hidden_size), torch.zeros(hidden_size, hidden_size)], dim=1
    )
    save_file(weights, draft_path / "model.safetensors")

    with vllm_runner(
        str(target_path),
        skip_tokenizer_init=True,
        dtype="float32",
        enforce_eager=True,
        max_model_len=128,
        max_num_seqs=32,
        gpu_memory_utilization=0.45,
        enable_prefix_caching=False,
        attention_config={"backend": "TRITON_ATTN"},
        speculative_config={
            "method": "eagle",
            "model": str(draft_path),
            "num_speculative_tokens": 3,
            "draft_sample_method": "probabilistic",
        },
    ) as runner:
        outputs = runner.llm.generate(
            [{"prompt_token_ids": [0, 1, 0]} for _ in range(512)],
            [
                SamplingParams(temperature=1, max_tokens=16, ignore_eos=True, seed=i)
                for i in range(512)
            ],
        )
    # Exclude the first token, which is sampled before speculative verification.
    tokens = torch.tensor([t for o in outputs for t in o.outputs[0].token_ids[1:]])
    assert tokens.numel() == 512 * 15
    assert torch.isin(tokens, torch.tensor(support)).all()
    observed = torch.stack([(tokens == token).float().mean() for token in support])
    torch.testing.assert_close(
        observed, torch.tensor(probabilities), atol=0.035, rtol=0
    )
