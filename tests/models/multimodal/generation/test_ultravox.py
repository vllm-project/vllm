# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from typing import Any

import numpy as np
import pytest
import pytest_asyncio
from transformers import AutoTokenizer

from ....conftest import AUDIO_ASSETS, AudioTestAssets, VllmRunner
from ....utils import RemoteOpenAIServer
from ...registry import HF_EXAMPLE_MODELS

MODEL_NAME = "fixie-ai/ultravox-v0_5-llama-3_2-1b"

AUDIO_PROMPTS = AUDIO_ASSETS.prompts(
    {
        "mary_had_lamb": "Transcribe this into English.",
        "winning_call": "What is happening in this audio clip?",
    }
)

MULTI_AUDIO_PROMPT = "Describe each of the audios above."

AudioTuple = tuple[np.ndarray, int]

VLLM_PLACEHOLDER = "<|audio|>"
HF_PLACEHOLDER = "<|audio|>"

CHUNKED_PREFILL_KWARGS = {
    "enable_chunked_prefill": True,
    "max_num_seqs": 2,
    # Use a very small limit to exercise chunked prefill.
    "max_num_batched_tokens": 16,
}


def params_kwargs_to_cli_args(params_kwargs: dict[str, Any]) -> list[str]:
    """Convert kwargs to CLI args."""
    args = []
    for key, value in params_kwargs.items():
        if isinstance(value, bool):
            if value:
                args.append(f"--{key.replace('_', '-')}")
        else:
            args.append(f"--{key.replace('_', '-')}={value}")
    return args


@pytest.fixture(
    params=[
        pytest.param({}, marks=pytest.mark.cpu_model),
        pytest.param(CHUNKED_PREFILL_KWARGS),
    ]
)
def server(request, audio_assets: AudioTestAssets):
    args = [
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "4096",
        "--enforce-eager",
        "--limit-mm-per-prompt",
        json.dumps({"audio": len(audio_assets)}),
        "--trust-remote-code",
    ] + params_kwargs_to_cli_args(request.param)

    with RemoteOpenAIServer(
        MODEL_NAME, args, env_dict={"VLLM_AUDIO_FETCH_TIMEOUT": "30"}
    ) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


def _get_prompt(audio_count, question, placeholder):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    placeholder = f"{placeholder}\n" * audio_count

    return tokenizer.apply_chat_template(
        [{"role": "user", "content": f"{placeholder}{question}"}],
        tokenize=False,
        add_generation_prompt=True,
    )


def run_multi_audio_test(
    vllm_runner: type[VllmRunner],
    prompts_and_audios: list[tuple[str, list[AudioTuple]]],
    model: str,
    *,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
    **kwargs,
):
    model_info = HF_EXAMPLE_MODELS.find_hf_info(model)
    model_info.check_available_online(on_fail="skip")
    model_info.check_transformers_version(on_fail="skip")

    with vllm_runner(
        model,
        dtype=dtype,
        enforce_eager=True,
        limit_mm_per_prompt={
            "audio": max((len(audio) for _, audio in prompts_and_audios))
        },
        **kwargs,
    ) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy_logprobs(
            [prompt for prompt, _ in prompts_and_audios],
            max_tokens,
            num_logprobs=num_logprobs,
            audios=[audios for _, audios in prompts_and_audios],
        )

    # The HuggingFace model doesn't support multiple audios yet, so
    # just assert that some tokens were generated.
    assert all(tokens for tokens, *_ in vllm_outputs)


@pytest.mark.core_model
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [128])
@pytest.mark.parametrize("num_logprobs", [5])
@pytest.mark.parametrize(
    "vllm_kwargs",
    [
        pytest.param({}, marks=pytest.mark.cpu_model),
        pytest.param(CHUNKED_PREFILL_KWARGS),
    ],
)
def test_models_with_multiple_audios(
    vllm_runner,
    audio_assets: AudioTestAssets,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
    vllm_kwargs: dict,
) -> None:
    vllm_prompt = _get_prompt(len(audio_assets), MULTI_AUDIO_PROMPT, VLLM_PLACEHOLDER)
    run_multi_audio_test(
        vllm_runner,
        [(vllm_prompt, [audio.audio_and_sample_rate for audio in audio_assets])],
        MODEL_NAME,
        dtype=dtype,
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
        **vllm_kwargs,
    )


@pytest.mark.core_model
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [32])
def test_variable_length_audio_batching(
    vllm_runner,
    audio_assets: AudioTestAssets,
    dtype: str,
    max_tokens: int,
) -> None:
    """Test batching of requests with different audio durations.

    This exercises the variable-length tensor handling in
    MultiModalFlatField._reduce_data() which was buggy before
    https://github.com/vllm-project/vllm/issues/31658 was fixed.
    """
    model_info = HF_EXAMPLE_MODELS.find_hf_info(MODEL_NAME)
    model_info.check_available_online(on_fail="skip")
    model_info.check_transformers_version(on_fail="skip")

    # Create prompts with single audio each (different durations)
    prompts_and_audios = []
    for audio, question in zip(audio_assets, AUDIO_PROMPTS):
        prompt = _get_prompt(1, question, VLLM_PLACEHOLDER)
        prompts_and_audios.append((prompt, [audio.audio_and_sample_rate]))

    with vllm_runner(
        MODEL_NAME,
        dtype=dtype,
        enforce_eager=True,
        limit_mm_per_prompt={"audio": 1},
    ) as vllm_model:
        # Generate for all prompts in a single batch
        # This triggers the variable-length batching code path
        outputs = vllm_model.generate_greedy(
            [prompt for prompt, _ in prompts_and_audios],
            max_tokens,
            audios=[audios for _, audios in prompts_and_audios],
        )

    # Verify outputs were generated for each request
    assert len(outputs) == len(prompts_and_audios)
    for output in outputs:
        assert len(output[1]) > 0, "Expected non-empty output"


@pytest.mark.asyncio
async def test_online_serving(client, audio_assets: AudioTestAssets):
    """Exercises online serving with/without chunked prefill enabled."""

    messages = [
        {
            "role": "user",
            "content": [
                *[
                    {"type": "audio_url", "audio_url": {"url": audio.url}}
                    for audio in audio_assets
                ],
                {
                    "type": "text",
                    "text": f"What's happening in these {len(audio_assets)} audio clips?",  # noqa: E501
                },
            ],
        }
    ]

    chat_completion = await client.chat.completions.create(
        model=MODEL_NAME, messages=messages, max_tokens=10
    )

    assert len(chat_completion.choices) == 1
    choice = chat_completion.choices[0]
    assert choice.finish_reason == "length"


@pytest.fixture(scope="module")
def tower_only_adapter(tmp_path_factory) -> str:
    """A synthetic nonzero tower-only LoRA adapter.

    Built procedurally (fixed seed) instead of loading a trained adapter so
    the test is hermetic and the delta is guaranteed nonzero: freshly
    initialized PEFT exports keep lora_B == 0 and cannot change any output.
    """
    import json

    import torch
    from safetensors.torch import save_file
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    d_model = config.audio_config.d_model
    num_layers = config.audio_config.encoder_layers
    rank = 8

    gen = torch.Generator().manual_seed(0)
    tensors = {}
    for layer in range(num_layers):
        for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
            base = f"base_model.model.audio_tower.layers.{layer}.self_attn.{proj}"
            tensors[f"{base}.lora_A.weight"] = (
                torch.randn((rank, d_model), generator=gen, dtype=torch.float32) * 0.02
            )
            tensors[f"{base}.lora_B.weight"] = (
                torch.randn((d_model, rank), generator=gen, dtype=torch.float32) * 0.02
            )

    dst = tmp_path_factory.mktemp("ultravox_tower_lora")
    save_file(tensors, str(dst / "adapter_model.safetensors"))
    adapter_config = {
        "peft_type": "LORA",
        "r": rank,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": ["q_proj", "k_proj", "v_proj", "out_proj"],
    }
    (dst / "adapter_config.json").write_text(json.dumps(adapter_config))
    return str(dst)


@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [16])
def test_tower_lora_is_applied(
    vllm_runner,
    audio_assets: AudioTestAssets,
    tower_only_adapter: str,
    dtype: str,
    max_tokens: int,
) -> None:
    """A tower-only LoRA adapter must shift the generated logprobs.

    Greedy tokens can be insensitive to tower perturbations (basin effects),
    but punica adds exact zeros when no adapter weights are active, so with
    prefix caching disabled the engine is deterministic and ANY logprob
    deviation proves the tower LoRA delta was applied (guards against the
    silent no-op behavior of #31479).
    """
    from vllm.lora.request import LoRARequest

    model_info = HF_EXAMPLE_MODELS.find_hf_info(MODEL_NAME)
    model_info.check_available_online(on_fail="skip")
    model_info.check_transformers_version(on_fail="skip")

    prompt = _get_prompt(1, "Transcribe this audio.", VLLM_PLACEHOLDER)
    audio = audio_assets[0].audio_and_sample_rate

    with vllm_runner(
        MODEL_NAME,
        dtype=dtype,
        enforce_eager=True,
        # Prefix caching would let the second run reuse the first run's KV
        # blocks, hiding (or leaking) the adapter's effect.
        enable_prefix_caching=False,
        limit_mm_per_prompt={"audio": 1},
        enable_lora=True,
        enable_tower_connector_lora=True,
        max_lora_rank=8,
    ) as vllm_model:

        def greedy_logprobs(lora_request):
            [(token_ids, _, logprobs)] = vllm_model.generate_greedy_logprobs(
                [prompt],
                max_tokens,
                num_logprobs=1,
                audios=[[audio]],
                lora_request=lora_request,
            )
            return {tok: lps[tok].logprob for tok, lps in zip(token_ids, logprobs)}

        base = greedy_logprobs(None)
        base_repeat = greedy_logprobs(None)
        with_lora = greedy_logprobs(LoRARequest("tower-lora", 1, tower_only_adapter))

    assert base == base_repeat, "engine is not deterministic; test is invalid"
    shared = base.keys() & with_lora.keys()
    max_delta = max((abs(base[tok] - with_lora[tok]) for tok in shared), default=0.0)
    assert len(base) != len(with_lora) or max_delta > 1e-4, (
        "tower-only LoRA left all logprobs unchanged; tower LoRA weights "
        "appear to be silently ignored"
    )
