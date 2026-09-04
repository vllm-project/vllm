# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm import SamplingParams
from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import MambaSpec

from ...utils import large_gpu_mark

QWEN_MODEL = "Qwen/Qwen3.5-4B"
QWEN_KV_CACHE_BYTES = 12 << 30
HYBRID_MTP_MODELS = [
    pytest.param(
        QWEN_MODEL,
        marks=[large_gpu_mark(min_gb=30)],
        id="qwen",
    ),
    pytest.param(
        "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
        marks=[large_gpu_mark(min_gb=80)]
        + [
            pytest.mark.skipif(
                not current_platform.is_cuda(),
                reason="modelopt quantization is supported only on CUDA",
            )
        ],
        id="nemotron",
    ),
]

# A trivial request with a short prompt to ensure we run a mixed batch
SMALL_MESSAGE = [
    {
        "role": "user",
        "content": "The secret beta value is 64. What is the secret beta?",
    }
]

# Sample prompt with a bunch of filler in between the critical fact and the request.
# Both parts need to be processed properly for the model to generate the correct answer
MESSAGES = [
    {
        "role": "user",
        "content": (
            "Important: The secret number is 42. "
            "The sky is green in this hypothetical world. "
            "Apples grow on trees in the forest. "
            "Rivers flow through the valleys and mountains. "
            "Birds sing songs in the early morning light. "
            "The weather today is sunny with clear skies ahead. "
            "Flowers bloom in the garden during spring season. "
            "Now answer with ONLY the number and nothing else: "
            "What is the secret number plus one?"
        ),
    }
]


@pytest.mark.parametrize(
    "model_name",
    HYBRID_MTP_MODELS,
)
@pytest.mark.parametrize("enable_prefix_caching", [False, True])
def test_mtp_speculative_mixed_batch_short_prefill(
    vllm_runner, model_name, enable_prefix_caching
):
    """Test to ensure MTP speculative decoding correctly handles
    short prefill chunks that fall below the reorder_batch_threshold."""

    # Set so large that both prefills will be classified as decodes in a mixed batch
    # note, with prefix caching we require chunk_size >= mamba_block_size
    chunk_size = 256 if not enable_prefix_caching else 2048
    num_draft_tokens = 100

    with vllm_runner(
        model_name,
        speculative_config={
            "method": "mtp",
            "num_speculative_tokens": num_draft_tokens,
        },
        max_num_batched_tokens=chunk_size,
        max_num_seqs=4,
        max_model_len=512,
        kv_cache_memory_bytes=QWEN_KV_CACHE_BYTES if model_name == QWEN_MODEL else None,
        enforce_eager=True,
        tensor_parallel_size=1,
        trust_remote_code=True,
        enable_chunked_prefill=True,
        enable_prefix_caching=enable_prefix_caching,
        mamba_cache_mode="align" if enable_prefix_caching else "none",
    ) as llm:
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=128,
        )

        # First small message gets prefilled first, under normal conditions since the
        # batch is not yet mixed. Then the second prefill arrives as a mixed batch, but
        # is shorter than num_speculative_tokens, so it gets misclassified as a decode
        # and processed with the wrong state management logic,  causing the critical
        # fact from the first chunk to be lost and the model to generate nonsense.
        outputs = llm.get_llm().chat(
            [SMALL_MESSAGE, MESSAGES],
            sampling_params,
            chat_template_kwargs={"enable_thinking": False},
        )

        responses = []
        for output in outputs:
            generated_text = output.outputs[0].text
            print(f"Generated text: {generated_text!r}")
            responses.append(generated_text)

        assert "64" in responses[0], (
            "The first response should contain the correct value of 64."
        )
        assert "43" in responses[1], (
            "The second response should contain the correct value of 42+1=43."
        )


def _get_mamba_block_size(llm) -> int:
    scheduler = llm.llm_engine.engine_core.engine_core.scheduler
    block_sizes = {
        group.kv_cache_spec.block_size
        for group in scheduler.kv_cache_config.kv_cache_groups
        if isinstance(group.kv_cache_spec, MambaSpec)
    }
    assert len(block_sizes) == 1
    block_size = block_sizes.pop()
    assert scheduler.cache_config.block_size == block_size
    return block_size


def _build_access_code_manual(tokenizer, target_tokens: int) -> tuple[str, list[str]]:
    codes = ["605341", "693278", "597596", "751982"]
    header = "Memorize this facility manual and its access codes.\n\n"
    facts = "".join(
        f"The access code for vault-{i:02d} is {code}.\n"
        for i, code in enumerate(codes)
    )
    filler = "Routine facility inspections are recorded in the audit ledger every day. "
    manual = header + facts
    while len(tokenizer.encode(manual)) < target_tokens:
        manual = header + filler + manual.removeprefix(header)
    return manual, codes


@large_gpu_mark(min_gb=30)
def test_qwen_mtp_mamba_prefix_cache_hit_is_bounded(vllm_runner, monkeypatch):
    """MTP must not extend a hybrid Mamba hit past the attention hit."""
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    with vllm_runner(
        QWEN_MODEL,
        tensor_parallel_size=1,
        max_model_len=8192,
        kv_cache_memory_bytes=QWEN_KV_CACHE_BYTES,
        block_size=None,
        enforce_eager=True,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        mamba_cache_mode="align",
        speculative_config={"method": "mtp", "num_speculative_tokens": 2},
    ) as runner:
        llm = runner.get_llm()
        block_size = _get_mamba_block_size(llm)
        manual, codes = _build_access_code_manual(
            llm.get_tokenizer(), 2 * block_size - 192
        )
        manual_tokens = len(llm.get_tokenizer().encode(manual))
        assert block_size < manual_tokens < 2 * block_size

        wave1_prompts = [
            manual
            + f"\nDescribe the audit procedure for vault-{i:02d} in detail.\nAnswer:"
            for i in range(len(codes))
        ]
        decode_tokens = 2 * block_size - manual_tokens + 96
        wave1_params = SamplingParams(
            temperature=0.0,
            min_tokens=decode_tokens,
            max_tokens=decode_tokens,
            ignore_eos=True,
        )
        wave1_outputs = llm.generate(wave1_prompts, wave1_params)

        wave2_prompts = [
            prompt
            + output.outputs[0].text
            + f"\n\nWhat is the access code for vault-{i:02d}? Answer:"
            for i, (prompt, output) in enumerate(zip(wave1_prompts, wave1_outputs))
        ]
        warm_params = SamplingParams(temperature=0.0, max_tokens=24, stop=["\n"])
        cold_params = SamplingParams(
            temperature=0.0,
            max_tokens=24,
            stop=["\n"],
            skip_reading_prefix_cache=True,
        )
        warm_outputs = llm.generate(wave2_prompts, warm_params)
        cold_outputs = llm.generate(wave2_prompts, cold_params)

    cached_tokens = [output.num_cached_tokens for output in warm_outputs]
    assert max(cached_tokens) == block_size

    for arm, outputs in (("warm", warm_outputs), ("cold", cold_outputs)):
        missed = [
            i
            for i, (code, output) in enumerate(zip(codes, outputs))
            if code not in output.outputs[0].text
        ]
        assert not missed, f"{arm} cache missed access codes for prompts {missed}"
