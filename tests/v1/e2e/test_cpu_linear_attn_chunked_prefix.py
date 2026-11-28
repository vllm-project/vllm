# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU chunked prefill and prefix caching for hybrid models with recurrent state."""

import os

import pytest
import torch

from tests.models.utils import TokensTextLogprobs, check_logprobs_close
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.outputs import RequestOutput
from vllm.platforms import current_platform

pytestmark = pytest.mark.cpu_model

if not current_platform.is_cpu():
    pytest.skip("skipping CPU-only tests", allow_module_level=True)

# Bound the KV cache so the run does not scale with host memory; these engines
# only need a few thousand tokens.
os.environ.setdefault("VLLM_CPU_KVCACHE_SPACE", "1")

MODEL = "Qwen/Qwen3.5-0.8B"
CHUNK_TOKENS = 128  # max_num_batched_tokens for the chunked engine
NUM_LOGPROBS = 5
SP = SamplingParams(max_tokens=32, temperature=0, logprobs=NUM_LOGPROBS)


def _long_prompt(repeat: int) -> str:
    return "Solve the following arithmetic step by step. " * repeat + "What is 7*8?"


# Prompts long enough to span several CHUNK_TOKENS-sized chunks; a single-chunk
# prompt is bit-identical to full prefill regardless of the bug.
PROMPTS = [_long_prompt(r) for r in (40, 60, 80)]
# Spans several full cache blocks; prefix caching only reuses complete blocks.
PREFIX_PROMPT = "You are a helpful assistant. " * 230 + " Now answer: what is 2+2?"


def _make_llm(**overrides) -> LLM:
    base = dict(model=MODEL, dtype="bfloat16", max_model_len=2048, enforce_eager=True)
    base.update(overrides)
    return LLM(**base)


def _tuples(outputs: list[RequestOutput]) -> list[TokensTextLogprobs]:
    """(token_ids, text, sample_logprobs) per request, for check_logprobs_close."""
    return [
        (list(o.outputs[0].token_ids), o.outputs[0].text, o.outputs[0].logprobs)
        for o in outputs
    ]


@pytest.fixture(scope="module")
def full_prefill_refs():
    """Reference (ids, text, logprobs) for PROMPTS and PREFIX_PROMPT, full prefill."""
    llm = _make_llm(enable_chunked_prefill=False, enable_prefix_caching=False)
    refs = _tuples(llm.generate(PROMPTS, SP))
    prefix_ref = _tuples(llm.generate([PREFIX_PROMPT], SP))[0]
    del llm
    return refs, prefix_ref


def test_chunked_prefill_matches_full_prefill(full_prefill_refs):
    """Batched multi-chunk prefill must stay close to per-prompt full prefill.

    Prompts are scheduled together so the scheduler interleaves prefill chunks
    across requests (the cross-request path where the accuracy gap was strongest).
    """
    refs, _ = full_prefill_refs
    llm = _make_llm(
        enable_chunked_prefill=True,
        max_num_batched_tokens=CHUNK_TOKENS,
        enable_prefix_caching=False,
    )
    got = _tuples(llm.generate(PROMPTS, SP))
    del llm

    check_logprobs_close(
        outputs_0_lst=refs,
        outputs_1_lst=got,
        name_0="full_prefill",
        name_1="chunked_prefill",
    )


def test_prefix_cache_hit_matches_cold_cache(full_prefill_refs):
    """A prefix-cache hit must stay close to the cold-cache (reference) output.

    The warm run continues prefill from the restored GDN state; the
    num_cached_tokens check guards against a vacuous (no-hit) pass.
    """
    _, ref = full_prefill_refs
    llm = _make_llm(enable_prefix_caching=True)
    llm.generate([PREFIX_PROMPT], SP)  # prime the cache
    warm_out = llm.generate([PREFIX_PROMPT], SP)[0]
    warm = _tuples([warm_out])[0]
    del llm

    assert warm_out.num_cached_tokens is not None
    assert warm_out.num_cached_tokens > 0, (
        "expected a prefix-cache hit but num_cached_tokens=0; "
        "PREFIX_PROMPT may be shorter than one cache block"
    )
    check_logprobs_close(
        outputs_0_lst=[ref],
        outputs_1_lst=[warm],
        name_0="cold_cache",
        name_1="warm_cache",
    )


def test_prefix_cache_hit_reaches_last_full_block():
    """A repeated prompt must hit every full block of the hybrid block size.

    The workers raise the block size to fit the GDN state pages, and the
    engine core reruns its config checks after its handshake; the raised size
    must survive them, or the prefill's state checkpoint misses the block
    boundaries and nothing is reused. One chunk holds the whole prompt.
    """
    llm = _make_llm(enable_prefix_caching=True, max_num_batched_tokens=2048)
    block_size = llm.llm_engine.vllm_config.cache_config.block_size
    # Ends mid-block: only the last full block's state can be reused.
    num_tokens = block_size + block_size // 2
    assert num_tokens <= 2048
    prompt = TokensPrompt(prompt_token_ids=[1000 + i for i in range(num_tokens)])
    sp = SamplingParams(max_tokens=1)
    hits = [llm.generate(prompt, sp)[0].num_cached_tokens for _ in range(2)]
    del llm

    assert hits == [0, block_size]


def test_lfm2_shortconv_internal_checkpoint(tmp_path, monkeypatch, vllm_runner):
    """One CPU prefill exports an internal ShortConv block for the next request."""
    from transformers import Lfm2Config, Lfm2ForCausalLM

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    config = Lfm2Config(
        hidden_size=128,
        conv_dim=128,
        block_dim=128,
        intermediate_size=256,
        block_multiple_of=64,
        block_auto_adjust_ff_dim=False,
        vocab_size=256,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_hidden_layers=2,
        layer_types=["conv", "full_attention"],
        full_attn_idxs=[1],
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(82)
        Lfm2ForCausalLM(config).to(torch.bfloat16).save_pretrained(tmp_path)

    with vllm_runner(
        str(tmp_path),
        dtype="bfloat16",
        skip_tokenizer_init=True,
        enforce_eager=True,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        async_scheduling=False,
        block_size=64,
        max_model_len=512,
        max_num_batched_tokens=512,
        max_num_seqs=1,
        kv_cache_memory_bytes=64 << 20,
    ) as runner:
        llm = runner.llm
        rng = torch.Generator().manual_seed(186)
        prefix = torch.randint(16, 256, (256,), generator=rng).tolist()
        params = SamplingParams(temperature=0, max_tokens=1, detokenize=False)
        llm.enqueue(
            TokensPrompt(prompt_token_ids=prefix + [3, 4, 5, 6, 7, 8, 9]), params
        )
        # A prefill split at token 256 cannot emit the first token on this step.
        outputs = llm.llm_engine.step()
        assert len(outputs) == 1 and outputs[0].finished

        branch = TokensPrompt(prompt_token_ids=prefix + [10, 11, 12, 13, 14, 15])
        params = SamplingParams(
            temperature=0, max_tokens=4, ignore_eos=True, detokenize=False, logprobs=1
        )
        warm = llm.generate(branch, params)[0]
        assert llm.reset_prefix_cache()
        cold = llm.generate(branch, params)[0]
        assert warm.num_cached_tokens == len(prefix) and cold.num_cached_tokens == 0
        warm_out, cold_out = warm.outputs[0], cold.outputs[0]
        assert warm_out.token_ids == cold_out.token_ids
        assert warm_out.logprobs is not None and cold_out.logprobs is not None
        for token, warm_step, cold_step in zip(
            warm_out.token_ids, warm_out.logprobs, cold_out.logprobs, strict=True
        ):
            assert warm_step[token].logprob == pytest.approx(
                cold_step[token].logprob, abs=0.02
            )
