# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step5 CSA on ROCm through the engine, on the reduced registry checkpoint
with top-k 16 so prompts past 128 tokens reach scored prefill and real decode
selection.

Dummy weights give a near-uniform attention softmax and gate, so this guards
selection, paging and summary-page addressing (which V rows are read). Scores,
scale, masks and the gate are covered by test_amd_csa_paged.py.

_LOGPROB_ATOL is provisional and will be calibrated on GPU.
"""

import random

import pytest

from vllm import SamplingParams
from vllm.inputs import TokensPrompt
from vllm.platforms import current_platform

from ..registry import HF_EXAMPLE_MODELS
from ..utils import dummy_hf_overrides

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="Step5 CSA is the ROCm path"
)

ARCH = "Step5ForConditionalGeneration"
TOPK = 16
DECODE_TOKENS = 24
_LOGPROB_ATOL = 1e-2


def tiny_step5(cfg):
    info = HF_EXAMPLE_MODELS.get_hf_info(ARCH)
    cfg = dummy_hf_overrides(
        cfg,
        model_arch=ARCH,
        exist_overrides=info.hf_overrides,
        use_original_num_layers=True,
    )
    text = cfg.get_text_config()
    text.sparse_config = {**text.sparse_config, "topk": TOPK}
    return cfg


def _prompt(length: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(1000, 100000) for _ in range(length)]


def test_sparse_prefill_matches_decode(vllm_runner, monkeypatch):
    """Decode (fused summary update, summary scores, sparse decode attention)
    must give each sampled token the logprob a chunked teacher-forced re-prefill
    gives it (ring replay across chunks that split regions, scored prefill,
    paged sparse attention), on the FP8 KV cache at block size 32. The
    300-token steps also admit the scored-prefill warmup."""
    from vllm.models.step5.amd import warmup
    from vllm.platforms.rocm import on_mi3xx

    if not on_mi3xx():
        pytest.skip("Step5 on ROCm requires MI300 or MI350")
    info = HF_EXAMPLE_MODELS.get_hf_info(ARCH)
    info.check_available_online(on_fail="skip")
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    warmed = []
    run_warmup = warmup.run_step5_scored_prefill_warmup

    def spy(worker, topk):
        warmed.append(run_warmup(worker, topk))
        return warmed[-1]

    monkeypatch.setattr(warmup, "run_step5_scored_prefill_warmup", spy)
    prompts = [_prompt(700, 0), _prompt(1003, 1)]
    with vllm_runner(
        info.default,
        load_format="dummy",
        hf_overrides=tiny_step5,
        max_model_len=4096,
        max_num_seqs=4,
        gpu_memory_utilization=0.8,
        enable_prefix_caching=False,
        limit_mm_per_prompt={"image": 0, "video": 0},
        kv_cache_dtype="fp8",
        block_size=32,
        enable_chunked_prefill=True,
        max_num_batched_tokens=300,
    ) as runner:
        cache = runner.llm.llm_engine.vllm_config.cache_config
        assert (cache.cache_dtype, cache.block_size) == ("fp8", 32)
        decoded = runner.llm.generate(
            [TokensPrompt(prompt_token_ids=p) for p in prompts],
            SamplingParams(
                temperature=0.0,
                max_tokens=DECODE_TOKENS,
                logprobs=0,
                ignore_eos=True,
            ),
        )
        forced = runner.llm.generate(
            [
                TokensPrompt(prompt_token_ids=p + list(o.outputs[0].token_ids))
                for p, o in zip(prompts, decoded)
            ],
            SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0),
        )
    assert warmed == [True]

    deltas = []
    for prompt, dec, ref in zip(prompts, decoded, forced):
        sample = dec.outputs[0]
        teacher = ref.prompt_logprobs[len(prompt) :]
        assert len(teacher) == len(sample.token_ids) == DECODE_TOKENS
        for token, lp, tlp in zip(sample.token_ids, sample.logprobs, teacher):
            deltas.append(abs(lp[token].logprob - tlp[token].logprob))
    worst = max(range(len(deltas)), key=deltas.__getitem__)
    assert deltas[worst] <= _LOGPROB_ATOL, (
        f"max |dlogprob| {deltas[worst]:.3e} at request {worst // DECODE_TOKENS}, "
        f"decode step {worst % DECODE_TOKENS}"
    )
