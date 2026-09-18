# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-GPU end-to-end checks on the layer-pruned GLM-5.3-Flash checkpoint.

``JaredforReal/GLM-5.3-Flash-4L`` keeps layers 0-3 (+MTP) of the real
checkpoint with every width unchanged, so the model exercises the same
hybrid KV cache layout (KDA Mamba groups aliasing the MLA slot, kpool
indexer + tail scratch group), fp8 MoE, mHC and MTP kernels as the 300B
model while loading in seconds. Requires SM90/SM100 (DeepGEMM indexer).
"""

import os

import pytest
import torch

from vllm import LLM, SamplingParams, TokensPrompt
from vllm.platforms import current_platform

PRUNED_MODEL = os.getenv("GLM5_PRUNED_MODEL", "JaredforReal/GLM-5.3-Flash-4L")
# > index_topk (2048) so the sparse top-k path runs, and > one auto block
# (4352 tokens at TP1) so a request spans a block boundary.
LONG_PROMPT_LEN = 6000
MAX_MODEL_LEN = 8192

pytestmark = pytest.mark.skipif(
    not (
        current_platform.is_cuda()
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] in (9, 10)
    ),
    reason="GLM-5.3-Flash sparse indexer needs SM90/SM100",
)


def _prompt_ids(n: int, seed: int = 0) -> TokensPrompt:
    g = torch.Generator().manual_seed(seed)
    return TokensPrompt(
        prompt_token_ids=torch.randint(1000, 100000, (n,), generator=g).tolist()
    )


def _make_llm(**kwargs) -> LLM:
    return LLM(
        model=PRUNED_MODEL,
        max_model_len=MAX_MODEL_LEN,
        enforce_eager=True,
        # Text-only: skips vision-tower profiling (slow JIT) and its encoder cache.
        limit_mm_per_prompt={"image": 0, "video": 0},
        gpu_memory_utilization=0.6,
        max_num_batched_tokens=MAX_MODEL_LEN,
        **{"max_num_seqs": 8, **kwargs},
    )


def _kv_layout(worker):
    cfg = worker.model_runner.kv_cache_config
    groups = []
    for g in cfg.kv_cache_groups:
        spec = g.kv_cache_spec
        inner = getattr(spec, "kv_cache_specs", None)
        kinds = (
            sorted(type(s).__name__ for s in inner.values())
            if inner
            else [type(spec).__name__]
        )
        groups.append((kinds, sorted(g.layer_names), spec.block_size))
    tensors = {tuple(t.layers): t.offset for t in cfg.kv_cache_tensors}
    return groups, tensors


def test_kv_cache_layout(monkeypatch: pytest.MonkeyPatch):
    """One MLA+indexer group, one kpool-tail group and 3 Mamba groups that
    alias the MLA tensor, as with the full model."""
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    llm = _make_llm()
    groups, tensors = llm.collective_rpc(_kv_layout)[0]

    kinds = sorted(tuple(k) for k, _, _ in groups)
    assert kinds == sorted(
        [
            ("MLAAttentionSpec", "MLAAttentionSpec"),
            ("KpoolTailSpec",),
            ("MambaSpec",),
            ("MambaSpec",),
            ("MambaSpec",),
        ]
    )
    tail_group = next(g for g in groups if g[0] == ["KpoolTailSpec"])
    assert tail_group[2] == 4  # index_kpool
    block_size = llm.llm_engine.vllm_config.cache_config.block_size
    assert block_size % (4 * 32) == 0  # paged-MQA pool-page alignment
    for _, layers, bs in groups:
        if "KpoolTailSpec" not in layers:
            assert bs in (block_size, 4)

    attn = next(k for k in tensors if k[0].endswith("self_attn.attn"))
    mamba = [k for k in tensors if k[0].endswith("self_attn")]
    assert len(mamba) == 3
    assert all(tensors[m] == tensors[attn] for m in mamba)
    idx = next(k for k in tensors if k[0].endswith("indexer.k_cache"))
    tail = next(k for k in tensors if k[0].endswith("indexer.tail_cache"))
    assert tensors[idx] == tensors[tail]


def test_greedy_is_reproducible_across_block_boundary():
    llm = _make_llm()
    sp = SamplingParams(temperature=0, max_tokens=16)
    prompts = [_prompt_ids(LONG_PROMPT_LEN), _prompt_ids(8, seed=1)]
    first = [o.outputs[0].token_ids for o in llm.generate(prompts, sp)]
    second = [o.outputs[0].token_ids for o in llm.generate(prompts, sp)]
    assert first == second
    assert all(len(t) == 16 for t in first)


def test_mtp_speculative_decoding_runs():
    """MTP layer (renumbered layers.4) loads as the drafter and greedy output
    agrees with the target-only run on the leading tokens."""
    sp = SamplingParams(temperature=0, max_tokens=8)
    prompt = _prompt_ids(64, seed=2)
    base = _make_llm().generate([prompt], sp)[0].outputs[0].token_ids
    spec = _make_llm(speculative_config={"method": "mtp", "num_speculative_tokens": 1})
    out = spec.generate([prompt], sp)[0].outputs[0].token_ids
    assert len(out) == 8
    assert list(out[:4]) == list(base[:4])


@pytest.mark.xfail(
    strict=True,
    reason="fused_recurrent_kda_fwd launches grid.z = num_seqs * local KDA heads; "
    "64 heads x 1024 seqs (the default max_num_seqs on >=70GiB GPUs) exceeds the "
    "CUDA limit of 65535 -> 'Triton Error [CUDA]: invalid argument' "
    "(vllm-project/vllm#56973, fixed by #56974; drop this marker once it lands)",
)
def test_default_max_num_seqs_tp1():
    llm = _make_llm(max_num_seqs=1024)
    out = llm.generate([_prompt_ids(8, seed=3)], SamplingParams(max_tokens=2))
    assert len(out[0].outputs[0].token_ids) == 2
