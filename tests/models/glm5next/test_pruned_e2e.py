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
import weakref

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


def _ids(n: int, seed: int = 0) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1000, 100000, (n,), generator=g).tolist()


def _prompt_ids(n: int, seed: int = 0) -> TokensPrompt:
    return TokensPrompt(prompt_token_ids=_ids(n, seed))


_live_llms: weakref.WeakSet[LLM] = weakref.WeakSet()


@pytest.fixture(autouse=True)
def _shutdown_llms():
    """A failed test's traceback keeps its LLM alive past the global cleanup;
    shut the engines down so the next test starts with a free GPU."""
    yield
    for llm in list(_live_llms):
        llm.llm_engine.engine_core.shutdown()


def _make_llm(**kwargs) -> LLM:
    args = dict(
        model=PRUNED_MODEL,
        max_model_len=MAX_MODEL_LEN,
        enforce_eager=True,
        # Text-only: skips vision-tower profiling (slow JIT) and its encoder cache.
        limit_mm_per_prompt={"image": 0, "video": 0},
        gpu_memory_utilization=0.6,
        max_num_batched_tokens=MAX_MODEL_LEN,
        max_num_seqs=8,
    )
    args.update(kwargs)
    llm = LLM(**args)
    _live_llms.add(llm)
    return llm


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
    """One MLA+indexer group, one kpool-tail ring group and 3 Mamba groups that
    alias the MLA tensor, as with the full model."""
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    llm = _make_llm()
    groups, tensors = llm.collective_rpc(_kv_layout)[0]

    kinds = sorted(tuple(k) for k, _, _ in groups)
    assert kinds == sorted(
        [
            ("MLAAttentionSpec", "MLAAttentionSpec"),
            ("CircularBufferSpec",),
            ("MambaSpec",),
            ("MambaSpec",),
            ("MambaSpec",),
        ]
    )
    tail_group = next(g for g in groups if g[0] == ["CircularBufferSpec"])
    assert tail_group[2] == 4  # index_kpool
    block_size = llm.llm_engine.vllm_config.cache_config.block_size
    assert block_size % (4 * 32) == 0  # paged-MQA pool-page alignment
    for kinds, _, bs in groups:
        # Without drafts the kpool tail ring holds index_kpool tokens; every
        # other group uses the auto block size.
        assert bs == (4 if kinds == ["CircularBufferSpec"] else block_size), kinds

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


def _mamba_groups(scheduler) -> list[tuple[int, list[str]]]:
    from vllm.v1.kv_cache_interface import MambaSpec

    return [
        (gid, list(group.layer_names))
        for gid, group in enumerate(scheduler.kv_cache_config.kv_cache_groups)
        if isinstance(group.kv_cache_spec, MambaSpec)
    ]


def _cached_block_ids(scheduler, group_id: int) -> list[int]:
    from vllm.v1.core.kv_cache_utils import get_group_id

    return sorted(
        block.block_id
        for block in scheduler.kv_cache_manager.block_pool.blocks
        if block.block_hash is not None and get_group_id(block.block_hash) == group_id
    )


def _state_pages(worker, layer_name: str, block_id: int) -> list[torch.Tensor]:
    """The (conv, recurrent) state pages of one Mamba layer at one block."""
    forward_context = worker.model_runner.vllm_config.compilation_config
    layer = forward_context.static_forward_context[layer_name]
    return [state[block_id].cpu().clone() for state in layer.kv_cache]


def test_prefill_checkpoint_lands_in_every_mamba_group(
    monkeypatch: pytest.MonkeyPatch,
):
    """A prefill that crosses a block boundary saves its internal checkpoint,
    the Mamba state at that boundary, into every KDA group's own cached page,
    and that page holds the same state as a cold prefill ending exactly there.

    The KDA groups share one metadata build per step (#58762); a per-group
    field left pointing at the first group's block table sends the other
    groups' checkpoints to the wrong page (#59536), which a token-level repeat
    on this model does not notice.
    """
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    # Dense retention keeps the reference run's retired block; the default
    # (0) retains only semantic checkpoints such as the long prompt's.
    llm = _make_llm(prefix_cache_retention_interval=None)
    scheduler = llm.llm_engine.engine_core.engine_core.scheduler
    block = llm.llm_engine.vllm_config.cache_config.block_size
    assert block < LONG_PROMPT_LEN < 2 * block
    groups = _mamba_groups(scheduler)
    assert len(groups) >= 2
    prompt = _ids(LONG_PROMPT_LEN)
    one_token = SamplingParams(temperature=0, max_tokens=1)

    def cached_pages() -> dict[str, list[torch.Tensor]]:
        pages = {}
        for group_id, layer_names in groups:
            # Exactly one cached Mamba block per group: the boundary state.
            (block_id,) = _cached_block_ids(scheduler, group_id)
            for name in layer_names:
                pages[name] = llm.collective_rpc(_state_pages, args=(name, block_id))[0]
        return pages

    # Reference through the ordinary state path: prefill exactly one block,
    # then one decode step crosses into the next block and retires the first
    # block's page, which now holds the boundary state, into the cache.
    two_tokens = SamplingParams(temperature=0, max_tokens=2)
    llm.generate([TokensPrompt(prompt_token_ids=prompt[:block])], two_tokens)
    reference = cached_pages()
    assert llm.reset_prefix_cache()

    # The long prompt crosses the boundary inside one prefill, so the only
    # cached Mamba block is its internal checkpoint at that boundary.
    llm.generate([TokensPrompt(prompt_token_ids=prompt)], one_token)
    checkpoint = cached_pages()
    # The two paths feed the KDA layers through different prefill batch
    # shapes, so the pages differ by bf16 rounding (relative error <= 0.03
    # observed); a page written by another group, or never written, is off by
    # a relative error of 1 or more.
    report = []
    for name, ref_pages in reference.items():
        for kind, ref, got in zip(("conv", "recurrent"), ref_pages, checkpoint[name]):
            ref, got = ref.float(), got.float()
            report.append((name, kind, ((got - ref).norm() / ref.norm()).item()))
    summary = "\n".join(
        f"{name} {kind}: relative error vs boundary state {rel:.3g}"
        for name, kind, rel in report
    )
    print(summary)
    assert all(rel <= 0.2 for _, _, rel in report), summary

    # The checkpoint is what a request sharing the first block resumes from.
    probe = TokensPrompt(prompt_token_ids=prompt[:block] + _ids(64, seed=4))
    assert llm.generate([probe], one_token)[0].num_cached_tokens == block


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


DFLASH2_DRAFT = os.getenv("GLM5_DFLASH2_DRAFT", "incoai/GLM-5.3-Flash-DFlash2")


def _remap_dflash2_to_pruned_target(hf_config):
    """Point the drafter's five aux-state taps at the boundaries the 4-layer
    target has: the embeddings and the outputs of layers 0-3 (ids are
    "after layer i", so -1 is the embedding stream). Applied to the target
    config too, where it is a no-op."""
    dflash_config = getattr(hf_config, "dflash_config", None)
    if isinstance(dflash_config, dict):
        dflash_config["target_layer_ids"] = [-1, 0, 1, 2, 3]
    return hf_config


def _layer_names(llm: LLM) -> set[str]:
    groups, _ = llm.collective_rpc(_kv_layout)[0]
    return {name for _, layers, _ in groups for name in layers}


def _spec_decode_drafts(llm: LLM) -> float:
    return sum(
        m.value
        for m in llm.get_metrics()
        if m.name.startswith("vllm:spec_decode_num_drafts")
    )


def test_dflash2_drafter_runs_losslessly(monkeypatch: pytest.MonkeyPatch):
    """The published DFlash2 drafter loads against the hybrid target (#56983:
    aux hidden-state capture through mHC, the drafter's GQA layers in the KV
    cache groups, no expert parallelism copied onto the dense drafter), its
    five attention layers get KV cache pages, drafts are verified, and greedy
    output agrees with the target-only run on the leading tokens. The drafter
    was trained for the 45-layer model, so acceptance on the pruned target is
    not asserted."""
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    sp = SamplingParams(temperature=0, max_tokens=8)
    prompts = [_prompt_ids(LONG_PROMPT_LEN), _prompt_ids(64, seed=2)]

    target = _make_llm()
    target_layers = _layer_names(target)
    base = [o.outputs[0].token_ids for o in target.generate(prompts, sp)]
    del target

    spec = _make_llm(
        speculative_config={
            "method": "dflash",
            "model": DFLASH2_DRAFT,
            "num_speculative_tokens": 6,
        },
        hf_overrides=_remap_dflash2_to_pruned_target,
        gpu_memory_utilization=0.9,
        disable_log_stats=False,
    )
    draft_layers = _layer_names(spec) - target_layers
    assert len(draft_layers) == 5, sorted(draft_layers)
    out = [o.outputs[0].token_ids for o in spec.generate(prompts, sp)]
    assert all(len(t) == 8 for t in out)
    assert [list(t[:4]) for t in out] == [list(t[:4]) for t in base]
    assert _spec_decode_drafts(spec) > 0


def test_default_max_num_seqs_tp1():
    """Startup and a generation at the default max_num_seqs of large GPUs
    (1024 sequences x 64 local KDA heads once hit CUDA's gridDim.z limit in
    the recurrent kernel's warmup, vllm-project/vllm#56973)."""
    llm = _make_llm(max_num_seqs=1024)
    out = llm.generate([_prompt_ids(8, seed=3)], SamplingParams(max_tokens=2))
    assert len(out[0].outputs[0].token_ids) == 2
