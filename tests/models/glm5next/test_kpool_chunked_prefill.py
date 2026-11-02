# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check boundary pools during real GLM-5.3-Flash chunked prefill."""

import os
from typing import TypedDict

import pytest

from vllm import SamplingParams
from vllm.platforms import current_platform

from ...utils import create_new_process_for_each_test


@pytest.mark.skipif(not current_platform.is_cuda_alike(), reason="Requires GPU Triton")
@create_new_process_for_each_test("spawn")
def test_prefill_gather_metadata_offsets_do_not_recompile():
    import torch

    from vllm.models.glm5next.common.sparse_indexer import _gather_kpool_prefill
    from vllm.utils import jit_monitor

    torch.manual_seed(19)
    n = 128
    key = torch.randn(n, 128, dtype=torch.bfloat16, device="cuda")
    score = torch.randn(n, 128, dtype=torch.float32, device="cuda")
    positions = torch.arange(35, 35 + n, device="cuda")
    slots = torch.where(positions % 4 == 3, positions // 4, -1)
    tail_slots = 16 + positions % 16
    tail = torch.randn(2, 2, 16, 128, dtype=torch.bfloat16, device="cuda")
    offsets = []
    for metadata in (slots, positions, tail_slots):
        view = torch.empty(n + 1, dtype=metadata.dtype, device="cuda")[1:]
        view.copy_(metadata)
        offsets.append(view)
    expected = _gather_kpool_prefill(key, score, slots, 4, positions, tail, tail_slots)
    torch.accelerator.synchronize()
    jit_monitor.activate(mode="error", verbose=True)
    actual = _gather_kpool_prefill(
        key, score, offsets[0], 4, offsets[1], tail, offsets[2]
    )
    assert torch.equal(actual[2], expected[2])
    for values, reference in zip(actual[:2], expected[:2]):
        assert torch.equal(values[actual[2]], reference[expected[2]])


class _PoolObservations(TypedDict):
    checked: int
    unaligned: int
    mismatches: list[dict[str, int]]
    starts: list[int]


def _attach_pool_observer(model):
    import torch

    from vllm.forward_context import get_forward_context
    from vllm.models.glm5next.nvidia.ops.kpool_compress import (
        kpool_compress_and_write_cache,
    )
    from vllm.models.glm5next.nvidia.sparse_indexer import SparseAttnIndexerKpool

    report: _PoolObservations = {
        "checked": 0,
        "unaligned": 0,
        "mismatches": [],
        "starts": [],
    }
    model.kpool_prefill_observations = report

    def pre(mod, inputs, kwargs):
        meta = get_forward_context().attn_metadata
        mod.kpool_prefill_pending = []
        if not isinstance(meta, dict) or mod.k_cache.prefix not in meta:
            return
        km = meta[mod.k_cache.prefix]
        if not km.num_prefill_tokens:
            return
        tm = meta[mod.tail_cache.prefix]
        begin = km.num_decode_tokens
        end = begin + km.num_prefill_tokens
        keys, scores = inputs[2][begin:end], kwargs["gate_score"][begin:end]
        positions = kwargs["positions"][begin:end].tolist()
        slots = km.slot_mapping[begin:end].tolist()
        tail_slots = tm.slot_mapping[begin:end].tolist()
        pool = kwargs["index_kpool"]
        ring = mod.tail_cache.kv_cache.shape[2]
        chosen = set()
        lookup = {
            (t // ring, pos): i
            for i, (t, pos) in enumerate(zip(tail_slots, positions))
            if t >= 0
        }
        for pos, slot, ts in zip(positions, slots, tail_slots):
            if slot < 0 or ts < 0 or ts // ring in chosen:
                continue
            block = ts // ring
            chosen.add(block)
            start = min(p for b, p in lookup if b == block)
            report["starts"].append(start)
            report["unaligned"] += int(start % pool != 0)
            ks, ss = [], []
            for member in range(pos - pool + 1, pos + 1):
                src = lookup.get((block, member))
                if src is None:
                    ks.append(mod.tail_cache.kv_cache[block, 0, member % ring])
                    ss.append(mod.tail_cache.kv_cache[block, 1, member % ring])
                else:
                    ks.append(keys[src])
                    ss.append(scores[src])
            expected = torch.zeros(1, 64, 132, dtype=torch.uint8, device=keys.device)
            kpool_compress_and_write_cache(
                expected,
                torch.stack(ks)[None],
                torch.stack(ss)[None],
                kwargs["compress_ape"],
                torch.zeros(1, dtype=torch.int64, device=keys.device),
                pool,
                mod.head_dim,
                round_scale=mod.scale_fmt is not None,
            )
            flat = expected[0].reshape(-1)
            want = torch.cat((flat[:128], flat[64 * 128 : 64 * 128 + 4]))
            mod.kpool_prefill_pending.append((slot, want, start, pos))

    def post(mod, inputs, kwargs, output):
        for slot, want, start, pos in mod.kpool_prefill_pending:
            cache = mod.k_cache.kv_cache
            page, offset = divmod(slot, cache.shape[1])
            flat = cache[page].reshape(-1)
            got = torch.cat(
                (
                    flat[offset * 128 : (offset + 1) * 128],
                    flat[
                        cache.shape[1] * 128 + 4 * offset : cache.shape[1] * 128
                        + 4 * offset
                        + 4
                    ],
                )
            )
            report["checked"] += 1
            if not torch.equal(got, want):
                report["mismatches"].append(
                    {
                        "start": start,
                        "completion": pos,
                        "slot": slot,
                        "different_bytes": int((got != want).sum()),
                    }
                )
        mod.kpool_prefill_pending = []

    found = 0
    for mod in model.modules():
        if isinstance(mod, SparseAttnIndexerKpool):
            mod.register_forward_pre_hook(pre, with_kwargs=True)
            mod.register_forward_hook(post, with_kwargs=True)
            found += 1
    return found


@pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.support_deep_gemm(),
    reason="The real checkpoint requires CUDA Hopper/Blackwell DeepGEMM",
)
def test_llm_chunked_prefill_keeps_boundary_pool_history(vllm_runner, enable_pickle):
    model = os.environ.get("GLM5_PRUNED_MODEL", "JaredforReal/GLM-5.3-Flash-4L")
    with vllm_runner(
        model,
        revision="035b309e03a83deb2589d722ddf9e3c5d3de5caf",
        block_size=None,
        max_model_len=8192,
        max_num_batched_tokens=2045,
        max_num_seqs=2,
        enforce_eager=True,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        gpu_memory_utilization=float(os.environ.get("GLM5_TEST_GPU_UTIL", "0.9")),
        limit_mm_per_prompt={"image": 0, "video": 0},
        moe_backend="deep_gemm",
    ) as runner:
        llm = runner.llm
        assert llm.apply_model(_attach_pool_observer) == [1]
        tokenizer = llm.get_tokenizer()
        texts = [
            "The river flows through a quiet valley. " * 800,
            "A careful engineer records each measurement. " * 800,
        ]
        prompts = [
            {"prompt_token_ids": tokenizer.encode(text)[: 4807 + i]}
            for i, text in enumerate(texts)
        ]
        llm.generate(
            prompts,
            SamplingParams(temperature=0, max_tokens=4),
            use_tqdm=False,
        )
        report = llm.apply_model(lambda model: model.kpool_prefill_observations)[0]
        print(report)
        assert report["unaligned"] > 0, "Must exercise pools spanning prefill chunks"
        assert report["mismatches"] == [], report
