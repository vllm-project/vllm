# SPDX-License-Identifier: Apache-2.0
"""Regression test: mamba "align" prefix-cache poisoning via unaligned chunks.

With speculative decoding (use_eagle=True), Scheduler._mamba_block_aligned_split
zeroes its alignment target for prompts shorter than 2*mamba_block_size
(`last_cache_position = max(last - block_size, 0)`), so a prefill whose first
chunk is capped by leftover token budget (concurrent prefills sharing a step)
ends at an arbitrary offset. The GDN kernel then writes that mid-block state
into the request's position-0 mamba slot, and MambaManager.cache_blocks later
hashes that slot as the state at `block_size` tokens — poisoning the prefix
cache for every request that resumes from it.

This test drives the real KVCacheManager + the real chunk-split helper through
that scenario on CPU and asserts the invariant:

    a hash-cached mamba block at position p holds the recurrent state of
    exactly (p+1) * block_size tokens.

It fails on the unfixed v0.22.1 scheduler and passes with the fix.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.request import Request

FULL_BS = 16
MAMBA_BS = 1600
NUM_SPEC = 3
BUDGET = 16384
PROMPT_LEN = 2002
MAX_MODEL_LEN = 262144


def _make_request(rid: str, tokens: list[int]) -> Request:
    sp = SamplingParams(max_tokens=64)
    sp.update_from_generation_config({}, eos_token_id=100)
    return Request(
        request_id=rid,
        prompt_token_ids=tokens,
        mm_features=None,
        sampling_params=sp,
        pooling_params=None,
        lora_request=None,
        block_hasher=get_request_block_hasher(FULL_BS, sha256),
    )


def _make_manager() -> KVCacheManager:
    cfg = KVCacheConfig(
        num_blocks=10000,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full_layer"],
                FullAttentionSpec(
                    block_size=FULL_BS,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba_layer"],
                MambaSpec(
                    block_size=MAMBA_BS,
                    shapes=(1, 1),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                    num_speculative_blocks=NUM_SPEC,
                ),
            ),
        ],
    )
    return KVCacheManager(
        cfg,
        max_model_len=MAX_MODEL_LEN,
        scheduler_block_size=MAMBA_BS,
        hash_block_size=FULL_BS,
        max_num_batched_tokens=BUDGET,
        enable_caching=True,
        use_eagle=True,
    )


def _aligned_split(
    req, num_new, num_new_local_computed=0, num_external_computed=0, use_eagle=True
):
    stub = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=MAMBA_BS),
        use_eagle=use_eagle,  # MTP / EAGLE-family speculative decoding
    )
    return Scheduler._mamba_block_aligned_split(
        stub, req, num_new, num_new_local_computed, num_external_computed
    )


def test_fragmented_prefill_chunk_does_not_poison_mamba_prefix_cache():
    init_none_hash(sha256)
    mgr = _make_manager()
    mamba = mgr.coordinator.single_type_managers[1]
    tokens = list(range(3, 3 + PROMPT_LEN))
    reqs = [_make_request(f"req{i}", tokens) for i in range(10)]

    # oracle: physical mamba block id -> token offset of the state it holds
    block_tag: dict[int, int] = {}
    cached_seen: dict[str, int] = {}
    violations: list[str] = []

    waiting = list(reqs)
    running: list[Request] = []
    finished: set[str] = set()

    for _step in range(400):
        if not waiting and all(r.request_id in finished for r in running):
            break
        budget = BUDGET
        scheduled = []

        for req in running:
            if req.request_id in finished or budget <= 0:
                continue
            c = req.num_computed_tokens
            if c < req.num_tokens:
                num_new = _aligned_split(req, min(req.num_tokens - c, budget))
                if num_new == 0:
                    continue
                lookahead = NUM_SPEC
            else:
                req.spec_token_ids = [7] * NUM_SPEC
                num_new = 1 + NUM_SPEC
                lookahead = 0
            assert (
                mgr.allocate_slots(req, num_new, num_lookahead_tokens=lookahead)
                is not None
            )
            budget -= num_new
            scheduled.append((req, c, num_new))

        admitted = []
        for req in waiting:
            if budget <= 0:
                break
            computed_blocks, num_computed = mgr.get_computed_blocks(req)
            num_new = _aligned_split(
                req, min(req.num_tokens - num_computed, budget), num_computed
            )
            if num_new == 0:
                continue
            if (
                mgr.allocate_slots(
                    req,
                    num_new,
                    num_new_computed_tokens=num_computed,
                    new_computed_blocks=computed_blocks,
                    num_lookahead_tokens=0 if num_computed == 0 else NUM_SPEC,
                )
                is None
            ):
                continue
            if num_computed > 0:
                # a resumed request must load the state of exactly
                # `num_computed` tokens
                pos = num_computed // MAMBA_BS - 1
                blocks = mamba.req_to_blocks[req.request_id]
                if 0 <= pos < len(blocks) and not blocks[pos].is_null:
                    tag = block_tag.get(blocks[pos].block_id)
                    if tag != num_computed:
                        violations.append(
                            f"{req.request_id} resumed at {num_computed} from "
                            f"state@{tag}"
                        )
            req.num_computed_tokens = num_computed
            budget -= num_new
            admitted.append(req)
            scheduled.append((req, num_computed, num_new))
        for req in admitted:
            waiting.remove(req)
            running.append(req)

        # "execute": mirror the GDN kernel + fused postprocess state writes
        for req, c, num_new in scheduled:
            blocks = mamba.req_to_blocks[req.request_id]
            end = c + num_new
            if end <= req.num_tokens:  # prefill chunk
                accepted = 1 if end == req.num_tokens else 0
                num_draft = 0
            else:  # spec decode step, accept everything (worst case)
                accepted = num_new
                num_draft = NUM_SPEC
            run_pos = c + num_new - num_draft
            pos = cdiv(run_pos, MAMBA_BS) - 1
            if 0 <= pos < len(blocks) and not blocks[pos].is_null:
                block_tag[blocks[pos].block_id] = run_pos
            new_committed = run_pos + accepted - 1
            aligned = new_committed // MAMBA_BS * MAMBA_BS
            if accepted > 0 and aligned >= run_pos:
                dpos = aligned // MAMBA_BS - 1
                if 0 <= dpos < len(blocks) and not blocks[dpos].is_null:
                    block_tag[blocks[dpos].block_id] = aligned

            if end <= req.num_tokens:
                req.num_computed_tokens = end
            else:
                for _ in range(accepted):
                    req.append_output_token_ids(9)
                req.num_computed_tokens = req.num_tokens
            req.spec_token_ids = []
            if req.num_output_tokens >= 8:
                finished.add(req.request_id)

        # invariant check on everything that just got hash-cached
        for req in running:
            n_now = mamba.num_cached_block.get(req.request_id, 0)
            n_before = cached_seen.get(req.request_id, 0)
            if n_now > n_before:
                blocks = mamba.req_to_blocks[req.request_id]
                for p in range(n_before, n_now):
                    if (
                        p >= len(blocks)
                        or blocks[p].is_null
                        or blocks[p].block_hash is None
                    ):
                        continue
                    tag = block_tag.get(blocks[p].block_id)
                    want = (p + 1) * MAMBA_BS
                    if tag != want:
                        violations.append(
                            f"{req.request_id} cached mamba pos {p} as "
                            f"state@{want} but block holds state@{tag}"
                        )
                cached_seen[req.request_id] = n_now

    assert not violations, "\n".join(violations)


@pytest.mark.parametrize("use_eagle", [False, True])
@pytest.mark.parametrize("num_external", [1, 368, MAMBA_BS - 1, MAMBA_BS + 17])
def test_unaligned_external_tokens_realign_chunk_ends(use_eagle, num_external):
    """A prefill starting at an unaligned offset must re-align its chunk ends.

    Externally computed tokens (KV connector) need not be mamba-block
    aligned. Rounding the chunk *length* down to a block multiple would then
    keep every subsequent chunk end unaligned for the whole prefill — the
    same mid-block snapshots in aligned slots that poison the prefix cache
    in the budget-fragmentation scenario above. The split must instead align
    the chunk *end position*, recovering boundary alignment on the first
    chunk it schedules.
    """
    prompt_len = 5 * MAMBA_BS + 7
    req = _make_request("ext", list(range(prompt_len)))

    computed = num_external  # unaligned external KV-connector tokens
    chunk_ends: list[int] = []
    # Cycle through fragmenting budgets; the large one guarantees progress.
    budgets = [364, 700, MAMBA_BS + 13, prompt_len]
    for step in range(100):
        if computed >= prompt_len:
            break
        if chunk_ends:
            req.num_computed_tokens = computed
            num_new = _aligned_split(
                req,
                min(prompt_len - computed, budgets[step % len(budgets)]),
                use_eagle=use_eagle,
            )
        else:
            # first schedule: external tokens are not yet part of
            # request.num_computed_tokens
            num_new = _aligned_split(
                req,
                min(prompt_len - computed, budgets[step % len(budgets)]),
                num_external_computed=num_external,
                use_eagle=use_eagle,
            )
        if num_new == 0:
            continue  # deferred to a step with more budget
        computed += num_new
        chunk_ends.append(computed)

    assert computed == prompt_len, f"prefill stalled at {computed}/{prompt_len}"
    unaligned = [end for end in chunk_ends[:-1] if end % MAMBA_BS != 0]
    assert not unaligned, f"intermediate chunk ends not block-aligned: {unaligned}"
