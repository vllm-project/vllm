# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mamba "align" prefix-cache chunk splitting (_mamba_block_aligned_split).

In mamba_cache_mode="align", MambaManager.cache_blocks hashes a request's
block-table slot p as the recurrent state at exactly (p + 1) * block_size
tokens, so every non-final prefill chunk must end on a mamba block boundary
or the cached snapshot describes the wrong token offset (#43559).

The main test here drives the real KVCacheManager + the real chunk-split
helper through a multi-request simulation on CPU, mirroring the model
runner's state writes (`_mock_execute_model`), and asserts that invariant
for every hash-cached mamba block.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.request import Request

from .utils import create_requests

pytestmark = pytest.mark.cpu_test

# Mirrors the deployment where the poisoning was observed (Qwen3.6-27B):
# mamba block 1600, MTP with 3 draft tokens, 16k-token scheduler budget,
# prompts shorter than 2 * mamba block size.
ATTN_BLOCK_SIZE = 16
MAMBA_BLOCK_SIZE = 1600
NUM_SPEC = 3
TOKEN_BUDGET = 16384
PROMPT_LEN = 2002

# Oracle mapping a physical mamba block id to the token offset of the
# recurrent state it currently holds (what a correct kernel would have
# written there).
BlockTag = dict[int, int]


def _make_hybrid_kv_cache_manager() -> KVCacheManager:
    cfg = KVCacheConfig(
        num_blocks=10000,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full_layer"],
                FullAttentionSpec(
                    block_size=ATTN_BLOCK_SIZE,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba_layer"],
                MambaSpec(
                    block_size=MAMBA_BLOCK_SIZE,
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
        max_model_len=262144,
        scheduler_block_size=MAMBA_BLOCK_SIZE,
        hash_block_size=ATTN_BLOCK_SIZE,
        max_num_batched_tokens=TOKEN_BUDGET,
        enable_caching=True,
        use_eagle=True,
    )


def _mamba_aligned_split(
    req: Request,
    num_new: int,
    num_new_local_computed: int = 0,
    num_external_computed: int = 0,
    use_eagle: bool = True,
) -> int:
    """Call the real Scheduler._mamba_block_aligned_split on a stub self."""
    stub = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=MAMBA_BLOCK_SIZE),
        use_eagle=use_eagle,  # MTP / EAGLE-family speculative decoding
    )
    return Scheduler._mamba_block_aligned_split(
        stub, req, num_new, num_new_local_computed, num_external_computed
    )


def _schedule_running(
    mgr: KVCacheManager,
    running: list[Request],
    finished: set[str],
    budget: int,
    scheduled: list[tuple[Request, int, int]],
) -> int:
    """Schedule one step for the running requests, RUNNING-queue style.

    Requests mid-prefill get a budget-capped chunk through the real split
    helper; finished-prefill requests get a spec-decode step (1 token +
    NUM_SPEC drafts). Returns the remaining token budget.
    """
    for req in running:
        if req.request_id in finished or budget <= 0:
            continue
        c = req.num_computed_tokens
        if c < req.num_tokens:
            num_new = _mamba_aligned_split(req, min(req.num_tokens - c, budget))
            if num_new == 0:
                continue
            lookahead = NUM_SPEC
        else:
            req.spec_token_ids = [7] * NUM_SPEC
            num_new = 1 + NUM_SPEC
            lookahead = 0
        assert (
            mgr.allocate_slots(req, num_new, num_lookahead_tokens=lookahead) is not None
        )
        budget -= num_new
        scheduled.append((req, c, num_new))
    return budget


def _admit_waiting(
    mgr: KVCacheManager,
    waiting: list[Request],
    running: list[Request],
    budget: int,
    scheduled: list[tuple[Request, int, int]],
    block_tag: BlockTag,
    violations: list[str],
) -> int:
    """Admit waiting requests, taking prefix-cache hits through the manager.

    A request resuming from a cached prefix must load the recurrent state of
    exactly `num_computed` tokens; anything else is a poisoned-cache hit and
    is recorded in `violations`. Returns the remaining token budget.
    """
    mamba = mgr.coordinator.single_type_managers[1]
    admitted = []
    for req in waiting:
        if budget <= 0:
            break
        computed_blocks, num_computed = mgr.get_computed_blocks(req)
        num_new = _mamba_aligned_split(
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
            pos = num_computed // MAMBA_BLOCK_SIZE - 1
            blocks = mamba.req_to_blocks[req.request_id]
            if 0 <= pos < len(blocks) and not blocks[pos].is_null:
                tag = block_tag.get(blocks[pos].block_id)
                if tag != num_computed:
                    violations.append(
                        f"{req.request_id} resumed at {num_computed} from state@{tag}"
                    )
        req.num_computed_tokens = num_computed
        budget -= num_new
        admitted.append(req)
        scheduled.append((req, num_computed, num_new))
    for req in admitted:
        waiting.remove(req)
        running.append(req)
    return budget


def _mock_execute_model(
    mgr: KVCacheManager,
    scheduled: list[tuple[Request, int, int]],
    block_tag: BlockTag,
    finished: set[str],
) -> None:
    """Mirror the GDN kernel + fused postprocess state writes for one step.

    For each scheduled (request, chunk_start, num_new) the kernel runs to
    `run_pos` (chunk end, minus rejected drafts on decode) and writes the
    state into the block-table slot covering that position; on decode it
    additionally snapshots the last crossed block boundary. `block_tag`
    records which token offset each physical block's state belongs to.
    Spec decode accepts every draft (worst case for boundary snapshots).
    """
    mamba = mgr.coordinator.single_type_managers[1]
    for req, c, num_new in scheduled:
        blocks = mamba.req_to_blocks[req.request_id]
        end = c + num_new
        if end <= req.num_tokens:  # prefill chunk
            accepted = 1 if end == req.num_tokens else 0
            num_draft = 0
        else:  # spec decode step
            accepted = num_new
            num_draft = NUM_SPEC
        run_pos = c + num_new - num_draft
        pos = cdiv(run_pos, MAMBA_BLOCK_SIZE) - 1
        if 0 <= pos < len(blocks) and not blocks[pos].is_null:
            block_tag[blocks[pos].block_id] = run_pos
        new_committed = run_pos + accepted - 1
        aligned = new_committed // MAMBA_BLOCK_SIZE * MAMBA_BLOCK_SIZE
        if accepted > 0 and aligned >= run_pos:
            dpos = aligned // MAMBA_BLOCK_SIZE - 1
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


def _check_cached_mamba_states(
    mgr: KVCacheManager,
    running: list[Request],
    block_tag: BlockTag,
    cached_seen: dict[str, int],
    violations: list[str],
) -> None:
    """Check every newly hash-cached mamba block against the oracle.

    The invariant: a hash-cached mamba block at position p holds the
    recurrent state of exactly (p + 1) * block_size tokens.
    """
    mamba = mgr.coordinator.single_type_managers[1]
    for req in running:
        n_now = mamba.num_cached_block.get(req.request_id, 0)
        n_before = cached_seen.get(req.request_id, 0)
        if n_now <= n_before:
            continue
        blocks = mamba.req_to_blocks[req.request_id]
        for p in range(n_before, n_now):
            if p >= len(blocks) or blocks[p].is_null or blocks[p].block_hash is None:
                continue
            tag = block_tag.get(blocks[p].block_id)
            want = (p + 1) * MAMBA_BLOCK_SIZE
            if tag != want:
                violations.append(
                    f"{req.request_id} cached mamba pos {p} as "
                    f"state@{want} but block holds state@{tag}"
                )
        cached_seen[req.request_id] = n_now


def test_fragmented_prefill_chunk_does_not_poison_mamba_prefix_cache():
    """Budget-fragmented prefill chunks must not poison the prefix cache.

    With speculative decoding (use_eagle=True), _mamba_block_aligned_split
    zeroes its alignment target for prompts shorter than 2 * mamba_block_size
    (`last_cache_position = max(last - block_size, 0)`), so a prefill whose
    first chunk is capped by leftover token budget (concurrent prefills
    sharing a step) used to end at an arbitrary offset. The GDN kernel then
    writes that mid-block state into the request's position-0 mamba slot, and
    MambaManager.cache_blocks later hashes that slot as the state at
    `block_size` tokens — poisoning the prefix cache for every request that
    resumes from it. Fails on the unfixed v0.22.1 scheduler.
    """
    mgr = _make_hybrid_kv_cache_manager()
    waiting = create_requests(
        10,
        num_tokens=PROMPT_LEN,
        max_tokens=64,
        same_prompt=True,
        block_size=ATTN_BLOCK_SIZE,
    )
    running: list[Request] = []
    finished: set[str] = set()

    block_tag: BlockTag = {}
    cached_seen: dict[str, int] = {}
    violations: list[str] = []

    for _step in range(400):
        if not waiting and all(r.request_id in finished for r in running):
            break
        scheduled: list[tuple[Request, int, int]] = []
        budget = _schedule_running(mgr, running, finished, TOKEN_BUDGET, scheduled)
        _admit_waiting(mgr, waiting, running, budget, scheduled, block_tag, violations)
        _mock_execute_model(mgr, scheduled, block_tag, finished)
        _check_cached_mamba_states(mgr, running, block_tag, cached_seen, violations)

    assert not violations, "\n".join(violations)


@pytest.mark.parametrize("use_eagle", [False, True])
@pytest.mark.parametrize(
    "num_external", [1, 368, MAMBA_BLOCK_SIZE - 1, MAMBA_BLOCK_SIZE + 17]
)
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
    prompt_len = 5 * MAMBA_BLOCK_SIZE + 7
    req = create_requests(1, num_tokens=prompt_len, max_tokens=64)[0]

    computed = num_external  # unaligned external KV-connector tokens
    chunk_ends: list[int] = []
    # Cycle through fragmenting budgets; the large one guarantees progress.
    budgets = [364, 700, MAMBA_BLOCK_SIZE + 13, prompt_len]
    for step in range(100):
        if computed >= prompt_len:
            break
        # External tokens are folded into request.num_computed_tokens after
        # the first scheduled chunk.
        first = not chunk_ends
        req.num_computed_tokens = 0 if first else computed
        num_new = _mamba_aligned_split(
            req,
            min(prompt_len - computed, budgets[step % len(budgets)]),
            num_external_computed=num_external if first else 0,
            use_eagle=use_eagle,
        )
        if num_new == 0:
            continue  # deferred to a step with more budget
        computed += num_new
        chunk_ends.append(computed)

    assert computed == prompt_len, f"prefill stalled at {computed}/{prompt_len}"
    unaligned = [end for end in chunk_ends[:-1] if end % MAMBA_BLOCK_SIZE != 0]
    assert not unaligned, f"intermediate chunk ends not block-aligned: {unaligned}"
