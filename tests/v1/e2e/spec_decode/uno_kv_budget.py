# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV-cache budgets for the Uno e2e tests, importable without a GPU.

The survivor e2e test pins ``num_gpu_blocks_override`` above the floor vLLM's
admission rule enforces and below what the mixed phase's concurrent requests
need. Both e2e budgets are derived here from the pinned model geometry, and
``tests/v1/spec_decode/test_uno_mrv2.py`` checks them against the engine floor
on CPU, so a model-pin, block-size or ``max_model_len`` change fails without a
GB10.

The geometry is pinned as literals so the CPU suite stays hub-free; the e2e
module, which already requires the model, cross-checks the literals against
``AutoConfig`` for ``MODEL_REVISION``.
"""

from collections.abc import Iterable, Sequence
from itertools import combinations

from vllm.utils.math_utils import cdiv

MODEL_ID = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
BLOCK_SIZE = 16
DTYPE_BYTES = 2  # bfloat16

# Qwen/Qwen3-8B at MODEL_REVISION: 36 hidden layers, 8 KV heads, 128 head dim.
QWEN3_NUM_LAYERS = 36
QWEN3_NUM_KV_HEADS = 8
QWEN3_HEAD_DIM = 128

# The survivor engine runs at a shorter context so its admission floor (65
# blocks) sits below the mixed phase's no-prefix-sharing footprint (151
# blocks). At max_model_len=2048 the floor (129 blocks) nearly equals it.
SURVIVOR_MAX_MODEL_LEN = 1024

# Blocks granted above the engine's single-request admission floor. Four blocks
# put the budget at 69 (68 allocatable, see `allocatable_blocks`), which admits
# the four mixed-phase prompts (63 blocks) and still forces preemption for every
# interleaving of the two long peers (see `worst_case_crossing_tokens`). A wider
# margin re-opens the desynchronisation hole this replaced: with the former
# 25% headroom (81 blocks) a peer that lagged its twin by ~285 generated tokens
# let the leader reach its cap inside the pool, so `allocate_slots` never failed
# and the gate never fired.
SURVIVOR_KV_HEADROOM_BLOCKS = 4

# K for the survivor engine. This is the ONE source: the e2e passes it to
# `speculative_config`, the CPU twin builds its scheduler with it, and the
# resident-block arithmetic below reserves it. Uno's `num_lookahead_tokens` is
# exactly K (`VllmConfig.num_lookahead_tokens`), so a running request holds
# slots for its committed tokens plus one sampled token plus K.
SURVIVOR_NUM_SPECULATIVE_TOKENS = 8

# Prefix caching is disabled in the survivor test. The finish peers therefore
# keep generating until their resident sets grow past the 68 allocatable
# blocks. Growing together they cross at 279 generated tokens; with one peer
# stalled at its admission footprint the leader crosses at 551. A 640-token cap
# leaves five block-groups of margin in the worst case, so no peer can finish
# before the scheduler has to preempt one; both crossings are pinned by
# test_uno_mrv2 and printed by each round's geometry probe. The abort peer is
# retired after two tokens and holds only its own prompt blocks while it runs.
SURVIVOR_FINISH_MAX_TOKENS = 640

# Prompt lengths the survivor e2e tokenises at MODEL_REVISION. The e2e measures
# them at runtime and prints them; these literals exist so the CPU suite can
# check the geometry without the hub, and they are only allowed to be what a
# GPU receipt shows. The finish peers are 257 -- eight tokens more than the
# abort peer, which is the per-repeat cost of the two-word peer tag against the
# one-word abort tag, proved by the contract16 (GB10) and r3 (Ampere) receipts,
# which both print `worst_case_crossing_tokens=544` and `growth_blocks=151`.
# The earlier 265 was a copied literal that no receipt ever supported.
SURVIVOR_PROMPT_TOKENS = (171, 257, 257, 249)

# The K-matrix engine keeps upstream's 4096 context on a 2 GiB budget, far above
# its 257-block floor.
MATRIX_MAX_MODEL_LEN = 4096
MATRIX_KV_BUDGET_BYTES = 2 * 1024**3


def qwen3_geometry() -> tuple[int, int, int]:
    """(layers, KV heads, head dim) of the pinned Qwen3-8B as literals."""
    return QWEN3_NUM_LAYERS, QWEN3_NUM_KV_HEADS, QWEN3_HEAD_DIM


def kv_bytes_per_block(dtype_bytes: int = DTYPE_BYTES) -> int:
    """KV bytes held by one ``BLOCK_SIZE``-token block of the pinned model."""
    num_layers, num_kv_heads, head_dim = qwen3_geometry()
    return num_layers * num_kv_heads * head_dim * 2 * dtype_bytes * BLOCK_SIZE


def engine_minimum_kv_bytes(max_model_len: int) -> int:
    """Smallest ``kv_cache_memory_bytes`` that admits one max-len request.

    ``get_kv_cache_configs`` reserves one pool block (the null block) before
    ``_check_enough_kv_cache_memory`` compares the rest with
    ``_max_memory_usage_bytes_from_groups``. For a dense full-attention model
    that need is ``cdiv(max_model_len, block_size)`` blocks of
    ``FullAttentionSpec.max_memory_usage_bytes``, so the budget must cover one
    more block than the request itself. Speculative lookahead does not raise the
    floor: ``FullAttentionSpec`` sizes a full ``max_model_len`` and Uno clamps
    positions and sequence lengths to ``max_model_len``.
    """
    return (cdiv(max_model_len, BLOCK_SIZE) + 1) * kv_bytes_per_block()


def survivor_kv_budget() -> int:
    """Survivor e2e budget: the admission floor plus a fixed block margin.

    ``SURVIVOR_MAX_MODEL_LEN=1024`` gives a floor of 65 blocks and a budget of
    ``65 + SURVIVOR_KV_HEADROOM_BLOCKS`` = 69 blocks; the mixed phase is sized so
    that all four prompts are admitted together but the two long peers cannot
    both reach their cap inside the pool no matter how far apart their
    generation rates drift, which is what forces preemption/recompute.
    """
    floor_blocks = engine_minimum_kv_bytes(SURVIVOR_MAX_MODEL_LEN) // (
        kv_bytes_per_block()
    )
    return (floor_blocks + SURVIVOR_KV_HEADROOM_BLOCKS) * kv_bytes_per_block()


def allocatable_blocks(budget_blocks: int) -> int:
    """Blocks a request can actually be given out of a ``budget_blocks`` pool.

    ``BlockPool`` pops one block off the free queue as the null block, and
    ``get_usage`` divides by ``num_gpu_blocks - 1``. A pool pinned with
    ``num_gpu_blocks_override=69`` therefore hands out 68 blocks, so one free
    block reads as ``1 - 1/68`` = 98.529% and a full pool reads 100.000%. (On
    the historical 81-block geometry the same one-free-block state read
    98.750%, which is 79 of 80; that number belongs to that geometry only.)
    Every survivor inequality is stated against this count so the receipt and
    the arithmetic share one denominator.
    """
    return budget_blocks - 1


def usage_with_free_blocks(budget_blocks: int, free_blocks: int) -> float:
    """The occupancy ``get_kv_cache_usage`` reports for ``free_blocks`` free.

    Lets a receipt say what a printed percentage means in blocks for the pool
    actually pinned, instead of quoting a percentage from another geometry.
    """
    pool = allocatable_blocks(budget_blocks)
    return 1.0 - (free_blocks / pool)


def blocks_for_tokens(tokens: int) -> int:
    """Blocks the allocator rounds a request's ``tokens`` up to."""
    return cdiv(tokens, BLOCK_SIZE)


def resident_blocks(
    prompt_tokens: int,
    generated_tokens: int,
    num_speculative_tokens: int = SURVIVOR_NUM_SPECULATIVE_TOKENS,
) -> int:
    """Blocks a running request holds after ``generated_tokens`` tokens.

    ``allocate_slots`` reserves ``num_computed + num_new + num_lookahead``
    slots, and for Uno ``num_lookahead_tokens`` is K exactly, so the request
    holds its committed tokens plus the sampled token plus K, rounded up to
    blocks. Deriving it from K rather than a bare ``+1`` is what keeps the
    pre-gate honest when K changes: at K=1 a 257-token prompt holds 17 blocks,
    at K=8 it holds 17, and at K=16 it holds 18.
    """
    return blocks_for_tokens(
        prompt_tokens + generated_tokens + 1 + num_speculative_tokens
    )


def min_resident_blocks(
    prompt_tokens: int,
    num_speculative_tokens: int = SURVIVOR_NUM_SPECULATIVE_TOKENS,
) -> int:
    """Blocks an admitted request holds before it has generated anything.

    A request that is merely lagging, rather than preempted or finished, never
    holds less than this while it is running, which is what the worst-case
    crossing charges the slower peer.
    """
    return resident_blocks(prompt_tokens, 0, num_speculative_tokens)


def prompt_token_ids_are_pairwise_content_distinct(
    prompt_token_ids: Sequence[Sequence[int]],
) -> bool:
    """Return whether every prompt pair differs at a shared token position.

    A length-only difference is not enough: when one prompt is a prefix of the
    other, the shared token positions are identical and this returns ``False``.
    """
    return all(
        any(left != right for left, right in zip(first, second))
        for first, second in combinations(prompt_token_ids, 2)
    )


def mixed_admission_blocks(
    prompt_tokens: Sequence[int],
    shared_prefix_tokens: int,
    *,
    prefix_cache_enabled: bool,
) -> int:
    """Blocks resident when every mixed-phase prompt is admitted at once.

    With prefix caching, the warmed shared prefix is stored once; each request
    then adds only its unique suffix blocks plus one block to start decoding.
    Without prefix caching, every prompt owns its rounded prompt blocks plus
    one block to start decoding. This is the pressure the scheduler sees when
    the peers join, so it must stay below the budget or the prompts would wait
    in the queue instead of triggering preemption.
    """
    if not prefix_cache_enabled:
        return sum(min_resident_blocks(tokens) for tokens in prompt_tokens)

    prefix_blocks = blocks_for_tokens(shared_prefix_tokens)
    unique = sum(
        max(0, blocks_for_tokens(tokens) - prefix_blocks) for tokens in prompt_tokens
    )
    return prefix_blocks + unique + len(prompt_tokens)


def mixed_growth_blocks(
    prompt_tokens: Sequence[int],
    max_tokens: Sequence[int],
    shared_prefix_tokens: int,
    *,
    prefix_cache_enabled: bool,
) -> int:
    """Blocks resident if every request reaches its generation cap.

    Without prefix caching, each request contributes its complete rounded
    prompt-plus-generation footprint. With prefix caching, the warmed shared
    prefix is stored once and each request contributes only its unique growth.
    The result is an upper bound while the abort peer is alive, and the sum
    above the budget is what makes a running request's ``allocate_slots`` fail
    and preempt the last running peer.
    """
    if not prefix_cache_enabled:
        return sum(
            resident_blocks(prompt, cap)
            for prompt, cap in zip(prompt_tokens, max_tokens)
        )

    prefix_blocks = blocks_for_tokens(shared_prefix_tokens)
    return prefix_blocks + sum(
        max(0, blocks_for_tokens(prompt + cap) - prefix_blocks)
        for prompt, cap in zip(prompt_tokens, max_tokens)
    )


def token_agreement(
    reference: Sequence[Sequence[int]],
    candidate: Sequence[Sequence[int]],
) -> tuple[int, list[str]]:
    """Per-prompt exact-token agreement, plus where each disagreement starts.

    Returns ``(matched, divergences)`` where ``matched`` counts the prompts
    whose token ids are identical and each divergence reads ``p<i>/t<j>``: the
    prompt index and the first 0-based token position that differs (or the
    length, when one output is a prefix of the other). That is the same
    coordinate the Ampere control receipts use, so a test failure and a lane
    receipt can be compared by eye.
    """
    assert len(reference) == len(candidate), (
        f"output counts differ: {len(reference)} vs {len(candidate)}"
    )
    matched = 0
    divergences: list[str] = []
    for index, (left, right) in enumerate(zip(reference, candidate)):
        left_ids = list(left)
        right_ids = list(right)
        if left_ids == right_ids:
            matched += 1
            continue
        position = len(left_ids)
        for token_index, (left_id, right_id) in enumerate(zip(left_ids, right_ids)):
            if left_id != right_id:
                position = token_index
                break
        else:
            position = min(len(left_ids), len(right_ids))
        divergences.append(f"p{index}/t{position}")
    return matched, divergences


def exact_token_verdict(
    control_matched: int,
    candidate_matched: int,
    total: int,
) -> tuple[bool, str]:
    """Judge an exact-token comparison against the instrument's own noise floor.

    Greedy exact-token equality is only an instrument where the plain engine
    agrees with itself. On an RTX 3090 (sm_86) in graph mode it does not: two
    fresh plain engines on the same config and prompts agreed on 3 of 4 prompts,
    diverging at prompt 2 token 31, and the Uno arms diverged at the same
    prompt and token. A test that demands 4 of 4 from Uno there cannot tell an
    Uno defect from the baseline's own spread.

    So: when the control is perfect the candidate must be perfect too, and
    otherwise the candidate must be no worse than the control. A candidate that
    is worse is a real finding; a candidate that matches the control's
    imperfection is the card, and the authoritative correctness claim for that
    regime belongs to a sampled gate, not to this one.
    """
    if control_matched == total:
        ok = candidate_matched == total
        reason = (
            f"control is exact ({control_matched}/{total}), so the candidate "
            f"must be too; it matched {candidate_matched}/{total}"
        )
    else:
        ok = candidate_matched >= control_matched
        reason = (
            f"control is not exact ({control_matched}/{total}), so exact-token "
            "equality is not an instrument in this regime; the candidate must "
            f"at least match the control and it matched {candidate_matched}/"
            f"{total}"
        )
    return ok, reason


def resolve_internal_request_ids(
    scheduler_keys: Iterable[str],
    external_ids: Sequence[str],
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Map external request ids to the ids the scheduler is keyed by.

    ``InputProcessor.assign_request_id`` replaces the id a caller passes to
    ``add_request`` with ``f"{external}-{random_uuid():.8}"`` unless
    ``VLLM_DISABLE_REQUEST_ID_RANDOMIZATION`` is set, and ``Request`` keeps only
    that internal id. Looking a request up by the id the test supplied
    therefore returns ``None`` on every step -- which is why two GPU runs
    reported an empty per-request receipt while the scheduler's own pool
    reported 100% occupancy.

    Returns ``(resolved, problems)``: ``resolved`` maps each external id whose
    internal id was unambiguous, ``problems`` maps the rest to the candidates
    found, so a caller can fail with the ids the scheduler actually holds.
    """
    keys = list(scheduler_keys)
    resolved: dict[str, str] = {}
    problems: dict[str, list[str]] = {}
    for external in external_ids:
        candidates = [key for key in keys if _is_internal_id(key, external)]
        if len(candidates) == 1:
            resolved[external] = candidates[0]
        else:
            problems[external] = candidates
    return resolved, problems


def _is_internal_id(key: str, external: str) -> bool:
    """Whether ``key`` is the scheduler's id for the request ``external``."""
    if key == external:
        # VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1.
        return True
    if not key.startswith(f"{external}-"):
        return False
    suffix = key[len(external) + 1 :]
    # `random_uuid():.8` is eight characters of a uuid4 string.
    return len(suffix) == 8 and suffix.isalnum()


def mid_generation_preemption_counts(
    events: Sequence[tuple[str, int]],
) -> dict[str, int]:
    """Count only the preemptions that interrupted a request mid-generation.

    ``events`` are ``(request_id, generated_tokens_at_preemption)`` pairs. A
    request preempted while its prompt is still being chunked has produced no
    tokens, and its recompute re-runs a prefill the gate is not about: only a
    preemption at one or more generated tokens exercises the resume path the
    survivor claim covers.
    """
    counts: dict[str, int] = {}
    for request_id, generated_tokens in events:
        if generated_tokens > 0:
            counts[request_id] = counts.get(request_id, 0) + 1
    return counts


def worst_case_crossing_tokens(
    peer_prompt_tokens: int,
    other_peer_prompt_tokens: Sequence[int],
    allocatable: int,
    num_speculative_tokens: int = SURVIVOR_NUM_SPECULATIVE_TOKENS,
) -> int:
    """Generated tokens at which ONE peer alone must outgrow the pool.

    ``mixed_crossing_tokens`` assumes the long peers grow together. They do not:
    Uno's acceptance is prompt-dependent, so one peer can run several times
    faster than its twin, and a pair that only crosses *together* never crosses
    at all when the leader reaches its cap while the laggard still sits near its
    admission footprint. This is the counted version of that worst case -- the
    leader's own footprint plus the minimum resident footprint of every other
    live long peer -- and the survivor e2e asserts it is below the cap, so the
    scheduler must preempt for every interleaving rather than for the lucky
    ones.
    """
    held = sum(
        min_resident_blocks(tokens, num_speculative_tokens)
        for tokens in other_peer_prompt_tokens
    )
    tokens = 0
    while True:
        leader = resident_blocks(peer_prompt_tokens, tokens, num_speculative_tokens)
        if leader + held > allocatable:
            return tokens
        tokens += 1


def mixed_crossing_tokens(
    peer_prompt_tokens: int,
    shared_prefix_tokens: int,
    capacity_blocks: int,
    *,
    prefix_cache_enabled: bool,
) -> int:
    """Generated tokens at which the two long peers TOGETHER outgrow the pool.

    ``capacity_blocks`` is the allocatable count, not the pinned override (see
    ``allocatable_blocks``). By the time the pair has grown this far the short
    seed has finished and the abort peer is retired. Without prefix caching,
    each peer owns its complete prompt-plus-generation footprint; with prefix
    caching, the warmed prefix is stored once and each peer contributes only its
    unique growth. This is the *best* case for the gate and is reported for
    continuity; the assertion that makes the gate fire is
    ``worst_case_crossing_tokens``, which does not assume equal rates.
    """
    prefix_blocks = blocks_for_tokens(shared_prefix_tokens)
    tokens = 0
    while True:
        if prefix_cache_enabled:
            unique = max(
                0, blocks_for_tokens(peer_prompt_tokens + tokens) - prefix_blocks
            )
            resident = prefix_blocks + 2 * unique
        else:
            resident = 2 * resident_blocks(peer_prompt_tokens, tokens)
        if resident > capacity_blocks:
            return tokens
        tokens += 1
