# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Greedy Uno parity with the original Qwen3-8B adapter on one NVIDIA GPU."""

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import pytest
import torch
from huggingface_hub.constants import HF_HUB_OFFLINE
from transformers import AutoConfig, AutoTokenizer

import vllm.envs as envs
from vllm import SamplingParams
from vllm.platforms import current_platform
from vllm.transformers_utils.repo_utils import hf_api
from vllm.v1.metrics.reader import Counter, Histogram, Metric

from .uno_kv_budget import (
    BLOCK_SIZE,
    MATRIX_KV_BUDGET_BYTES,
    MATRIX_MAX_MODEL_LEN,
    MODEL_ID,
    MODEL_REVISION,
    SURVIVOR_FINISH_MAX_TOKENS,
    SURVIVOR_MAX_MODEL_LEN,
    allocatable_blocks,
    engine_minimum_kv_bytes,
    kv_bytes_per_block,
    mixed_admission_blocks,
    mixed_crossing_tokens,
    mixed_growth_blocks,
    prompt_token_ids_are_pairwise_content_distinct,
    qwen3_geometry,
    survivor_kv_budget,
    worst_case_crossing_tokens,
)
from .utils import (
    assert_request_outputs_match,
    compute_acceptance_len,
    get_spec_decode_metric_value,
    get_test_prompts,
)

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Uno requires an NVIDIA CUDA device"
)


# Eight repeats give each peer ~104 body tokens beyond the shared prefix: enough
# to stay tag-unique and cache-distinct, small enough that all four prompts are
# admitted together (mixed_admission_blocks below the allocatable pool) before
# their generations grow the resident footprint past it.
_PEER_REPEATS = 8


def _peer_body(tag: str, repeats: int = _PEER_REPEATS) -> str:
    """A tag-unique continuation so concurrent peers do not share blocks."""
    return f"{tag} carries the mill wheel past the old stone bridge. " * repeats


def _token_count(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _auto_config_geometry() -> tuple[int, int, int]:
    """The pinned model's geometry read from the hub.

    ``uno_kv_budget`` keeps the geometry as literals so the CPU suite stays
    hub-free; this e2e module already requires the model, so it cross-checks
    the literals against the real config for the pinned revision.
    """
    config = AutoConfig.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=HF_HUB_OFFLINE
    )
    text_config = getattr(config, "text_config", config)
    return (
        text_config.num_hidden_layers,
        getattr(text_config, "num_key_value_heads", text_config.num_attention_heads),
        getattr(
            text_config,
            "head_dim",
            text_config.hidden_size // text_config.num_attention_heads,
        ),
    )


def _uno_execution_state(worker) -> dict:
    from vllm.v1.worker.gpu.spec_decode.uno import UnoSpeculator

    runner = worker.model_runner
    proposer = runner.speculator
    assert isinstance(proposer, UnoSpeculator)
    return {
        "shared_model": proposer.model is runner.model,
        "draft_graph_replays": proposer.num_graph_replays,
        "draft_graphs": len(proposer.cudagraph_manager.graphs),
        "draft_eager_proposals": proposer.num_eager_proposals,
        "lora_plan_hits": runner.uno_lora_state.plan_cache.hits,
        "lora_plan_misses": runner.uno_lora_state.plan_cache.misses,
        "lora_plan_bypasses": runner.uno_lora_state.plan_cache.bypasses,
    }


def _disable_uno_adapter_for_control(worker) -> None:
    """Use identical noise and sampling, with base weights on every draft row."""
    runner = worker.model_runner
    proposer = runner.speculator

    def base_only(shape):
        if shape is not None:
            _num_reqs, num_tokens = shape
            runner.uno_lora_state.install_base(num_tokens, num_tokens)

    proposer.set_lora_hook(base_only)


@pytest.fixture(scope="module")
def uno_adapter_path() -> str:
    snapshot = hf_api().snapshot_download(
        repo_id="s-sahoo/uno-qwen3-8B",
        revision="8819e09ac901e7290d8d89d62c98b9f756c602fe",
        allow_patterns=["adapter/*"],
        local_files_only=HF_HUB_OFFLINE,
    )
    return str(Path(snapshot) / "adapter")


def _counter_total(metrics: list[Metric], name: str) -> float:
    return sum(
        metric.value
        for metric in metrics
        if isinstance(metric, Counter) and metric.name == name
    )


def _histogram_total(metrics: list[Metric], name: str) -> float:
    return sum(
        metric.sum
        for metric in metrics
        if isinstance(metric, Histogram) and metric.name == name
    )


def _run_request_to_finish(
    engine,
    request_id: str,
    prompt: str,
    sampling: SamplingParams,
) -> tuple[int, ...]:
    engine.add_request(request_id, prompt, sampling)
    final = None
    while engine.has_unfinished_requests():
        for output in engine.step():
            if output.request_id == request_id and output.finished:
                final = output
    assert final is not None and final.outputs, (
        f"request {request_id} produced no finished output"
    )
    return tuple(final.outputs[0].token_ids)


def _scheduler(engine):
    """Return the scheduler used by the in-process offline engine.

    The per-request preemption receipt is not exposed by ``RequestOutput`` or
    the output processor, so this e2e test intentionally uses V1 in-process
    mode. Multiprocess mode has no scheduler object in the client process.
    """
    client = getattr(engine, "engine_core", None)
    core = getattr(client, "engine_core", None)
    scheduler = getattr(core, "scheduler", None)
    if scheduler is None:
        client_name = type(client).__name__ if client is not None else "<missing>"
        mode = (
            "multiprocess"
            if client_name in {"SyncMPClient", "AsyncMPClient"}
            else "unknown"
        )
        raise AssertionError(
            "survivor scheduler receipt requires V1 in-process mode "
            "(VLLM_ENABLE_V1_MULTIPROCESSING=0); got "
            f"{client_name} in {mode} mode"
        )
    return scheduler


@dataclass
class _MixedPhase:
    """What the mixed phase produced, including the preemption receipt."""

    # Final greedy token IDs per finish peer, keyed by request id.
    finished: dict[str, tuple[int, ...]]
    # How many times each finish peer was preempted while still generating.
    preempted_while_active: dict[str, int]
    # One line per observed preemption: which request, at which token, while
    # which requests were running/waiting. Printed so a GPU run can be read.
    receipts: list[str]
    # Highest scheduler-reported KV occupancy observed during the mixed phase.
    peak_kv_cache_usage: float
    # Generated-token count per finish peer at every step, reduced to the
    # largest gap ever seen between the fastest and the slowest peer. This is
    # the receipt for the failure class the round-9 redesign closed: peers that
    # drift apart never hold their footprints at the same time, so a
    # grow-together budget is never crossed (see `worst_case_crossing_tokens`).
    max_generation_lag: int
    # Generated-token count per finish peer when it finished.
    final_lengths: dict[str, int]


def _run_survivor_with_peers(
    engine,
    *,
    seed_prompt: str,
    finish_prompts: list[str],
    abort_prompt: str,
    seed_sampling: SamplingParams,
    finish_sampling: SamplingParams,
    abort_sampling: SamplingParams,
    inject_after: int = 4,
) -> _MixedPhase:
    """Drive a seed while long peers join, finish and abort mid-stream.

    The engine loop is driven step by step so peers are injected only after the
    seed has already produced tokens, which is what makes this continuous
    batching rather than a static batch. Per-request preemption is read from the
    scheduler at the step it happens, so a finish peer that is preempted while
    still generating is recorded even though it only finishes after resuming;
    the global preemption counter, read after every request finished, cannot
    tell those two cases apart.
    """
    seed_id = "uno-seed"
    abort_id = "uno-abort-peer"
    finish_ids = [f"uno-finish-peer-{index}" for index in range(len(finish_prompts))]
    finish_id_set = set(finish_ids)
    engine.add_request(seed_id, seed_prompt, seed_sampling)
    scheduler = _scheduler(engine)
    injected = False
    aborted = False
    seed_len = 0
    seed_final = False
    abort_len = 0
    finished: dict[str, tuple[int, ...]] = {}
    preempted_while_active: dict[str, int] = {}
    receipts: list[str] = []
    peak_kv_cache_usage = 0.0
    max_generation_lag = 0
    seen_preemptions: dict[str, int] = {}
    while engine.has_unfinished_requests():
        outputs = engine.step()
        peak_kv_cache_usage = max(peak_kv_cache_usage, scheduler.get_kv_cache_usage())
        live_lengths = [
            len(request.output_token_ids)
            for request in (scheduler.requests.get(rid) for rid in finish_ids)
            if request is not None
        ]
        if len(live_lengths) == len(finish_ids):
            max_generation_lag = max(
                max_generation_lag, max(live_lengths) - min(live_lengths)
            )
        for output in outputs:
            if output.request_id == seed_id:
                if output.outputs:
                    seed_len = max(seed_len, len(output.outputs[0].token_ids))
                if output.finished:
                    seed_final = True
            elif output.request_id in finish_id_set:
                if output.finished and output.outputs:
                    finished[output.request_id] = tuple(output.outputs[0].token_ids)
            elif output.request_id == abort_id and output.outputs:
                abort_len = max(abort_len, len(output.outputs[0].token_ids))

        if not injected and seed_len >= inject_after:
            for index, prompt in enumerate(finish_prompts):
                engine.add_request(f"uno-finish-peer-{index}", prompt, finish_sampling)
            engine.add_request(abort_id, abort_prompt, abort_sampling)
            injected = True

        # Always retire the aborting peer so the loop terminates even if it was
        # never scheduled long enough to reach the threshold.
        if (
            injected
            and not aborted
            and (abort_len >= 2 or (seed_final and abort_len > 0))
        ):
            engine.abort_request([abort_id])
            aborted = True

        for request_id in finish_ids:
            request = scheduler.requests.get(request_id)
            if request is None:
                continue
            if request.num_preemptions <= seen_preemptions.get(request_id, 0):
                continue
            seen_preemptions[request_id] = request.num_preemptions
            # A request can only be preempted while it is running, so an
            # increment before its finished output was delivered is exactly the
            # "preempted while active" receipt this gate needs.
            if request_id not in finished:
                preempted_while_active[request_id] = (
                    preempted_while_active.get(request_id, 0) + 1
                )
            receipts.append(
                f"{request_id} preempted at {len(request.output_token_ids)} "
                f"generated tokens of {finish_sampling.max_tokens}; "
                f"running={sorted(r.request_id for r in scheduler.running)}, "
                f"waiting={sorted(r.request_id for r in scheduler.waiting)}"
            )

    assert injected, "peers never joined the seed mid-stream"
    assert aborted, "the aborting peer was never retired mid-stream"
    assert all(request_id in finished for request_id in finish_ids), (
        f"not every finish peer produced a finished output; finished={sorted(finished)}"
    )
    return _MixedPhase(
        finished=finished,
        preempted_while_active=preempted_while_active,
        receipts=receipts,
        peak_kv_cache_usage=peak_kv_cache_usage,
        max_generation_lag=max_generation_lag,
        final_lengths={
            request_id: len(tokens) for request_id, tokens in finished.items()
        },
    )


@pytest.mark.forked
@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="Uno survivor batching requires an NVIDIA CUDA device",
)
def test_uno_continuous_batching_survivor_matches_solo(
    vllm_runner,
    monkeypatch: pytest.MonkeyPatch,
    uno_adapter_path: str,
):
    """A peer preempted while active keeps its tokens when it resumes.

    Forces preemption/recompute by growth: all four prompts fit the KV budget
    together, then the two long finish peers generate long sequences so the
    running footprint exceeds it and the scheduler preempts a running peer. The
    compared request is one of the two long peers, and its preemption is read
    from the scheduler at the step it happens, so the token-equality gate can
    only pass for a request that was still generating (the short seed finishes
    before the pool is crossed and the abort peer is retired early). Each
    preempted peer's greedy token IDs are compared against the same prompt run
    alone under batch-invariant settings.

    The geometry is sized so the crossing happens for every interleaving of the
    two peers, not only when they grow at the same rate, so a run that observes
    no preemption FAILS with its receipt instead of skipping.
    """
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setattr(envs, "VLLM_USE_V2_MODEL_RUNNER", True)
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setattr(envs, "VLLM_ENABLE_V1_MULTIPROCESSING", False)
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    shared_prefix = "The quick brown fox jumps over the lazy dog. " * 16
    seed_suffix = " Now count from one to twenty slowly and carefully."
    seed_prompt = shared_prefix + seed_suffix
    finish_prompts = [
        shared_prefix + _peer_body("Peer Alpha"),
        shared_prefix + _peer_body("Peer Bravo"),
    ]
    abort_prompt = shared_prefix + _peer_body("Abort")

    seed_sampling = SamplingParams(
        temperature=0, max_tokens=96, ignore_eos=True, seed=0
    )
    finish_sampling = SamplingParams(
        # Long enough that the peers' generation growth, not their prompt size,
        # is what exhausts the KV budget and forces preemption. With prefix
        # caching disabled, the two long peers cross the 68 allocatable blocks
        # at 280 generated tokens growing together, and at 536 when one of them
        # stalls at its admission footprint -- both below the 640 cap, so a peer
        # cannot finish naturally before the scheduler preempts one whatever
        # their relative acceptance rates are.
        temperature=0,
        max_tokens=SURVIVOR_FINISH_MAX_TOKENS,
        ignore_eos=True,
        seed=0,
    )
    abort_sampling = SamplingParams(
        # Below the context-window margin and the seed's own cap: the peer is
        # retired after two tokens, so this is never a natural finish.
        temperature=0,
        max_tokens=64,
        ignore_eos=True,
        seed=0,
    )

    # Preemption is forced by GROWTH, not by prompt size. Prefix caching is off,
    # so all four prompts own their prompt blocks and are admitted together
    # (mixed_admission_blocks) below the 68 allocatable blocks. The two finish
    # peers then generate SURVIVOR_FINISH_MAX_TOKENS tokens each, so their
    # independent growth (mixed_growth_blocks) passes the pool. The scheduler's
    # running loop preempts the last running request when allocate_slots fails;
    # the short seed finishes before the crossing and the abort peer is retired,
    # so the preempted request is one of the two long peers and its resumed
    # tokens are the correctness gate.
    #
    # The crossing must not depend on the two peers growing at the same rate.
    # Uno's acceptance is prompt-dependent, and a peer that lags its twin far
    # enough lets the leader reach its cap inside the pool: the pair's
    # footprints are then never resident together, allocate_slots never fails
    # and the gate cannot fire (that is exactly how the 81-block/512-token
    # configuration skipped on two cards). So the load-bearing pre-gate is
    # `worst_case_crossing_tokens`, the crossing with every other long peer
    # stalled at its admission footprint. All four inequalities are asserted
    # with their counts here and pinned on CPU by test_uno_mrv2.py and
    # tests/v1/spec_decode/test_uno_preemption.py, which drives the real
    # scheduler over this geometry for several acceptance ratios.
    max_model_len = SURVIVOR_MAX_MODEL_LEN
    kv_cache_budget_bytes = survivor_kv_budget()
    budget_blocks = kv_cache_budget_bytes // kv_bytes_per_block()
    # One block of the pinned pool is the pool's null block and is never handed
    # to a request; `get_kv_cache_usage` also divides by this number, so the
    # printed peak and these inequalities share one denominator.
    pool_blocks = allocatable_blocks(budget_blocks)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, local_files_only=HF_HUB_OFFLINE
    )
    assert qwen3_geometry() == _auto_config_geometry(), (
        "uno_kv_budget's literal Qwen3-8B geometry no longer matches "
        f"AutoConfig at {MODEL_REVISION}"
    )
    prompts = [seed_prompt, *finish_prompts, abort_prompt]
    max_tokens = [
        seed_sampling.max_tokens,
        finish_sampling.max_tokens,
        finish_sampling.max_tokens,
        abort_sampling.max_tokens,
    ]
    prompt_token_ids = [
        tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompts
    ]
    prompt_tokens = [len(token_ids) for token_ids in prompt_token_ids]
    prompt_labels = ("seed", "finish-peer-0", "finish-peer-1", "abort-peer")
    prompt_distinctness = {
        f"{prompt_labels[left]}!={prompt_labels[right]}": (
            prompt_token_ids_are_pairwise_content_distinct(
                [prompt_token_ids[left], prompt_token_ids[right]]
            )
        )
        for left, right in combinations(range(len(prompt_labels)), 2)
    }
    shared_prefix_tokens = _token_count(tokenizer, shared_prefix)
    assert all(prompt_distinctness.values()), (
        "mixed-phase prompts must differ in token content, not only length: "
        f"{prompt_distinctness}"
    )
    admission_blocks = mixed_admission_blocks(
        prompt_tokens, shared_prefix_tokens, prefix_cache_enabled=False
    )
    growth_blocks = mixed_growth_blocks(
        prompt_tokens,
        max_tokens,
        shared_prefix_tokens,
        prefix_cache_enabled=False,
    )
    crossing_tokens = mixed_crossing_tokens(
        max(prompt_tokens[1:3]),
        shared_prefix_tokens,
        pool_blocks,
        prefix_cache_enabled=False,
    )
    worst_case_crossing = worst_case_crossing_tokens(
        max(prompt_tokens[1:3]),
        [min(prompt_tokens[1:3])],
        pool_blocks,
    )
    kv_floor = engine_minimum_kv_bytes(max_model_len)
    kv_floor_blocks = kv_floor // kv_bytes_per_block()
    assert kv_floor <= kv_cache_budget_bytes, (
        "the survivor KV budget is below vLLM's single-request floor: floor "
        f"{kv_floor} B ({kv_floor // kv_bytes_per_block()} blocks) > budget "
        f"{kv_cache_budget_bytes} B ({budget_blocks} blocks) at "
        f"max_model_len={max_model_len}"
    )
    assert admission_blocks < pool_blocks, (
        "the four prompts do not fit the KV pool together, so the peers would "
        "wait instead of growing into preemption: unique prompt footprint plus "
        f"one decode block each is {admission_blocks} blocks, the pool holds "
        f"{pool_blocks} allocatable blocks of the pinned {budget_blocks} "
        f"(prompt tokens {prompt_tokens}, shared prefix "
        f"{shared_prefix_tokens} tokens)"
    )
    assert growth_blocks > pool_blocks, (
        "the mixed phase cannot exhaust the KV pool by growth: if every "
        f"running request reached its cap the footprint is {growth_blocks} "
        f"blocks, the pool holds {pool_blocks} allocatable blocks (prompt "
        f"tokens {prompt_tokens}, max_tokens {max_tokens})"
    )
    # The load-bearing pre-gate: even with one long peer stalled at its
    # admission footprint, the other must outgrow the pool before it can reach
    # its cap. Without this the gate only fires when the peers happen to grow
    # at the same rate, which is what let two cards skip with every prompt
    # distinct and the pool one block short of full.
    assert worst_case_crossing < finish_sampling.max_tokens, (
        "the survivor geometry cannot force preemption for every interleaving: "
        f"one long peer plus the other's admission footprint crosses the "
        f"{pool_blocks}-block pool only after {worst_case_crossing} generated "
        f"tokens, at or past the {finish_sampling.max_tokens} cap, so a peer "
        "that outruns its twin can finish inside the pool and the resume path "
        "is never exercised. Raise SURVIVOR_FINISH_MAX_TOKENS or lower "
        "SURVIVOR_KV_HEADROOM_BLOCKS rather than accepting a non-firing run"
    )
    # The pair growing together must also cross well before the cap; this is
    # the best case and is reported for continuity with the earlier receipts.
    assert crossing_tokens < finish_sampling.max_tokens, (
        "the two long peers together cross the KV pool only after "
        f"{crossing_tokens} generated tokens, at or past the "
        f"{finish_sampling.max_tokens} cap, so a peer could finish before the "
        "scheduler ever preempts it"
    )
    # The prompts stay inside the context window; `_validate_prompt_len` raises
    # VLLMValidationError at `add_request` rather than clamping.
    for label, prompt_len, cap in zip(
        prompt_labels,
        prompt_tokens,
        max_tokens,
    ):
        assert prompt_len + cap < max_model_len, (
            f"{label} has {prompt_len} prompt tokens plus max_tokens={cap}, "
            f"which does not fit max_model_len={max_model_len}; "
            "_validate_prompt_len raises rather than clamping"
        )

    print(f"survivor prompt token-id distinctness: {prompt_distinctness}")
    print(
        "survivor KV arithmetic: prefix_cache=False, "
        f"floor_blocks={kv_floor_blocks}, budget_blocks={budget_blocks}, "
        f"pool_blocks={pool_blocks}, admission_blocks={admission_blocks}, "
        f"growth_blocks={growth_blocks}, crossing_tokens={crossing_tokens}, "
        f"worst_case_crossing_tokens={worst_case_crossing}, "
        f"prompt_tokens={prompt_tokens}, max_tokens={max_tokens}"
    )

    common = dict(
        revision=MODEL_REVISION,
        dtype="bfloat16",
        trust_remote_code=False,
        enforce_eager=True,
        compilation_config={"cudagraph_capture_sizes": [16]},
        async_scheduling=True,
        attention_config={"backend": "FLASH_ATTN", "flash_attn_version": 2},
        max_model_len=max_model_len,
        max_num_seqs=4,
        max_num_batched_tokens=256,
        enable_chunked_prefill=True,
        block_size=BLOCK_SIZE,
        enable_prefix_caching=False,
        num_gpu_blocks_override=budget_blocks,
        enable_lora=True,
        max_lora_rank=128,
        max_loras=2,
        max_cpu_loras=2,
        disable_log_stats=False,
    )

    with vllm_runner(
        "Qwen/Qwen3-8B",
        **common,
        speculative_config={
            "method": "uno",
            "uno_lora_path": uno_adapter_path,
            "uno_mask_token_id": 151669,
            "num_speculative_tokens": 8,
        },
    ) as runner:
        engine = runner.llm.llm_engine
        # One uninterrupted baseline per long peer, so whichever peer the
        # scheduler preempts has its own solo result to be compared against.
        solo_ids = {
            f"uno-finish-peer-{index}": _run_request_to_finish(
                engine, f"uno-solo-peer-{index}", prompt, finish_sampling
            )
            for index, prompt in enumerate(finish_prompts)
        }
        assert all(
            len(ids) == finish_sampling.max_tokens for ids in solo_ids.values()
        ), f"solo finish peers did not reach their cap: {solo_ids}"

        # The cumulative counters also cover the solo runs, so snapshot
        # at the start of the mixed phase and assert only the phase deltas.
        before_metrics = runner.llm.get_metrics()
        mixed = _run_survivor_with_peers(
            engine,
            seed_prompt=seed_prompt,
            finish_prompts=finish_prompts,
            abort_prompt=abort_prompt,
            seed_sampling=seed_sampling,
            finish_sampling=finish_sampling,
            abort_sampling=abort_sampling,
        )
        after_metrics = runner.llm.get_metrics()

    for request_id, tokens in mixed.finished.items():
        print(f"survivor mixed tokens {request_id}={list(tokens)}")
    for receipt in mixed.receipts:
        print(f"survivor engagement receipt: {receipt}")
    print(f"survivor peak KV cache usage: {mixed.peak_kv_cache_usage:.3%}")
    print(f"survivor prompt token-id distinctness: {prompt_distinctness}")
    print(
        f"survivor generation lag: max_generation_lag={mixed.max_generation_lag} "
        f"tokens, final_lengths={mixed.final_lengths}"
    )
    print(
        f"survivor preempted-while-active counts: {mixed.preempted_while_active} "
        f"(solo lengths: {[len(ids) for ids in solo_ids.values()]})"
    )

    # The token-equality gate is only load-bearing for a request that was still
    # generating when it was preempted. The geometry above makes that event
    # unavoidable for every interleaving, so a run without it is a FAILURE, not
    # a skip: either the pre-gate arithmetic no longer matches the engine's
    # allocator, or the resume path is not being reached (checklist rows 2, 29).
    #
    # Note that a peak below 100% is not evidence the pool never filled: the
    # step that takes the last block is the same step whose next allocation
    # fails and frees the victim's blocks, so the per-step sample can top out
    # one block short (98.750% of a 69-block pool is 79 of 80). Read
    # `receipts` and `max_generation_lag`, not the peak.
    assert mixed.preempted_while_active, (
        "no finish peer was preempted while still generating, so the resume "
        "path was not exercised. The geometry asserts this cannot happen: "
        f"pool_blocks={pool_blocks}, growth_blocks={growth_blocks}, "
        f"worst_case_crossing_tokens={worst_case_crossing} < "
        f"cap {finish_sampling.max_tokens}. Receipt: "
        f"peak_kv_cache_usage={mixed.peak_kv_cache_usage:.3%}; "
        f"max_generation_lag={mixed.max_generation_lag}; "
        f"final_lengths={mixed.final_lengths}; "
        f"prompt_token_id_distinctness={prompt_distinctness}; "
        f"receipts={mixed.receipts}"
    )

    for request_id, count in mixed.preempted_while_active.items():
        assert count >= 1, (request_id, count)
        assert mixed.finished[request_id] == solo_ids[request_id], (
            f"preemption/resume changed {request_id}'s tokens: "
            f"solo={list(solo_ids[request_id])}, "
            f"mixed={list(mixed.finished[request_id])}"
        )

    drafts = _counter_total(
        after_metrics, "vllm:spec_decode_num_drafts"
    ) - _counter_total(before_metrics, "vllm:spec_decode_num_drafts")
    preemptions = _counter_total(
        after_metrics, "vllm:num_preemptions"
    ) - _counter_total(before_metrics, "vllm:num_preemptions")
    # Second, independent signal that the gate cannot pass vacuously:
    # `vllm:num_preemptions` counts scheduler iterations, while this histogram is
    # observed in `IterationStats.update_from_finished_request` ->
    # `FinishedRequestStats.num_preemptions` (`vllm/v1/metrics/loggers.py`), so
    # `sum > 0` only when a request was preempted and then recomputed to a
    # natural finish rather than merely aborted.
    finished_preemptions = _histogram_total(
        after_metrics, "vllm:request_num_preemptions"
    ) - _histogram_total(before_metrics, "vllm:request_num_preemptions")
    print(
        f"survivor mixed-phase deltas: drafts={drafts}, preemptions={preemptions}, "
        f"finished_preemptions={finished_preemptions}"
    )
    assert drafts > 0, "Uno did not draft during the mixed phase"
    assert preemptions > 0, (
        "the mixed phase did not preempt/recompute despite the derived KV "
        f"budget (preemptions={preemptions})"
    )
    assert finished_preemptions > 0, (
        "no preempted request was recomputed to a finished state "
        f"(vllm:request_num_preemptions delta={finished_preemptions}); the "
        "mixed phase produced no surviving recompute"
    )


@pytest.mark.parametrize(
    ("k", "enable_prefix_caching", "enforce_eager", "dual_stream"),
    [
        (1, False, True, False),
        (1, True, True, False),
        (1, True, False, False),
        (1, True, False, True),
        (8, True, True, False),
        (8, True, False, False),
        (8, True, False, True),
    ],
    ids=[
        "seed_only",
        "seed_only_cached",
        "seed_only_graphs",
        "seed_only_overlap",
        "eight_candidates_cached",
        "eight_candidates_graphs",
        "native_overlap_graphs",
    ],
)
def test_uno_greedy_matches_base_model(
    vllm_runner,
    monkeypatch: pytest.MonkeyPatch,
    uno_adapter_path: str,
    k: int,
    enable_prefix_caching: bool,
    enforce_eager: bool,
    dual_stream: bool,
):
    """Verify unequal prefills, batched decoding, and reused prefixes against base.

    A draft decode graph is captured only when ``CudaGraphManager._init_candidates``
    finds a ``cudagraph_capture_sizes`` entry at most ``max_num_seqs * k``: a
    candidate whose rounded token count exceeds
    ``max_num_reqs * decode_query_len`` is skipped. Here ``max_num_seqs=4`` and
    the capture sizes are ``[8, 16, 32, 64]``, so the expectation is derived
    per parameterization instead of hard-coded: K=8 admits graphs, while K=1
    (smallest size 8 > 4) drafts eagerly. The adapter noise rows
    (``lora_capture_cases = [2 if k > 1 else 0]``) are a separate rule;
    ``test_uno_mrv2.py`` pins the capture-size arithmetic on CPU.
    """
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setattr(envs, "VLLM_USE_V2_MODEL_RUNNER", True)
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)

    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    monkeypatch.setenv("VLLM_LORA_ENABLE_DUAL_STREAM", str(int(dual_stream)))
    monkeypatch.setattr(envs, "VLLM_LORA_ENABLE_DUAL_STREAM", dual_stream)

    prompts = get_test_prompts(mm_enabled=False, num_prompts=4)
    batches = [prompts]
    if enable_prefix_caching:
        batches.append(list(reversed(prompts)))
    sampling = SamplingParams(temperature=0, max_tokens=64, ignore_eos=True, seed=0)
    prefill_budget = 256
    # The K=1/K=8 matrix uses its own budget; check it against the same floor.
    assert engine_minimum_kv_bytes(MATRIX_MAX_MODEL_LEN) <= MATRIX_KV_BUDGET_BYTES, (
        f"the K-matrix KV budget {MATRIX_KV_BUDGET_BYTES} B is below the engine "
        f"floor {engine_minimum_kv_bytes(MATRIX_MAX_MODEL_LEN)} B at "
        f"max_model_len={MATRIX_MAX_MODEL_LEN}"
    )
    # Bound the reservation on larger devices while keeping both engines matched.
    total_memory = torch.cuda.get_device_properties(0).total_memory
    max_num_seqs = 4
    capture_sizes = [8, 16, 32, 64]
    # A captured decode graph exists only when some capture size fits the
    # max_num_seqs * k draft rows; --enforce-eager removes every capture size.
    expect_graphs = (not enforce_eager) and min(capture_sizes) <= max_num_seqs * k
    common = dict(
        revision=MODEL_REVISION,
        dtype="bfloat16",
        trust_remote_code=False,
        enforce_eager=enforce_eager,
        compilation_config={"cudagraph_capture_sizes": capture_sizes},
        async_scheduling=True,
        attention_config={"backend": "FLASH_ATTN", "flash_attn_version": 2},
        max_model_len=MATRIX_MAX_MODEL_LEN,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=prefill_budget,
        enable_chunked_prefill=True,
        enable_prefix_caching=enable_prefix_caching,
        gpu_memory_utilization=min(0.9, 24 * 1024**3 / total_memory),
        kv_cache_memory_bytes=MATRIX_KV_BUDGET_BYTES,
        enable_lora=True,
        max_lora_rank=128,
        max_loras=2,
        max_cpu_loras=2,
        disable_log_stats=False,
    )
    chat_kwargs = dict(
        sampling_params=sampling,
        chat_template_kwargs={"enable_thinking": False},
        use_tqdm=False,
    )

    # Keep the same LoRA infrastructure in the reference without a target adapter.
    # Sequential runner contexts release the first engine before loading the second.
    with vllm_runner("Qwen/Qwen3-8B", **common) as reference:
        ref_outputs = [reference.llm.chat(batch, **chat_kwargs) for batch in batches]
    prompt_lengths = [len(output.prompt_token_ids) for output in ref_outputs[0]]
    assert len(set(prompt_lengths)) > 1
    assert max(prompt_lengths) > prefill_budget

    with vllm_runner(
        "Qwen/Qwen3-8B",
        **common,
        speculative_config={
            "method": "uno",
            "uno_lora_path": uno_adapter_path,
            "uno_mask_token_id": 151669,
            "num_speculative_tokens": k,
        },
    ) as speculative:
        spec_outputs = []
        previous_metrics = None
        trained_acceptance = []
        for batch in batches:
            spec_outputs.append(speculative.llm.chat(batch, **chat_kwargs))
            metrics = speculative.llm.get_metrics()
            draft_tokens = get_spec_decode_metric_value(
                metrics, "vllm:spec_decode_num_draft_tokens"
            )
            if previous_metrics is not None:
                draft_tokens -= get_spec_decode_metric_value(
                    previous_metrics, "vllm:spec_decode_num_draft_tokens"
                )
                assert _counter_total(
                    metrics, "vllm:prefix_cache_hits"
                ) > _counter_total(previous_metrics, "vllm:prefix_cache_hits")
            assert draft_tokens > 0, "Uno did not verify any draft candidates"
            acceptance_len = compute_acceptance_len(metrics, previous_metrics)
            assert (1 if k == 1 else 3) < acceptance_len <= k + 1
            trained_acceptance.append(acceptance_len)
            print(f"Uno K={k}: mean acceptance length={acceptance_len:.3f}")
            previous_metrics = metrics

        # Exact greedy output alone cannot detect a disabled adapter: the
        # verifier corrects poor proposals, and the base seed already gives
        # acceptance length near two. This eager control must lose the trained
        # adapter's advantage while preserving the final target tokens. At K=1
        # there are no noise rows, so the adapter is never drafted with and the
        # control is expected to stay equivalent.
        if enforce_eager:
            speculative.llm.llm_engine.collective_rpc(_disable_uno_adapter_for_control)
            control_outputs = speculative.llm.chat(batches[-1], **chat_kwargs)
            control_metrics = speculative.llm.get_metrics()
            control_acceptance = compute_acceptance_len(
                control_metrics, previous_metrics
            )
            print(f"Uno K={k} adapter-disabled acceptance={control_acceptance:.3f}")
            if k == 8:
                assert control_acceptance < min(trained_acceptance) - 1.0
            else:
                assert abs(control_acceptance - trained_acceptance[-1]) < 0.5
            assert_request_outputs_match(
                ref_outputs[-1],
                control_outputs,
                required_matches=len(control_outputs),
                context="adapter-disabled",
            )
            assert [
                list(output.outputs[0].token_ids) for output in control_outputs
            ] == [list(output.outputs[0].token_ids) for output in ref_outputs[-1]]
        states = speculative.llm.llm_engine.collective_rpc(_uno_execution_state)
        assert all(state["shared_model"] for state in states)
        assert all(
            state["lora_plan_hits"]
            + state["lora_plan_misses"]
            + state["lora_plan_bypasses"]
            > 0
            for state in states
        ), f"Uno LoRA plan cache did not observe any installs: {states}"
        assert all(
            state["lora_plan_hits"]
            / (
                state["lora_plan_hits"]
                + state["lora_plan_misses"]
                + state["lora_plan_bypasses"]
            )
            >= 0.8
            for state in states
        ), f"Uno LoRA plans are rebuilding too often: {states}"
        assert all(state["lora_plan_bypasses"] == 0 for state in states), states
        if expect_graphs:
            assert all(state["draft_graphs"] > 0 for state in states), states
            assert all(state["draft_graph_replays"] > 0 for state in states), states
        else:
            assert all(state["draft_graphs"] == 0 for state in states), states
            assert all(state["draft_graph_replays"] == 0 for state in states), states
            assert all(state["draft_eager_proposals"] > 0 for state in states), states
        print(f"Uno execution state: {states}")

    for batch_index, (ref_batch, spec_batch) in enumerate(
        zip(ref_outputs, spec_outputs)
    ):
        context = (
            f"Uno K={k}, prefix_cache={enable_prefix_caching}, batch={batch_index}"
        )
        assert_request_outputs_match(
            ref_batch,
            spec_batch,
            required_matches=len(ref_batch),
            context=context,
        )
        for index, (ref_output, spec_output) in enumerate(zip(ref_batch, spec_batch)):
            ref_ids = list(ref_output.outputs[0].token_ids)
            spec_ids = list(spec_output.outputs[0].token_ids)
            assert ref_ids == spec_ids, (
                f"{context}, request={index}: reference tokens={ref_ids}, "
                f"Uno tokens={spec_ids}"
            )
