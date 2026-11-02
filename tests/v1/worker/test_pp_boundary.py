# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace

import pytest
import torch

from vllm.v1.worker.pp.boundary import (
    DynamicPPConfig,
    PPBalanceSample,
    PPBoundary,
    PPBoundaryController,
    PPStrategyReporter,
)


def _choose_with_cost(controller, batch_id, depth, pp0_ms, pp1_ms):
    samples = (
        []
        if batch_id == 0
        else [
            PPBalanceSample(
                batch_id - 1,
                controller.boundary,
                controller.version,
                20480,
                1,
                pp0_ms,
                pp1_ms,
            )
        ]
    )
    return controller.choose(batch_id, depth, samples)


def test_backlog_moves_work_upstream_only_after_completed_timing_samples():
    config = DynamicPPConfig(enabled=True, redundant_layers=3, cooldown_steps=2)
    controller = PPBoundaryController(config, 10, 20)
    assert controller.choose(0, 2).boundary == 10
    assert _choose_with_cost(controller, 1, 2, 50, 100).boundary == 10
    assert _choose_with_cost(controller, 2, 2, 50, 100).boundary == 11
    assert _choose_with_cost(controller, 3, 0, 70, 80).boundary == 11
    assert _choose_with_cost(controller, 4, 0, 70, 80).boundary == 11
    assert _choose_with_cost(controller, 5, 0, 70, 80).boundary == 11
    assert _choose_with_cost(controller, 6, 0, 70, 80).boundary == 12
    assert controller.version == 2


def test_host_ahead_full_send_ring_never_moves_boundary_without_device_feedback():
    controller = PPBoundaryController(DynamicPPConfig(cooldown_steps=1), 22, 43)
    for batch_id in range(100):
        assert controller.choose(batch_id, 2).boundary == 22
    assert controller.version == 0


def test_high_watermark_cannot_add_layers_when_upstream_is_slower():
    controller = PPBoundaryController(DynamicPPConfig(cooldown_steps=1), 22, 43)
    controller.choose(0, 2)
    assert _choose_with_cost(controller, 1, 2, 1300, 1160).boundary == 22
    assert _choose_with_cost(controller, 2, 2, 1300, 1160).boundary == 21


@pytest.mark.parametrize("faster", [True, False])
@pytest.mark.parametrize("depth", [0, 1, 2])
def test_real_ready_watermark_changes_how_soon_the_controller_responds(depth, faster):
    controller = PPBoundaryController(DynamicPPConfig(cooldown_steps=4), 22, 43)
    confirmed = depth == (2 if faster else 0)
    change_batch = 4 if confirmed else 8
    times = (1100, 1300) if faster else (1300, 1100)
    for batch_id in range(change_batch):
        assert _choose_with_cost(controller, batch_id, depth, *times).boundary == 22
    assert _choose_with_cost(controller, change_batch, depth, *times).boundary == (
        23 if faster else 21
    )


@pytest.mark.parametrize("start", [20, 24])
def test_completed_cost_feedback_converges_to_cut22_from_both_directions(start):
    controller = PPBoundaryController(
        DynamicPPConfig(initial_offset=start - 22, cooldown_steps=4), 22, 43
    )
    history = []
    for batch_id in range(100):
        cut = controller.boundary
        pp0_ms, pp1_ms = 1119 + (cut - 20) * 62, 1358 - (cut - 20) * 64
        history.append(
            _choose_with_cost(
                controller, batch_id, 2 if pp0_ms < pp1_ms else 0, pp0_ms, pp1_ms
            ).boundary
        )
    assert history[-60:] == [22] * 60
    assert controller.version == 2
    assert controller.feedback_stats["decision"] == "balanced_hold"


def test_old_version_and_stale_feedback_do_not_cascade_into_more_changes():
    controller = PPBoundaryController(DynamicPPConfig(cooldown_steps=1), 22, 43)
    controller.choose(0, 2)
    sample = PPBalanceSample(0, 22, 0, 20480, 1, 1100, 1300)
    assert controller.choose(1, 2, [sample]).boundary == 23
    old = replace(sample, batch_id=1)
    assert controller.choose(2, 2, [old]).boundary == 23
    assert "pp0_device_ms" not in controller.feedback_stats
    assert controller.feedback_stats["feedback_version"] == 1
    current = replace(sample, batch_id=2, boundary=23, version=1)
    assert controller.choose(3, 2, [current], feedback_fresh=False).boundary == 23
    assert controller.feedback_stats["decision"] == "stale_feedback_hold"


def test_adjacent_integer_boundaries_do_not_oscillate():
    controller = PPBoundaryController(DynamicPPConfig(cooldown_steps=2), 22, 43)
    for batch_id in range(50):
        pp0_ms, pp1_ms = (1100, 1300) if controller.boundary == 22 else (1250, 1150)
        _choose_with_cost(controller, batch_id, 2, pp0_ms, pp1_ms)
    assert controller.boundary == 23 and controller.version == 1
    assert controller.feedback_stats["decision"] == "adjacent_boundary_not_faster"


def test_changed_device_load_invalidates_old_neighbor_costs():
    controller = PPBoundaryController(DynamicPPConfig(cooldown_steps=2), 22, 43)
    for batch_id in range(10):
        pp0_ms, pp1_ms = (1100, 1300) if controller.boundary == 22 else (1250, 1150)
        _choose_with_cost(controller, batch_id, 2, pp0_ms, pp1_ms)
    assert controller.boundary == 23
    _choose_with_cost(controller, 10, 0, 1250, 600)
    assert _choose_with_cost(controller, 11, 0, 1250, 600).boundary == 22


def test_replayed_samples_and_future_batches_cannot_satisfy_cooldown():
    controller = PPBoundaryController(DynamicPPConfig(cooldown_steps=2), 22, 43)
    sample = PPBalanceSample(0, 22, 0, 20480, 1, 1100, 1300)
    controller.choose(0, 2)
    for batch in range(1, 5):
        assert controller.choose(batch, 2, [sample]).boundary == 22
    assert controller.feedback_stats["matched_samples"] == 1
    with pytest.raises(ValueError, match="future batch"):
        controller.choose(5, 2, [replace(sample, batch_id=5)])


def test_different_workload_sizes_do_not_share_a_cooldown_window():
    controller = PPBoundaryController(DynamicPPConfig(cooldown_steps=2), 22, 43)
    controller.choose(0, 2)
    first = PPBalanceSample(0, 22, 0, 20480, 1, 1100, 1300)
    controller.choose(1, 2, [first])
    assert (
        controller.choose(2, 2, [replace(first, batch_id=1, num_tokens=100)]).boundary
        == 22
    )
    assert controller.feedback_stats["matched_samples"] == 1


def _choose_decode_shape(controller, batch_id, reqs, pp0_ms=100, pp1_ms=100):
    return controller.choose(
        batch_id,
        0,
        [
            PPBalanceSample(
                batch_id - 1,
                controller.boundary,
                controller.version,
                reqs * 4,
                reqs,
                pp0_ms,
                pp1_ms,
            )
        ],
    )


@pytest.mark.parametrize("mode", ["allow_mtp_decode", "allow_prefix_caching"])
def test_alternating_decode_or_prefix_shapes_accumulate_cooldown(mode):
    controller = PPBoundaryController(DynamicPPConfig(**{mode: True}), 22, 43)
    controller.choose(0, 0)
    for batch in range(1, 15):
        assert (
            _choose_decode_shape(
                controller, batch, 1 if batch % 2 else 7, 200, 100
            ).boundary
            == 22
        )
    assert _choose_decode_shape(controller, 15, 1, 200, 100).boundary == 21
    assert controller.version == 1


@pytest.mark.parametrize("mode", ["allow_mtp_decode", "allow_prefix_caching"])
def test_decode_or_prefix_interleaved_shapes_keep_their_own_timing_medians(mode):
    controller = PPBoundaryController(DynamicPPConfig(**{mode: True}), 22, 43)
    controller.choose(0, 0)
    for batch in range(1, 17):
        reqs, ms = (1, 100) if batch % 2 else (7, 1000)
        _choose_decode_shape(controller, batch, reqs, ms, ms)
        if batch >= 15:
            assert controller.feedback_stats["matched_samples"] == 8
            assert controller.feedback_stats["pp0_device_ms"] == ms
            assert controller.feedback_stats["pp1_device_ms"] == ms
            assert controller.feedback_stats["timing_num_reqs"] == reqs
            assert controller.feedback_stats["decision"] == "balanced_hold"


def test_decode_boundary_change_discards_every_shape_window():
    controller = PPBoundaryController(DynamicPPConfig(allow_mtp_decode=True), 22, 43)
    controller.choose(0, 0)
    for batch in range(1, 8):
        _choose_decode_shape(controller, batch, 7)
    for batch in range(8, 16):
        _choose_decode_shape(controller, batch, 1, 200, 100)
    assert controller.version == 1 and controller.boundary == 21
    _choose_decode_shape(controller, 16, 7)
    assert controller.feedback_stats["matched_samples"] == 1
    assert controller.feedback_stats["decision"] == "awaiting_completed_samples"


def test_decode_expired_feedback_discards_every_shape_window():
    controller = PPBoundaryController(DynamicPPConfig(allow_mtp_decode=True), 22, 43)
    controller.choose(0, 0)
    for batch in range(1, 15):
        _choose_decode_shape(controller, batch, 1 if batch % 2 else 7)
    controller.choose(15, 0, feedback_fresh=False)
    for batch, reqs in ((16, 1), (17, 7)):
        _choose_decode_shape(controller, batch, reqs)
        assert controller.feedback_stats["matched_samples"] == 1
        assert controller.feedback_stats["decision"] == "awaiting_completed_samples"


def test_decode_shape_windows_evict_least_recently_used_after_sixty_four():
    controller = PPBoundaryController(DynamicPPConfig(allow_mtp_decode=True), 22, 43)
    controller.choose(0, 0)
    for batch in range(1, 65):
        _choose_decode_shape(controller, batch, batch)
    _choose_decode_shape(controller, 65, 1)
    _choose_decode_shape(controller, 66, 65)
    _choose_decode_shape(controller, 67, 1)
    assert controller.feedback_stats["matched_samples"] == 3
    _choose_decode_shape(controller, 68, 2)
    assert controller.feedback_stats["matched_samples"] == 1


def test_watermark_deadband_does_not_change_the_boundary():
    controller = PPBoundaryController(DynamicPPConfig(cooldown_steps=1), 10, 20)
    for batch_id in range(8):
        assert _choose_with_cost(controller, batch_id, 1, 100, 101).boundary == 10
    assert controller.version == 0


def test_strategy_report_exposes_oscillation_between_periodic_samples():
    controller = PPBoundaryController(DynamicPPConfig(redundant_layers=1), 22, 43)
    reporter = PPStrategyReporter(controller, 0)
    first = reporter.observe(PPBoundary(0, 21, 1), 0, now=0)
    assert first is not None
    assert first["window_batches"] == 1
    assert first["trigger"] == "initial"
    assert first["observation"] == "host_forward_submit"
    assert first["watermark_kind"] == "device_ready_minus_consumer_credit"
    for batch, cut, depth, now in ((1, 22, 1, 1), (2, 23, 2, 2), (3, 22, 2, 5)):
        changed = reporter.observe(PPBoundary(batch, cut, batch + 1), depth, now=now)
        assert changed is not None
        assert changed["trigger"] == "boundary_change"
        assert changed["boundary_changed"] and not changed["periodic_snapshot"]
        assert changed["batch_id"] == batch and changed["boundary"] == cut
    report = reporter.observe(PPBoundary(4, 21, 5), 0, now=10)
    assert report is not None
    assert report["trigger"] == "periodic"
    assert report["boundary_changed"] and report["periodic_snapshot"]
    assert report["boundary"] == 21
    assert report["execution_range"] == (0, 21)
    assert (report["pp0_layers"], report["pp1_layers"]) == (21, 22)
    assert report["boundary_limits"] == [21, 23] and report["at_boundary_limit"]
    assert report["window_boundary_counts"] == {21: 1, 22: 2, 23: 1}
    assert report["window_boundary_changes"] == 4
    assert report["window_batches"] == 4 and report["window_seconds"] == 10
    assert (
        report["window_pending_send_min"] == 0
        and report["window_pending_send_max"] == 2
    )
    assert report["unchanged_seconds"] == 0


def test_strategy_report_uses_inflight_header_and_distinguishes_missing_watermark():
    controller = PPBoundaryController(DynamicPPConfig(redundant_layers=1), 22, 43)
    reporter = PPStrategyReporter(controller, 1)
    report = reporter.observe(PPBoundary(7, 21, 1), None, now=100)
    assert report is not None
    assert report["batch_id"] == 7 and report["version"] == 1
    assert report["execution_range"] == (21, 43)
    assert report["pending_send_count"] is None
    assert (
        report["window_pending_send_min"] is None
        and report["window_pending_send_max"] is None
    )
    report = reporter.observe(PPBoundary(8, 21, 1), None, now=110)
    assert report is not None
    assert report["trigger"] == "periodic"
    assert report["unchanged_seconds"] == 10
    assert report["window_boundary_changes"] == 0
    assert report["window_boundary_counts"] == {21: 1} and report["window_batches"] == 1


def test_hash_routing_layers_remain_on_the_rank_that_owns_input_ids():
    controller = PPBoundaryController(
        DynamicPPConfig(redundant_layers=3, cooldown_steps=1), 4, 8, minimum_boundary=3
    )
    for batch_id in range(8):
        header = controller.choose(batch_id, 0)
        assert header.boundary >= 3
    assert controller.loaded_range(1) == (3, 8)


@pytest.mark.parametrize("queue_depth,expected", [(0, 1), (2, 19)])
def test_boundary_saturates_without_emptying_a_stage(queue_depth, expected):
    controller = PPBoundaryController(
        DynamicPPConfig(redundant_layers=30, cooldown_steps=1), 10, 20
    )
    for batch_id in range(50):
        header = _choose_with_cost(
            controller,
            batch_id,
            queue_depth,
            *((100, 50) if queue_depth == 0 else (50, 100)),
        )
    assert header.boundary == expected
    assert controller.loaded_range(0) == (0, 19)
    assert controller.loaded_range(1) == (1, 20)


def test_inflight_batches_keep_their_own_boundary_versions():
    config = DynamicPPConfig(cooldown_steps=1)
    receiver = PPBoundaryController(config, 10, 20)
    headers = [
        PPBoundary(batch_id, boundary, batch_id)
        for batch_id, boundary in enumerate((10, 11, 10, 9))
    ]
    for batch_id, original in enumerate(headers):
        header = receiver.accept(original.to_wire(), batch_id)
        assert receiver.execution_range(1, header) == (original.boundary, 20)


def test_every_layer_runs_exactly_once_for_each_allowed_boundary():
    """Non-commuting layer transforms expose missing, duplicated, or reordered work."""
    torch.manual_seed(7)
    layers = [(torch.randn(8, 8) / 4, torch.randn(8)) for _ in range(12)]
    hidden = torch.randn(3, 4, 8)

    def execute(value, start, end):
        for weight, bias in layers[start:end]:
            value = torch.tanh(value @ weight + bias)
        return value

    expected = execute(hidden, 0, len(layers))
    controller = PPBoundaryController(
        DynamicPPConfig(redundant_layers=3), 6, len(layers)
    )
    for boundary in range(controller.minimum, controller.maximum + 1):
        header = PPBoundary(0, boundary, 0)
        first = execute(hidden, *controller.execution_range(0, header))
        actual = execute(first, *controller.execution_range(1, header))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "wire", [None, (0, 10), (0, 10, -1), (True, 10, 0), (0, 10, "0")]
)
def test_bad_headers_are_rejected(wire):
    with pytest.raises(ValueError):
        PPBoundaryController(DynamicPPConfig(), 10, 20).accept(wire, 0)


@pytest.mark.parametrize("wire", [(1, 10, 0), (0, 15, 0)])
def test_wrong_batch_or_unloaded_boundary_is_rejected(wire):
    with pytest.raises(ValueError):
        PPBoundaryController(DynamicPPConfig(), 10, 20).accept(wire, 0)


def test_receiver_rejects_an_unversioned_boundary_change():
    controller = PPBoundaryController(DynamicPPConfig(), 10, 20)
    controller.accept((0, 10, 0), 0)
    with pytest.raises(ValueError, match="without changing its version"):
        controller.accept((1, 11, 0), 1)


@pytest.mark.parametrize(
    "values",
    [
        {"enabled": "false"},
        {"allow_mtp_prefill": 1},
        {"redundant_layers": True},
        {"redundant_layers": 0},
        {"low_watermark": 2, "high_watermark": 2},
        {"initial_offset": 3},
        {"cooldown_steps": 0},
        {"max_step": 0},
        {"log_interval_seconds": -1},
        {"log_interval_seconds": 1.5},
        {"log_interval_seconds": True},
        {"balance_tolerance_percent": 100},
        {"balance_tolerance_percent": -1},
        {"feedback_timeout_seconds": 0},
        {"redundant_layres": 2},
    ],
)
def test_invalid_config_is_rejected(values):
    with pytest.raises(ValueError):
        DynamicPPConfig.from_mapping(values)


@pytest.fixture
def mtp_prefill_config(tmp_path, monkeypatch):
    from transformers import DeepseekV4Config, LlamaConfig

    from vllm.config import (
        CacheConfig,
        DeviceConfig,
        ModelConfig,
        ParallelConfig,
        SchedulerConfig,
        SpeculativeConfig,
        VllmConfig,
    )

    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path / "cache"))
    for name, config in (
        (
            "v4",
            DeepseekV4Config(
                architectures=["DeepseekV4ForCausalLM"],
                num_hidden_layers=4,
                hidden_size=128,
                num_attention_heads=4,
                vocab_size=256,
                num_nextn_predict_layers=1,
                mlp_layer_types=["moe"] * 4,
            ),
        ),
        ("llama", LlamaConfig(num_hidden_layers=4, hidden_size=128, vocab_size=256)),
    ):
        config.save_pretrained(tmp_path / name)

    def build(
        *,
        allow=True,
        decode=None,
        enabled=True,
        speculative=True,
        method="mtp",
        tokens=1,
        draft_eager=True,
        target="v4",
        prefix=False,
        allow_prefix=False,
        backend="collective",
    ):
        model = ModelConfig(
            model=str(tmp_path / target),
            skip_tokenizer_init=True,
            enforce_eager=True,
            max_model_len=256,
            dtype="float32",
        )
        parallel = ParallelConfig(
            pipeline_parallel_size=2, distributed_executor_backend="mp"
        )
        draft = (
            SpeculativeConfig(
                model=str(tmp_path / "v4"),
                method=method,
                num_speculative_tokens=tokens,
                enforce_eager=draft_eager,
                target_model_config=model,
                target_parallel_config=parallel,
            )
            if speculative
            else None
        )
        return VllmConfig(
            model_config=model,
            device_config=DeviceConfig(device="cpu"),
            parallel_config=parallel,
            scheduler_config=SchedulerConfig(
                max_model_len=256,
                is_encoder_decoder=False,
                max_num_batched_tokens=256,
                enable_chunked_prefill=False,
            ),
            cache_config=CacheConfig(enable_prefix_caching=prefix),
            speculative_config=draft,
            additional_config={
                "pp_send_buffer_size": 2,
                "dynamic_pp_config": {
                    "enabled": enabled,
                    "allow_mtp_prefill": allow,
                    "allow_prefix_caching": allow_prefix,
                    "kv_transfer_backend": backend,
                    **({"allow_mtp_decode": decode} if decode is not None else {}),
                },
            },
        )

    return build


def test_mtp_prefill_requires_explicit_opt_in(mtp_prefill_config):
    from vllm.v1.worker.pp.config import get_pp_config

    with pytest.raises(ValueError, match="requires allow_mtp_prefill"):
        get_pp_config(mtp_prefill_config(allow=False))


def test_opt_in_accepts_only_the_fresh_prefill_mtp_configuration(mtp_prefill_config):
    from vllm.v1.worker.pp.config import get_pp_config

    config = mtp_prefill_config()
    size, dynamic = get_pp_config(config)
    assert size == 2 and dynamic.enabled and dynamic.allow_mtp_prefill
    assert config.speculative_config.num_speculative_tokens == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"speculative": False},
        {"tokens": 2},
        {"draft_eager": False},
        {"target": "llama"},
    ],
)
def test_mtp_prefill_opt_in_rejects_unsupported_drafters(mtp_prefill_config, kwargs):
    from vllm.v1.worker.pp.config import get_pp_config

    with pytest.raises(ValueError, match="requires DeepSeek-V4 with eager MTP"):
        get_pp_config(mtp_prefill_config(**kwargs))


def test_ngram_cannot_enter_the_dynamic_pp_mtp_path(mtp_prefill_config):
    from vllm.v1.worker.pp.config import get_pp_config

    with pytest.raises(ValueError, match="allow_mtp_prefill requires DeepSeek-V4"):
        get_pp_config(mtp_prefill_config(method="ngram"))


def test_disabled_dynamic_pp_ignores_the_mtp_opt_in(mtp_prefill_config):
    from vllm.v1.worker.pp.config import get_pp_config

    size, dynamic = get_pp_config(mtp_prefill_config(enabled=False, speculative=False))
    assert size == 2 and not dynamic.enabled


def _fresh_schedule(*, computed=0, scheduled=3, max_tokens=1, cached=False):
    from vllm import SamplingParams
    from vllm.v1.core.sched.output import (
        CachedRequestData,
        NewRequestData,
        SchedulerOutput,
    )

    request = NewRequestData(
        req_id="request",
        prompt_token_ids=[1, 2, 3],
        mm_features=[],
        sampling_params=SamplingParams(max_tokens=max_tokens),
        pooling_params=None,
        block_ids=([0],),
        num_computed_tokens=computed,
        lora_request=None,
    )
    history = (
        CachedRequestData(
            req_ids=["cached"],
            resumed_req_ids=set(),
            new_token_ids=[[4]],
            all_token_ids={},
            new_block_ids=[None],
            num_computed_tokens=[3],
            num_output_tokens=[1],
        )
        if cached
        else CachedRequestData.make_empty()
    )
    return replace(
        SchedulerOutput.make_empty(),
        scheduled_new_reqs=[request],
        scheduled_cached_reqs=history,
        num_scheduled_tokens={"request": scheduled},
        total_num_scheduled_tokens=scheduled,
    )


def test_complete_fresh_prefill_is_allowed():
    from vllm.v1.worker.pp.boundary import validate_fresh_prefill

    validate_fresh_prefill(_fresh_schedule())


@pytest.mark.parametrize("decode", [False, True])
def test_prefix_hit_executes_only_the_uncached_suffix(decode):
    from vllm.v1.worker.pp.boundary import (
        validate_dynamic_decode,
        validate_fresh_prefill,
    )

    validate = validate_dynamic_decode if decode else validate_fresh_prefill
    # Full logical prefix hit still recomputes the final prompt token.
    validate(_fresh_schedule(computed=2, scheduled=1), allow_cached_prefix=True)
    validate(_fresh_schedule(computed=1, scheduled=2), allow_cached_prefix=True)
    with pytest.raises(ValueError, match="prefix"):
        validate(_fresh_schedule(computed=2, scheduled=1))


@pytest.mark.parametrize(
    "computed,scheduled", [(-1, 4), (3, 0), (4, 1), (1, 1), (1, 3)]
)
def test_prefix_opt_in_still_rejects_partial_or_invalid_suffix(computed, scheduled):
    from vllm.v1.worker.pp.boundary import (
        validate_dynamic_decode,
        validate_fresh_prefill,
    )

    for validate in (validate_fresh_prefill, validate_dynamic_decode):
        with pytest.raises(ValueError):
            validate(
                _fresh_schedule(computed=computed, scheduled=scheduled),
                allow_cached_prefix=True,
            )


@pytest.mark.parametrize("decode", [False, True])
def test_prefix_config_requires_explicit_state_handoff_opt_in(
    mtp_prefill_config, decode
):
    from vllm.v1.worker.pp.config import get_pp_config

    options = dict(
        allow=not decode, decode=decode, tokens=3 if decode else 1, prefix=True
    )
    with pytest.raises(ValueError, match="allow_prefix_caching"):
        get_pp_config(mtp_prefill_config(**options))
    _, dynamic = get_pp_config(mtp_prefill_config(**options, allow_prefix=True))
    assert dynamic.allow_prefix_caching


def test_mtp_verification_is_rejected_before_communication():
    from vllm.v1.worker.pp.boundary import validate_fresh_prefill

    schedule = replace(_fresh_schedule(), scheduled_spec_decode_tokens={"request": [4]})
    with pytest.raises(ValueError, match="cannot execute speculative verification"):
        validate_fresh_prefill(schedule)


@pytest.mark.parametrize(
    "kwargs",
    [dict(computed=1), dict(scheduled=2), dict(max_tokens=2), dict(cached=True)],
)
def test_unsynchronized_history_is_rejected_before_communication(kwargs):
    from vllm.v1.worker.pp.boundary import validate_fresh_prefill

    with pytest.raises(ValueError):
        validate_fresh_prefill(_fresh_schedule(**kwargs))


def test_mtp3_decode_requires_its_separate_state_handoff_opt_in(mtp_prefill_config):
    from vllm.v1.worker.pp.config import get_pp_config

    config = mtp_prefill_config(allow=False, decode=True, tokens=3)
    size, dynamic = get_pp_config(config)
    assert size == 2 and dynamic.allow_mtp_decode and not dynamic.allow_mtp_prefill
    assert dynamic.enabled


def test_mooncake_backend_requires_prefix_or_mtp_decode(mtp_prefill_config):
    from vllm.v1.worker.pp.config import get_pp_config

    with pytest.raises(ValueError, match="needs prefix caching or MTP decode"):
        get_pp_config(mtp_prefill_config(backend="mooncake"))
    _, dynamic = get_pp_config(
        mtp_prefill_config(prefix=True, allow_prefix=True, backend="mooncake")
    )
    assert dynamic.kv_transfer_backend == "mooncake"
    _, dynamic = get_pp_config(
        mtp_prefill_config(allow=False, decode=True, tokens=3, backend="mooncake")
    )
    assert dynamic.kv_transfer_backend == "mooncake"


@pytest.mark.parametrize(
    "values",
    [
        {"kv_transfer_backend": True},
        {"kv_transfer_backend": "nixl"},
        {"mooncake_protocol": ""},
        {"mooncake_device_name": 3},
    ],
)
def test_mooncake_configuration_rejects_invalid_transport_options(values):
    with pytest.raises(ValueError):
        DynamicPPConfig.from_mapping(values)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tokens": 1},
        {"tokens": 2},
        {"tokens": 4},
        {"draft_eager": False},
        {"speculative": False},
        {"target": "llama"},
    ],
)
def test_state_handoff_decode_remains_narrowly_guarded(mtp_prefill_config, kwargs):
    from vllm.v1.worker.pp.config import get_pp_config

    options = {"allow": False, "decode": True, "tokens": 3, **kwargs}
    with pytest.raises(ValueError, match="allow_mtp_decode requires DeepSeek-V4"):
        get_pp_config(mtp_prefill_config(**options))


def test_prefill_and_decode_modes_cannot_be_enabled_together(mtp_prefill_config):
    from vllm.v1.worker.pp.config import get_pp_config

    with pytest.raises(ValueError, match="Choose either"):
        get_pp_config(mtp_prefill_config(allow=True, decode=True))


def test_state_handoff_accepts_history_but_old_prefill_mode_still_rejects_it():
    from vllm.v1.worker.pp.boundary import (
        validate_dynamic_decode,
        validate_fresh_prefill,
    )

    fresh = _fresh_schedule(max_tokens=1024)
    validate_dynamic_decode(fresh)
    history = replace(
        _fresh_schedule(cached=True),
        scheduled_new_reqs=[],
        scheduled_spec_decode_tokens={"cached": [4, 5, 6]},
    )
    validate_dynamic_decode(history)
    with pytest.raises(ValueError, match="cannot execute speculative verification"):
        validate_fresh_prefill(history)


@pytest.mark.parametrize("failure", ["prefix", "partial", "resume", "draft4"])
def test_decode_handoff_rejects_unvalidated_imports(failure):
    from vllm.v1.worker.pp.boundary import validate_dynamic_decode

    schedule = _fresh_schedule(max_tokens=1024)
    if failure == "prefix":
        schedule = _fresh_schedule(computed=1, max_tokens=1024)
    elif failure == "partial":
        schedule = _fresh_schedule(scheduled=2, max_tokens=1024)
    elif failure == "resume":
        schedule = replace(
            schedule,
            scheduled_cached_reqs=replace(
                _fresh_schedule(cached=True).scheduled_cached_reqs,
                resumed_req_ids={"cached"},
            ),
        )
    else:
        schedule = replace(schedule, scheduled_spec_decode_tokens={"request": [1] * 4})
    with pytest.raises(ValueError):
        validate_dynamic_decode(schedule)
