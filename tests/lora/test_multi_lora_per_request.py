# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import math
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import msgspec
import numpy as np
import pytest
import torch
from pydantic import ValidationError

import vllm.lora.punica_wrapper.punica_gpu as punica_gpu_mod
import vllm.triton_utils as triton_utils
from tests.v1.core.utils import create_requests, create_scheduler
from vllm.config.lora import LoRAConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.exceptions import VLLMValidationError
from vllm.lora.layers.utils import LoRAMapping
from vllm.lora.ops.torch_ops import bgmv_expand_slice, bgmv_shrink
from vllm.lora.punica_wrapper.punica_base import PunicaWrapperBase
from vllm.lora.punica_wrapper.punica_cpu import PunicaWrapperCPU
from vllm.lora.punica_wrapper.punica_gpu import PunicaWrapperGPU
from vllm.lora.request import LoRARequest
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager, WorkerLoRAManager
from vllm.platforms import current_platform
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.hashing import sha256, sha256_cbor
from vllm.v1.core.kv_cache_utils import (
    generate_block_hash_extra_keys,
    hash_block_tokens,
    to_event_extra_keys,
)
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.engine.llm_engine import LLMEngine
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
from vllm.v1.worker.cpu.lora_utils import LoraState as CpuLoraState
from vllm.v1.worker.gpu.lora_utils import (
    LoraState as GpuLoraState,
)
from vllm.v1.worker.gpu.lora_utils import (
    get_num_active_loras_for_dispatch,
)
from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin

# ---------------------------------------------------------------------------
# Request representation & validation
# ---------------------------------------------------------------------------


def test_lora_request_single_and_composite_api():
    a = LoRARequest("style", 1, "/path/style")
    b = LoRARequest("domain", 2, "/path/domain")

    assert not a.is_composite
    assert a.adapter_id == 1
    assert a.lora_scale == 1.0
    assert a.get_adapters() == ((a, 1.0),)
    assert a.get_adapter_requests() == (a,)
    assert a.get_adapter_ids() == (1,)
    assert a.get_adapter_scales() == (1.0,)

    # Scaled single adapter shares identity with unit-scale adapter for slot dedup.
    a_scaled = LoRARequest("style", 1, "/path/style", lora_scale=0.5)
    assert not a_scaled.is_composite
    assert a_scaled == a
    assert hash(a_scaled) == hash(a)
    assert len({a, a_scaled}) == 1
    assert a_scaled.get_adapter_scales() == (0.5,)

    # Negative zero normalizes cleanly.
    a_neg_zero = LoRARequest("style", 1, "/path/style", lora_scale=-0.0)
    assert math.copysign(1.0, a_neg_zero.lora_scale) == 1.0

    # Composite via factory (mixed tuple and bare LoRARequest).
    comp = LoRARequest.from_adapters([(a, 0.7), b])
    assert comp.is_composite
    assert comp.adapter_id is None
    assert comp.lora_name == "style+domain"
    assert comp.get_adapters() == ((a, 0.7), (b, 1.0))
    assert comp.get_adapter_requests() == (a, b)
    assert comp.get_adapter_ids() == (1, 2)
    assert comp.get_adapter_scales() == (0.7, 1.0)

    comp_same = LoRARequest(adapters=[(a, 0.7), (b, 1.0)])
    comp_diff_scale = LoRARequest(adapters=[(a, 0.3), (b, 1.0)])
    assert comp == comp_same
    assert hash(comp) == hash(comp_same)
    assert comp != comp_diff_scale
    assert comp != a


def test_lora_request_validation_rejects_invalid_inputs():
    a = LoRARequest("a", 1, "/path/a")
    b = LoRARequest("b", 2, "/path/b")

    with pytest.raises(ValueError, match="non-empty"):
        LoRARequest(adapters=[])

    with pytest.raises(ValueError, match="cannot be empty"):
        LoRARequest.from_adapters([])

    # Duplicate lora_int_id.
    a_dup_id = LoRARequest("other", 1, "/path/other")
    with pytest.raises(ValueError, match="Duplicate lora_int_id"):
        LoRARequest.from_adapters([(a, 0.5), (a_dup_id, 0.5)])

    # Duplicate lora_name.
    a_dup_name = LoRARequest("a", 3, "/path/a2")
    with pytest.raises(ValueError, match="Duplicate lora_name"):
        LoRARequest.from_adapters([(a, 0.5), (a_dup_name, 0.5)])

    # Nested composite.
    comp = LoRARequest.from_adapters([(a, 0.5), (b, 0.5)])
    with pytest.raises(ValueError, match="Nested composite"):
        LoRARequest.from_adapters([(comp, 1.0)])

    # Non-finite, float32-overflowing, and boolean scales.
    bad_scales: list[Any] = [
        float("nan"),
        float("inf"),
        float("-inf"),
        1e39,
        -1e39,
        True,
        False,
        "0.5",
    ]
    for bad_scale in bad_scales:
        with pytest.raises(ValueError, match="scale"):
            LoRARequest("a", 1, "/path/a", lora_scale=bad_scale)
        with pytest.raises(ValueError, match="scale"):
            LoRARequest.from_adapters([(a, bad_scale)])

    # Large float32-representable scale is accepted.
    assert LoRARequest("a", 1, "/path/a", lora_scale=1e38).lora_scale == 1e38

    # Conflicting composite fields.
    with pytest.raises(ValueError, match="lora_int_id"):
        LoRARequest(lora_int_id=5, adapters=[(a, 1.0)])
    with pytest.raises(ValueError, match="lora_path"):
        LoRARequest(lora_path="/path/x", adapters=[(a, 1.0)])
    with pytest.raises(ValueError, match="lora_scale"):
        LoRARequest(lora_scale=0.5, adapters=[(a, 1.0)])
    with pytest.raises(ValueError, match="load_inplace"):
        LoRARequest(load_inplace=True, adapters=[(a, 1.0)])

    # Constituent adapter with non-unit lora_scale.
    a_scaled = LoRARequest("a", 1, "/path/a", lora_scale=0.5)
    with pytest.raises(ValueError, match="lora_scale=1.0"):
        LoRARequest.from_adapters([(a_scaled, 0.5)])

    # Malformed adapter entry.
    with pytest.raises(ValueError, match="pair"):
        LoRARequest(adapters=[(a, 0.5, 1.0)])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="must be a LoRARequest"):
        LoRARequest(adapters=[("not-a-lora", 0.5)])  # type: ignore[list-item]


def test_lora_request_msgspec_roundtrip():
    a = LoRARequest("style", 1, "/path/style")
    b = LoRARequest("domain", 2, "/path/domain")
    comp = LoRARequest.from_adapters([(a, 0.7), (b, -0.25)], lora_name="mix")

    for req in (a, LoRARequest("style", 1, "/path/style", lora_scale=0.4), comp):
        decoded = msgspec.msgpack.decode(msgspec.msgpack.encode(req), type=LoRARequest)
        assert decoded == req
        assert decoded.get_adapter_ids() == req.get_adapter_ids()
        assert decoded.get_adapter_scales() == req.get_adapter_scales()

    # Also verify round-trip through vLLM's MsgpackEncoder/MsgpackDecoder.
    encoder = MsgpackEncoder()
    core_req = EngineCoreRequest(
        request_id="r1",
        prompt_token_ids=None,
        mm_features=None,
        sampling_params=None,
        pooling_params=None,
        arrival_time=0.0,
        lora_request=comp,
        cache_salt=None,
        data_parallel_rank=None,
    )
    decoded_core = MsgpackDecoder(EngineCoreRequest).decode(encoder.encode(core_req))
    assert decoded_core.lora_request == comp

    (req,) = create_requests(num_requests=1, num_tokens=3, req_ids=["r1"])
    req.lora_request = comp
    new_req = NewRequestData.from_request(
        req, block_ids=([0],), prefill_token_ids=[1, 2, 3]
    )
    assert new_req.lora_request == comp


# ---------------------------------------------------------------------------
# Config, engine validation, and prefix-cache hashing
# ---------------------------------------------------------------------------


def test_lora_config_and_engine_args_max_loras_per_request():
    default_cfg = LoRAConfig()
    assert default_cfg.max_loras_per_request == 1

    valid_cfg = LoRAConfig(max_loras=4, max_loras_per_request=2)
    assert valid_cfg.max_loras_per_request == 2
    assert (
        LoRAConfig(max_loras=4, max_loras_per_request=1).compile_factors()
        != valid_cfg.compile_factors()
    )

    with pytest.raises(ValueError, match="max_loras_per_request"):
        LoRAConfig(max_loras=2, max_loras_per_request=3)

    with pytest.raises(ValueError, match="enable_tower_connector_lora"):
        LoRAConfig(
            max_loras=2,
            max_loras_per_request=2,
            enable_tower_connector_lora=True,
        )

    with pytest.raises(ValidationError):
        LoRAConfig(max_loras=2, max_loras_per_request=0)

    # CLI argument parsing.
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = parser.parse_args(
        [
            "--model",
            "facebook/opt-125m",
            "--enable-lora",
            "--max-loras",
            "4",
            "--max-loras-per-request",
            "3",
        ]
    )
    engine_args = EngineArgs.from_cli_args(args)
    assert engine_args.max_loras_per_request == 3


def test_input_processor_and_engine_lora_validation():
    proc = InputProcessor.__new__(InputProcessor)
    proc.lora_config = LoRAConfig(max_loras=4, max_loras_per_request=2)
    proc.renderer = SimpleNamespace(tokenizer=None)

    a = LoRARequest("a", 1, "/path/a")
    b = LoRARequest("b", 2, "/path/b")
    c = LoRARequest("c", 3, "/path/c")

    proc._validate_lora(a)
    proc._validate_lora(LoRARequest.from_adapters([(a, 0.7), (b, 0.5)]))

    with pytest.raises(VLLMValidationError, match="max_loras_per_request"):
        proc._validate_lora(LoRARequest.from_adapters([(a, 0.5), (b, 0.5), (c, 0.5)]))

    # Tower/connector LoRA rejects composite/scaled requests upfront and
    # _get_mm_identifier preserves execution order.
    proc.lora_config = LoRAConfig(
        max_loras=4, max_loras_per_request=1, enable_tower_connector_lora=True
    )
    proc._validate_lora(a)
    with pytest.raises(VLLMValidationError, match="enable_tower_connector_lora"):
        proc._validate_lora(LoRARequest("a", 1, "/path/a", lora_scale=0.5))

    id_a = proc._get_mm_identifier("img0", a)
    id_a_scaled = proc._get_mm_identifier(
        "img0", LoRARequest("a", 1, "/path/a", lora_scale=0.5)
    )
    id_ab = proc._get_mm_identifier(
        "img0", LoRARequest.from_adapters([(a, 0.7), (b, 0.5)])
    )
    id_ba = proc._get_mm_identifier(
        "img0", LoRARequest.from_adapters([(b, 0.5), (a, 0.7)])
    )
    assert id_a != id_a_scaled
    assert id_a != id_ab
    assert id_ab != id_ba

    # LLMEngine._register_lora registers all constituent adapters atomically.
    engine = LLMEngine.__new__(LLMEngine)
    engine._lora_adapters = {}
    comp_ab = LoRARequest.from_adapters([(a, 0.7), (b, 0.5)])
    engine._register_lora(comp_ab)
    assert engine._lora_adapters == {1: ("a", "/path/a"), 2: ("b", "/path/b")}

    # Collision on second constituent adapter fails without partial registration.
    c_new = LoRARequest("c", 3, "/path/c")
    b_conflict = LoRARequest("b_wrong", 2, "/path/wrong")
    with pytest.raises(VLLMValidationError, match="lora_int_id 2 is already used"):
        engine._register_lora(
            LoRARequest.from_adapters([(c_new, 0.5), (b_conflict, 0.5)])
        )
    assert 3 not in engine._lora_adapters

    # add_lora rejects composite requests on engine and worker managers.
    with pytest.raises(VLLMValidationError, match="composite"):
        engine.add_lora(comp_ab)

    async_llm = AsyncLLM.__new__(AsyncLLM)
    with pytest.raises(VLLMValidationError, match="composite"):
        asyncio.run(async_llm.add_lora(comp_ab))

    for mgr_cls in (WorkerLoRAManager, LRUCacheWorkerLoRAManager):
        mgr = mgr_cls.__new__(mgr_cls)
        with pytest.raises(ValueError, match="composite"):
            mgr.add_adapter(comp_ab)


@pytest.mark.parametrize("hash_fn", [sha256, sha256_cbor])
def test_prefix_cache_keys_for_composite_and_scaled_loras(hash_fn):
    a = LoRARequest("a", 1, "/path/a")
    b = LoRARequest("b", 2, "/path/b")
    tokens = [10, 20, 30, 40]

    def block_hash(lora_req: LoRARequest | None):
        (req,) = create_requests(num_requests=1, num_tokens=len(tokens), req_ids=["r"])
        req.lora_request = lora_req
        extra_keys, _ = generate_block_hash_extra_keys(req, 0, len(tokens), 0)
        return extra_keys, hash_block_tokens(hash_fn, None, tokens, extra_keys)

    # Single unit-scale adapter preserves legacy key shape and shares cache with
    # an equivalent 1-adapter unit-scale composite.
    ek_a, h_a = block_hash(a)
    ek_a1, h_a1 = block_hash(LoRARequest.from_adapters([(a, 1.0)]))
    assert ek_a == (("lora", "a", "/path/a"),)
    assert ek_a == ek_a1
    assert h_a == h_a1

    # Different scales produce distinct hashes and distinct event keys.
    ek_a_half, h_a_half = block_hash(LoRARequest("a", 1, "/path/a", lora_scale=0.5))
    assert h_a_half != h_a
    assert to_event_extra_keys([ek_a_half]) != to_event_extra_keys([ek_a])

    # Composite cache keys preserve execution order (FP accumulation is
    # non-associative), and swapping scales produces a different hash.
    ek_ab, h_ab = block_hash(LoRARequest.from_adapters([(a, 0.7), (b, 0.5)]))
    ek_ab_repeat, h_ab_repeat = block_hash(
        LoRARequest.from_adapters([(a, 0.7), (b, 0.5)])
    )
    ek_ba, h_ba = block_hash(LoRARequest.from_adapters([(b, 0.5), (a, 0.7)]))
    ek_ab_swapped, h_ab_swapped = block_hash(
        LoRARequest.from_adapters([(a, 0.5), (b, 0.7)])
    )
    assert ek_ab == ek_ab_repeat
    assert h_ab == h_ab_repeat
    assert ek_ab != ek_ba
    assert h_ab != h_ba
    assert h_ab != h_ab_swapped
    assert h_ab != h_a
    assert to_event_extra_keys([ek_ab]) != to_event_extra_keys([ek_ab_swapped])


# ---------------------------------------------------------------------------
# Scheduler slot accounting
# ---------------------------------------------------------------------------


def test_scheduler_max_loras_accounting_with_composite_requests(monkeypatch):
    if current_platform.is_cpu() and not triton_utils.HAS_TRITON:
        monkeypatch.setattr(triton_utils, "HAS_TRITON", True)

    scheduler = create_scheduler(
        lora_config=LoRAConfig(max_loras=3, max_loras_per_request=2)
    )
    a = LoRARequest("a", 1, "/path/a")
    b = LoRARequest("b", 2, "/path/b")
    c = LoRARequest("c", 3, "/path/c")
    d = LoRARequest("d", 4, "/path/d")

    r1, r2, r3, r4 = create_requests(
        num_requests=4, num_tokens=4, req_ids=["r1", "r2", "r3", "r4"]
    )
    r1.lora_request = LoRARequest.from_adapters([(a, 0.7), (b, 0.5)])  # {1, 2}
    r2.lora_request = LoRARequest.from_adapters([(c, 0.6), (d, 0.4)])  # {3, 4} (+2)
    r3.lora_request = LoRARequest.from_adapters([(b, 0.2), (c, 0.8)])  # {2, 3} (+1)

    # Add r1 ({1, 2}) and r2 ({3, 4}): max_loras=3 allows r1 (2 adapters) but
    # must NOT admit r2 because 2 + 2 = 4 > 3.
    scheduler.add_request(r1)
    scheduler.add_request(r2)
    out = scheduler.schedule()
    assert set(out.num_scheduled_tokens.keys()) == {"r1"}
    assert r1.status == RequestStatus.RUNNING
    assert r2.status == RequestStatus.WAITING

    scheduler.update_from_output(
        out,
        ModelRunnerOutput(
            req_ids=["r1"],
            sampled_token_ids=[[42]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )

    # Add r3 ({2, 3}): overlaps with adapter 2, so only 1 new adapter (3) is
    # needed -> 2 + 1 = 3 <= max_loras -> admitted alongside r1 while r2 waits!
    scheduler.add_request(r3)
    out2 = scheduler.schedule()
    assert set(out2.num_scheduled_tokens.keys()) == {"r1", "r3"}
    assert r3.status == RequestStatus.RUNNING
    assert r2.status == RequestStatus.WAITING

    scheduler.update_from_output(
        out2,
        ModelRunnerOutput(
            req_ids=["r1", "r3"],
            sampled_token_ids=[[43], [44]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )

    # Add r4 (base request, no LoRA): consumes 0 LoRA slots, admitted freely.
    r4.lora_request = None
    scheduler.add_request(r4)
    out3 = scheduler.schedule()
    assert set(out3.num_scheduled_tokens.keys()) == {"r1", "r3", "r4"}
    assert r2.status == RequestStatus.WAITING

    # Finish r1 and r3 so slots free up; now r2 ({3, 4}) is admitted.
    scheduler.finish_requests(["r1", "r3"], RequestStatus.FINISHED_STOPPED)
    out4 = scheduler.schedule()
    assert "r2" in out4.num_scheduled_tokens
    assert r2.status == RequestStatus.RUNNING


# ---------------------------------------------------------------------------
# Worker LoraState & dummy LoRA selection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("state_cls", [GpuLoraState, CpuLoraState])
def test_lora_state_multi_slot_and_fast_path(state_cls):
    a = LoRARequest("a", 1, "/path/a")
    b = LoRARequest("b", 2, "/path/b")
    c = LoRARequest("c", 3, "/path/c")

    state = state_cls(max_num_reqs=4, max_loras_per_request=3)

    # Fast path: single-adapter unit-scale + base request produces None slot fields.
    state.add_request("r0", 0, a)
    state.add_request("r1", 1, None)
    mapping, active = state.make_lora_mapping(
        ["r0", "r1"],
        np.array([0, 1], dtype=np.int32),
        np.array([2, 3], dtype=np.int32),
        np.array([1, 1], dtype=np.int32),
    )
    assert mapping.index_mapping == (1, 1, 0, 0, 0)
    assert mapping.prompt_mapping == (1, 0)
    assert mapping.slot_index_mappings is None
    assert mapping.slot_index_scales is None
    assert active == {a}

    # Backward-compatible 3-tuple make_lora_inputs still works.
    p_map, t_map, act = state.make_lora_inputs(
        ["r0", "r1"],
        np.array([0, 1], dtype=np.int32),
        np.array([2, 3], dtype=np.int32),
        np.array([1, 1], dtype=np.int32),
    )
    assert p_map == (1, 0)
    assert t_map == (1, 1, 0, 0, 0)
    assert act == {a}

    # Mixed batch: 2-adapter composite, single scaled adapter sharing 'a', and base.
    comp_ab = LoRARequest.from_adapters([(a, 0.7), (b, 0.5)])
    a_scaled = LoRARequest("a", 1, "/path/a", lora_scale=0.3)
    state.add_request("r0", 0, comp_ab)
    state.add_request("r1", 1, a_scaled)
    state.add_request("r2", 2, None)

    mapping2, active2 = state.make_lora_mapping(
        ["r0", "r1", "r2"],
        np.array([0, 1, 2], dtype=np.int32),
        np.array([2, 1, 2], dtype=np.int32),
        np.array([1, 2, 1], dtype=np.int32),
    )
    assert mapping2.num_slots == 2
    assert mapping2.slot_index_mappings == (
        (1, 1, 1, 0, 0),
        (2, 2, 0, 0, 0),
    )
    assert mapping2.slot_prompt_mappings == (
        (1, 1, 1, 0),
        (2, 0, 0, 0),
    )
    assert mapping2.slot_index_scales is not None
    assert np.allclose(mapping2.slot_index_scales[0], [0.7, 0.7, 0.3, 0.0, 0.0])
    assert np.allclose(mapping2.slot_index_scales[1], [0.5, 0.5, 0.0, 0.0, 0.0])
    assert mapping2.slot_prompt_scales is not None
    assert np.allclose(mapping2.slot_prompt_scales[0], [0.7, 0.3, 0.3, 0.0])
    assert np.allclose(mapping2.slot_prompt_scales[1], [0.5, 0.0, 0.0, 0.0])
    # Active set holds leaf single adapters (a and b), deduplicated.
    assert active2 == {a, b}
    assert all(not lr.is_composite for lr in active2)

    if state_cls is GpuLoraState:
        cfg = LoRAConfig(max_loras=4, max_loras_per_request=3)
        # 2 distinct adapters (a, b) + 1 no-LoRA request (r2) => 3.
        assert (
            get_num_active_loras_for_dispatch(cfg, state, ["r0", "r1", "r2"], False)
            == 3
        )
        assert state.has_custom_scales(["r0"])
        assert state.has_custom_scales(["r1"])
        assert not state.has_custom_scales(["r2"])

    # Exceeding max_loras_per_request fails loudly.
    d = LoRARequest("d", 4, "/path/d")
    with pytest.raises(ValueError, match="max_loras_per_request"):
        state.add_request(
            "r3", 3, LoRARequest.from_adapters([(a, 1.0), (b, 1.0), (c, 1.0), (d, 1.0)])
        )

    # Reusing req_index for a base request clears all slots.
    state.add_request("r0_new", 0, None)
    assert np.all(state.slot_lora_ids[0] == 0)
    assert np.all(state.slot_lora_scales[0] == 0.0)


def test_maybe_select_dummy_loras_populates_multi_slot():
    mock_manager = MagicMock()

    class DummyRunner(LoRAModelRunnerMixin):
        def __init__(self):
            self.lora_manager = mock_manager

    runner = DummyRunner()
    cfg = LoRAConfig(max_loras=3, max_loras_per_request=2)

    # Single request with 3 active LoRAs + no-LoRA case (max_loras + 1 = 4) still
    # populates 2 distinct active slots on that single request.
    with runner.maybe_select_dummy_loras(
        cfg,
        num_scheduled_tokens=np.array([4], dtype=np.int32),
        num_sampled_tokens=np.array([1], dtype=np.int32),
        num_active_loras=4,
    ):
        pass

    mock_manager.set_active_adapters.assert_called_once()
    requests_arg, mapping_arg = mock_manager.set_active_adapters.call_args[0]
    assert len(requests_arg) == 3
    assert mapping_arg.num_slots == 2
    assert mapping_arg.slot_index_mappings == ((1, 1, 1, 1), (2, 2, 2, 2))
    assert mapping_arg.slot_prompt_mappings == ((1,), (2,))

    # Multi-request batch including a no-LoRA request keeps no-LoRA at 0 in all slots.
    mock_manager.set_active_adapters.reset_mock()
    with runner.maybe_select_dummy_loras(
        cfg,
        num_scheduled_tokens=np.array([2, 1, 1, 2], dtype=np.int32),
        num_sampled_tokens=np.array([1, 1, 1, 1], dtype=np.int32),
        num_active_loras=4,
    ):
        pass
    _, mapping_arg2 = mock_manager.set_active_adapters.call_args[0]
    assert mapping_arg2.slot_prompt_mappings == ((1, 2, 3, 0), (2, 3, 1, 0))
    assert mapping_arg2.slot_prompt_scales == (
        (1.0, 1.0, 1.0, 0.0),
        (1.0, 1.0, 1.0, 0.0),
    )


# ---------------------------------------------------------------------------
# PunicaWrapperCPU exact numerical correctness & PEFT cat-merge oracle
# ---------------------------------------------------------------------------


def _manual_multi_lora_linear_ref(
    base_y: torch.Tensor,
    x: torch.Tensor,
    lora_a_stacked: tuple[torch.Tensor, ...],
    lora_b_stacked: tuple[torch.Tensor, ...],
    output_slices: tuple[int, ...],
    token_slots: list[list[tuple[int, float]]],
    add_inputs: bool = True,
) -> torch.Tensor:
    out = base_y.clone() if add_inputs else torch.zeros_like(base_y)
    for tok_idx, entries in enumerate(token_slots):
        if not entries and not add_inputs:
            out[tok_idx] = base_y[tok_idx]
            continue
        offset = 0
        for s_idx, slice_size in enumerate(output_slices):
            delta = torch.zeros(slice_size, dtype=torch.float32)
            for adapter_slot, scale in entries:
                a = lora_a_stacked[s_idx][adapter_slot, 0].float()  # [r, in]
                b = lora_b_stacked[s_idx][adapter_slot, 0].float()  # [out_s, r]
                delta = delta + scale * (b @ (a @ x[tok_idx].float()))
            if add_inputs:
                out[tok_idx, offset : offset + slice_size] = (
                    out[tok_idx, offset : offset + slice_size].float() + delta
                ).to(out.dtype)
            else:
                out[tok_idx, offset : offset + slice_size] = delta.to(out.dtype)
            offset += slice_size
    return out


@pytest.mark.parametrize("is_prefill", [True, False])
@pytest.mark.parametrize("add_inputs", [True, False])
def test_punica_cpu_multi_lora_linear_matches_reference_and_cat_merge(
    is_prefill: bool, add_inputs: bool
):
    torch.manual_seed(42)
    max_loras = 4
    max_slots = 3
    hidden = 16
    rank = 4
    output_slices = (8, 6, 6)
    total_out = sum(output_slices)
    num_tokens = 7

    # 3 distinct adapters (IDs 1, 2, 3 -> slots 0, 1, 2).
    lora_a = tuple(
        torch.randn(max_loras, 1, rank, hidden, dtype=torch.float32)
        for _ in output_slices
    )
    lora_b = tuple(
        torch.randn(max_loras, 1, s, rank, dtype=torch.float32) for s in output_slices
    )
    x = torch.randn(num_tokens, hidden, dtype=torch.float32)
    base_y = torch.randn(num_tokens, total_out, dtype=torch.float32)

    # Heterogeneous batch:
    # tok 0,1: adapters (1, 0.7), (2, 0.5)
    # tok 2,3: adapters (1, -0.5), (2, 0.0), (3, 1.25)
    # tok 4:   adapter  (2, 0.3)
    # tok 5,6: no LoRA
    mapping = LoRAMapping(
        index_mapping=(1, 1, 1, 1, 2, 0, 0),
        prompt_mapping=(1, 1, 2, 0),
        is_prefill=is_prefill,
        slot_index_mappings=(
            (1, 1, 1, 1, 2, 0, 0),
            (2, 2, 2, 2, 0, 0, 0),
            (0, 0, 3, 3, 0, 0, 0),
        ),
        slot_prompt_mappings=(
            (1, 1, 2, 0),
            (2, 2, 0, 0),
            (0, 3, 0, 0),
        ),
        slot_index_scales=(
            (0.7, 0.7, -0.5, -0.5, 0.3, 0.0, 0.0),
            (0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 1.25, 1.25, 0.0, 0.0, 0.0),
        ),
        slot_prompt_scales=(
            (0.7, -0.5, 0.3, 0.0),
            (0.5, 0.0, 0.0, 0.0),
            (0.0, 1.25, 0.0, 0.0),
        ),
    )

    wrapper = PunicaWrapperCPU(
        max_num_batched_tokens=32,
        max_batches=8,
        device="cpu",
        lora_config=LoRAConfig(max_loras=max_loras, max_loras_per_request=max_slots),
    )
    wrapper.update_metadata(
        mapping,
        lora_index_to_id=[1, 2, 3, None],
        max_loras=max_loras + 1,
        vocab_size=64,
    )

    y = base_y.clone()
    wrapper.add_lora_linear(
        y, x, lora_a, lora_b, 1.0, output_slices, add_inputs=add_inputs
    )

    token_slots = [
        [(0, 0.7), (1, 0.5)],
        [(0, 0.7), (1, 0.5)],
        [(0, -0.5), (1, 0.0), (2, 1.25)],
        [(0, -0.5), (1, 0.0), (2, 1.25)],
        [(1, 0.3)],
        [],
        [],
    ]
    expected = _manual_multi_lora_linear_ref(
        base_y, x, lora_a, lora_b, output_slices, token_slots, add_inputs=add_inputs
    )
    torch.testing.assert_close(y, expected, atol=1e-5, rtol=1e-5)

    # PEFT `cat`-merge oracle for tokens 0..1 (adapters 0 @ 0.7, 1 @ 0.5):
    # Concatenating A_cat = [A_0; A_1] and B_cat = [0.7 * B_0, 0.5 * B_1]
    # gives a single rank-(2*r) adapter whose output must match runtime composition.
    if add_inputs:
        offset = 0
        for s_idx, slice_size in enumerate(output_slices):
            a_cat = torch.cat(
                [lora_a[s_idx][0, 0], lora_a[s_idx][1, 0]], dim=0
            )  # [2r, hidden]
            b_cat = torch.cat(
                [0.7 * lora_b[s_idx][0, 0], 0.5 * lora_b[s_idx][1, 0]], dim=1
            )  # [slice_size, 2r]
            cat_out = (
                base_y[:2, offset : offset + slice_size] + (x[:2] @ a_cat.T) @ b_cat.T
            )
            torch.testing.assert_close(
                y[:2, offset : offset + slice_size], cat_out, atol=1e-5, rtol=1e-5
            )
            offset += slice_size


def test_punica_cpu_logits_and_embedding_and_stale_scale_reset():
    torch.manual_seed(7)
    max_loras = 3
    vocab_size = 12
    hidden = 8
    rank = 3

    wrapper = PunicaWrapperCPU(
        max_num_batched_tokens=16,
        max_batches=4,
        device="cpu",
        lora_config=LoRAConfig(max_loras=max_loras, max_loras_per_request=2),
    )

    # Step 1: composite with custom scales.
    mapping1 = LoRAMapping(
        index_mapping=(1, 2, 0),
        prompt_mapping=(1, 2, 0),
        is_prefill=True,
        slot_index_mappings=((1, 2, 0), (2, 0, 0)),
        slot_prompt_mappings=((1, 2, 0), (2, 0, 0)),
        slot_index_scales=((0.6, 0.4, 0.0), (0.8, 0.0, 0.0)),
        slot_prompt_scales=((0.6, 0.4, 0.0), (0.8, 0.0, 0.0)),
    )
    wrapper.update_metadata(
        mapping1,
        lora_index_to_id=[1, 2, None],
        max_loras=max_loras + 1,
        vocab_size=vocab_size,
    )

    # Test add_lora_logits.
    x_logits = torch.randn(3, hidden, dtype=torch.float32)
    lora_a_logits = torch.randn(max_loras, 1, rank, hidden, dtype=torch.float32)
    lora_b_logits = torch.randn(max_loras, 1, vocab_size, rank, dtype=torch.float32)
    y_logits = torch.randn(3, vocab_size, dtype=torch.float32)
    expected_logits = y_logits.clone()
    expected_logits[0] += 0.6 * (
        lora_b_logits[0, 0] @ (lora_a_logits[0, 0] @ x_logits[0])
    ) + 0.8 * (lora_b_logits[1, 0] @ (lora_a_logits[1, 0] @ x_logits[0]))
    expected_logits[1] += 0.4 * (
        lora_b_logits[1, 0] @ (lora_a_logits[1, 0] @ x_logits[1])
    )
    wrapper.add_lora_logits(y_logits, x_logits, lora_a_logits, lora_b_logits, 1.0)
    torch.testing.assert_close(y_logits, expected_logits, atol=1e-5, rtol=1e-5)

    # Test multi-slot embedding lookup + add_lora_embedding.
    embed_a = torch.randn(max_loras * vocab_size, rank, dtype=torch.float32)
    embed_b = torch.randn(max_loras, 1, hidden, rank, dtype=torch.float32)
    token_ids = torch.tensor([2, 5, 1], dtype=torch.long)
    y_emb = torch.randn(3, hidden, dtype=torch.float32)
    expected_emb = y_emb.clone()
    expected_emb[0] += 0.6 * (embed_b[0, 0] @ embed_a[0 * vocab_size + 2]) + 0.8 * (
        embed_b[1, 0] @ embed_a[1 * vocab_size + 2]
    )
    expected_emb[1] += 0.4 * (embed_b[1, 0] @ embed_a[1 * vocab_size + 5])

    for slot_idx in range(wrapper.num_active_slots):
        emb_idx = wrapper.get_embeddings_indices(3, slot_idx)
        lora_a_emb = torch.nn.functional.embedding(token_ids + emb_idx, embed_a)
        wrapper.add_lora_embedding(
            y_emb, lora_a_emb, embed_b, add_inputs=True, slot_idx=slot_idx
        )
    torch.testing.assert_close(y_emb, expected_emb, atol=1e-5, rtol=1e-5)

    # FP16 embedding scaling with scale > FP16 max (65504) must scale in FP32
    # so small embeddings produce finite, accurate results instead of inf.
    mapping_fp16 = LoRAMapping(
        index_mapping=(1,),
        prompt_mapping=(1,),
        is_prefill=False,
        slot_index_scales=((100000.0,),),
        slot_prompt_scales=((100000.0,),),
    )
    wrapper.update_metadata(
        mapping_fp16,
        lora_index_to_id=[1, 2, None],
        max_loras=max_loras + 1,
        vocab_size=vocab_size,
    )
    x_fp16 = torch.full((1, rank), 1e-5, dtype=torch.float16)
    b_fp16 = torch.ones((max_loras, 1, hidden, rank), dtype=torch.float16)
    y_fp16 = torch.zeros((1, hidden), dtype=torch.float16)
    wrapper.add_lora_embedding(y_fp16, x_fp16, b_fp16, add_inputs=True, slot_idx=0)
    assert torch.isfinite(y_fp16).all()
    expected_fp16 = torch.full((1, hidden), float(rank), dtype=torch.float16)
    torch.testing.assert_close(y_fp16, expected_fp16, atol=1e-2, rtol=1e-2)

    # Step 2: revert to plain single-adapter mapping (no scales) and verify no
    # stale scale or slot-1 leakage from Step 1.
    mapping2 = LoRAMapping(
        index_mapping=(1, 0, 2),
        prompt_mapping=(1, 0, 2),
        is_prefill=True,
    )
    wrapper.update_metadata(
        mapping2,
        lora_index_to_id=[1, 2, None],
        max_loras=max_loras + 1,
        vocab_size=vocab_size,
    )
    assert wrapper.num_active_slots == 1
    assert not wrapper.has_multi_slot
    assert not wrapper.has_custom_scales
    assert torch.all(wrapper.get_token_lora_scales(0, 3) == 1.0)
    assert torch.all(wrapper.get_token_lora_scales(1, 3) == 0.0)

    y_logits2 = torch.zeros(3, vocab_size, dtype=torch.float32)
    wrapper.add_lora_logits(y_logits2, x_logits, lora_a_logits, lora_b_logits, 1.0)
    expected_logits2 = torch.zeros_like(y_logits2)
    expected_logits2[0] = lora_b_logits[0, 0] @ (lora_a_logits[0, 0] @ x_logits[0])
    expected_logits2[2] = lora_b_logits[1, 0] @ (lora_a_logits[1, 0] @ x_logits[2])
    torch.testing.assert_close(y_logits2, expected_logits2, atol=1e-5, rtol=1e-5)


def test_punica_fail_closed_for_unsupported_backend_and_full_linear():
    class DummyUnportedWrapper(PunicaWrapperBase):
        def add_shrink(self, *args, **kwargs):
            pass

        def add_expand(self, *args, **kwargs):
            pass

        def add_lora_embedding(self, *args, **kwargs):
            pass

        def add_lora_linear(self, *args, **kwargs):
            pass

        def add_lora_logits(self, *args, **kwargs):
            pass

        def apply_lora_full_linear(self, *args, **kwargs):
            pass

    unported = DummyUnportedWrapper(16, 4, "cpu", max_loras_per_request=2)
    multi_mapping = LoRAMapping(
        index_mapping=(1, 0),
        prompt_mapping=(1, 0),
        slot_index_mappings=((1, 0), (2, 0)),
        slot_prompt_mappings=((1, 0), (2, 0)),
    )
    with pytest.raises(NotImplementedError, match="does not support"):
        unported.update_metadata(multi_mapping, [1, 2], 3, 32)

    # Exceeding configured max_loras_per_request on CPU wrapper fails loudly.
    cpu_1slot = PunicaWrapperCPU(
        16, 4, "cpu", lora_config=LoRAConfig(max_loras=2, max_loras_per_request=1)
    )
    with pytest.raises(ValueError, match="exceeding max_loras_per_request"):
        cpu_1slot.update_metadata(multi_mapping, [1, 2], 3, 32)

    # apply_lora_full_linear fails loudly when multi-slot is active.
    cpu_2slot = PunicaWrapperCPU(
        16, 4, "cpu", lora_config=LoRAConfig(max_loras=2, max_loras_per_request=2)
    )
    cpu_2slot.update_metadata(multi_mapping, [1, 2], 3, 32)
    with pytest.raises(NotImplementedError, match="classification head"):
        cpu_2slot.apply_lora_full_linear(
            torch.zeros(2, 4),
            torch.zeros(2, 8),
            torch.zeros(2, 1, 4, 8),
            torch.zeros(2, 4),
            torch.tensor([True, True]),
        )


# ---------------------------------------------------------------------------
# PunicaWrapperGPU metadata & multi-slot orchestration (on CPU w/ ref ops)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("add_inputs", [True, False])
def test_punica_gpu_multi_slot_metadata_and_reference_loop_on_cpu(
    monkeypatch, add_inputs: bool
):
    torch.manual_seed(123)
    max_loras = 3
    max_slots = 3
    hidden = 12
    rank = 4
    output_slices = (6, 6)
    num_tokens = 4

    def fake_lora_shrink(
        inputs,
        lora_a_weights,
        output_tensor,
        token_lora_mapping,
        token_indices_sorted_by_lora_ids,
        num_tokens_per_lora,
        lora_token_start_loc,
        lora_ids,
        no_lora_flag_cpu,
        num_active_loras,
        scaling,
        split_k_reduction="atomic",
    ):
        if no_lora_flag_cpu.item():
            return
        output_tensor.zero_()
        for s_idx, w in enumerate(lora_a_weights):
            bgmv_shrink(
                inputs, w, output_tensor[s_idx], token_lora_mapping.long(), scaling
            )

    def fake_lora_expand(
        inputs,
        lora_b_weights,
        output_tensor,
        token_lora_mapping,
        token_indices_sorted_by_lora_ids,
        num_tokens_per_lora,
        lora_token_start_loc,
        lora_ids,
        no_lora_flag_cpu,
        num_active_loras,
        offset_start=0,
        add_inputs=False,
    ):
        if no_lora_flag_cpu.item():
            return
        offset = offset_start
        for s_idx, w in enumerate(lora_b_weights):
            slice_size = w.shape[-2]
            bgmv_expand_slice(
                inputs[s_idx],
                w,
                output_tensor,
                token_lora_mapping.long(),
                offset,
                slice_size,
                add_inputs=add_inputs,
            )
            offset += slice_size

    monkeypatch.setattr(punica_gpu_mod, "lora_shrink", fake_lora_shrink, raising=False)
    monkeypatch.setattr(punica_gpu_mod, "lora_expand", fake_lora_expand, raising=False)

    # Ensure LoRAKernelMeta is available even when triton is not installed.
    from vllm.lora.ops.triton_ops.lora_kernel_metadata import LoRAKernelMeta

    monkeypatch.setattr(punica_gpu_mod, "LoRAKernelMeta", LoRAKernelMeta, raising=False)

    wrapper = PunicaWrapperGPU(
        max_num_batched_tokens=16,
        max_batches=4,
        device="cpu",
        lora_config=LoRAConfig(max_loras=max_loras, max_loras_per_request=max_slots),
    )

    # 2 active slots out of max_slots=3 -> slot 2 must be reset with no_lora_flag=True.
    mapping = LoRAMapping(
        index_mapping=(1, 1, 2, 0),
        prompt_mapping=(1, 2, 0),
        is_prefill=True,
        slot_index_mappings=((1, 1, 2, 0), (2, 2, 0, 0)),
        slot_prompt_mappings=((1, 2, 0), (2, 0, 0)),
        slot_index_scales=((0.7, 0.7, 0.4, 0.0), (0.5, 0.5, 0.0, 0.0)),
        slot_prompt_scales=((0.7, 0.4, 0.0), (0.5, 0.0, 0.0)),
    )
    wrapper.update_metadata(
        mapping, lora_index_to_id=[1, 2, None], max_loras=max_loras + 1, vocab_size=32
    )

    assert wrapper.num_active_slots == 2
    assert not wrapper._slot_token_mapping_metas[0].no_lora_flag_cpu.item()
    assert not wrapper._slot_token_mapping_metas[1].no_lora_flag_cpu.item()
    assert wrapper._slot_token_mapping_metas[2].no_lora_flag_cpu.item()
    assert torch.all(wrapper._slot_token_mapping_metas[2].active_lora_ids == -1)

    lora_a = tuple(
        torch.randn(max_loras, 1, rank, hidden, dtype=torch.float32)
        for _ in output_slices
    )
    lora_b = tuple(
        torch.randn(max_loras, 1, s, rank, dtype=torch.float32) for s in output_slices
    )
    x = torch.randn(num_tokens, hidden, dtype=torch.float32)
    base_y = torch.randn(num_tokens, sum(output_slices), dtype=torch.float32)
    y = base_y.clone()

    wrapper.add_lora_linear(
        y, x, lora_a, lora_b, 1.0, output_slices, add_inputs=add_inputs
    )

    expected = _manual_multi_lora_linear_ref(
        base_y,
        x,
        lora_a,
        lora_b,
        output_slices,
        [[(0, 0.7), (1, 0.5)], [(0, 0.7), (1, 0.5)], [(1, 0.4)], []],
        add_inputs=add_inputs,
    )
    torch.testing.assert_close(y, expected, atol=1e-5, rtol=1e-5)

    # Multi-slot mapping with omitted scales must default every active slot to
    # 1.0 (not 0.0) and match CPU execution.
    mapping_unscaled = LoRAMapping(
        index_mapping=(1, 1, 2, 0),
        prompt_mapping=(1, 2, 0),
        is_prefill=True,
        slot_index_mappings=((1, 1, 2, 0), (2, 2, 0, 0)),
        slot_prompt_mappings=((1, 2, 0), (2, 0, 0)),
    )
    wrapper.update_metadata(
        mapping_unscaled,
        lora_index_to_id=[1, 2, None],
        max_loras=max_loras + 1,
        vocab_size=32,
    )
    assert torch.all(wrapper.get_token_lora_scales(0, num_tokens) == 1.0)
    assert torch.all(wrapper.get_token_lora_scales(1, num_tokens) == 1.0)
    assert torch.all(wrapper.get_token_lora_scales(2, num_tokens) == 0.0)
    y_unscaled = base_y.clone()
    wrapper.add_lora_linear(
        y_unscaled, x, lora_a, lora_b, 1.0, output_slices, add_inputs=add_inputs
    )
    expected_unscaled = _manual_multi_lora_linear_ref(
        base_y,
        x,
        lora_a,
        lora_b,
        output_slices,
        [[(0, 1.0), (1, 1.0)], [(0, 1.0), (1, 1.0)], [(1, 1.0)], []],
        add_inputs=add_inputs,
    )
    torch.testing.assert_close(y_unscaled, expected_unscaled, atol=1e-5, rtol=1e-5)

    # FusedMoE LoRA fails loudly when multi-slot or custom scales are active.
    with pytest.raises(NotImplementedError, match="FusedMoE"):
        wrapper.add_lora_fused_moe(
            y=torch.zeros(1),
            x=torch.zeros(1, 1),
            lora_a_stacked=(torch.zeros(1),),
            lora_b_stacked=(torch.zeros(1),),
            topk_weights=torch.zeros(1),
            sorted_token_ids=None,
            expert_ids=torch.zeros(1),
            num_tokens_post_padded=None,
            max_lora_rank=4,
            top_k_num=1,
            shrink_config={},
            expand_config={},
            adapter_enabled=torch.zeros(1),
        )
