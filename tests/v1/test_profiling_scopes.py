# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib
import sys
import types
from collections.abc import Callable
from concurrent.futures import Future
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest

from vllm.v1 import utils as v1_utils
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput


@pytest.fixture
def should_do_global_cleanup_after_test() -> bool:
    return False


def _make_scheduler_output(
    num_scheduled_tokens: dict[str, int],
    new_req_ids: list[str] | None = None,
    cached_context_req_ids: list[str] | None = None,
    cached_generation_req_ids: list[str] | None = None,
) -> SchedulerOutput:
    new_req_ids = new_req_ids or []
    cached_context_req_ids = cached_context_req_ids or []
    cached_generation_req_ids = cached_generation_req_ids or []

    cached_req_ids = cached_context_req_ids + cached_generation_req_ids
    cached_num_output_tokens = (
        [0] * len(cached_context_req_ids) + [1] * len(cached_generation_req_ids)
    )
    cached_reqs = CachedRequestData(
        req_ids=cached_req_ids,
        resumed_req_ids=set(),
        new_token_ids=[[] for _ in cached_req_ids],
        all_token_ids={},
        new_block_ids=[None for _ in cached_req_ids],
        num_computed_tokens=[0 for _ in cached_req_ids],
        num_output_tokens=cached_num_output_tokens,
    )

    return SchedulerOutput(
        scheduled_new_reqs=[
            SimpleNamespace(req_id=req_id) for req_id in new_req_ids
        ],
        scheduled_cached_reqs=cached_reqs,
        num_scheduled_tokens=num_scheduled_tokens,
        total_num_scheduled_tokens=sum(num_scheduled_tokens.values()),
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


def _recorded_scope_factory(scope_names: list[str]) -> Callable[[str], object]:
    @contextmanager
    def _record_scope(name: str):
        scope_names.append(name)
        yield

    return _record_scope


def _import_gpu_worker(monkeypatch):
    stub_module = types.ModuleType("vllm.model_executor.warmup.kernel_warmup")
    stub_module.kernel_warmup = lambda *args, **kwargs: None
    worker_utils_stub = types.ModuleType("vllm.v1.worker.utils")
    worker_utils_stub.is_residual_scattered_for_sp = lambda *args, **kwargs: False
    worker_utils_stub.request_memory = lambda *args, **kwargs: None
    model_loader_stub = types.ModuleType("vllm.model_executor.model_loader")
    model_loader_stub.TensorizerLoader = object
    gpu_warmup_stub = types.ModuleType("vllm.v1.worker.gpu.warmup")
    gpu_warmup_stub.warmup_kernels = lambda *args, **kwargs: None
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.warmup.kernel_warmup",
        stub_module,
    )
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.utils", worker_utils_stub)
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.model_loader",
        model_loader_stub,
    )
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.gpu.warmup", gpu_warmup_stub)
    sys.modules.pop("vllm.v1.worker.gpu_worker", None)
    return importlib.import_module("vllm.v1.worker.gpu_worker")


@pytest.mark.parametrize(
    "scheduler_output,expected",
    [
        (
            _make_scheduler_output(
                num_scheduled_tokens={"new_req": 4},
                new_req_ids=["new_req"],
            ),
            "prefill_batch",
        ),
        (
            _make_scheduler_output(
                num_scheduled_tokens={"decode_req": 2},
                cached_generation_req_ids=["decode_req"],
            ),
            "decode_batch",
        ),
        (
            _make_scheduler_output(
                num_scheduled_tokens={"new_req": 3, "decode_req": 2},
                new_req_ids=["new_req"],
                cached_generation_req_ids=["decode_req"],
            ),
            "mixed_batch",
        ),
        (
            _make_scheduler_output(num_scheduled_tokens={}),
            "empty_batch",
        ),
    ],
)
def test_classify_batch_stage(scheduler_output: SchedulerOutput, expected: str):
    assert v1_utils.classify_batch_stage(scheduler_output) == expected


def test_engine_core_step_emits_scheduler_scope(monkeypatch):
    import vllm.v1.engine.core as engine_core_mod

    scheduler_output = _make_scheduler_output(
        num_scheduled_tokens={"req": 1},
        new_req_ids=["req"],
    )
    scope_names: list[str] = []
    future: Future[None] = Future()
    future.set_result(None)

    monkeypatch.setattr(
        engine_core_mod,
        "record_function_or_nullcontext",
        _recorded_scope_factory(scope_names),
    )

    scheduler = SimpleNamespace(
        has_requests=lambda: True,
        schedule=lambda: scheduler_output,
        get_grammar_bitmask=lambda _: None,
        update_from_output=lambda _, model_output: {"model_output": model_output},
    )
    model_executor = SimpleNamespace(
        execute_model=lambda _, non_block=True: future,
        sample_tokens=lambda _: "sampled",
    )
    engine = SimpleNamespace(
        scheduler=scheduler,
        model_executor=model_executor,
        log_error_detail=lambda _: nullcontext(),
        log_iteration_details=lambda _: nullcontext(),
        _process_aborts_queue=lambda: None,
    )

    outputs, model_executed = engine_core_mod.EngineCore.step(engine)

    assert outputs == {"model_output": "sampled"}
    assert model_executed is True
    assert scope_names == ["scheduler_step"]


def test_worker_sample_tokens_emits_sampling_scope(monkeypatch):
    gpu_worker_mod = _import_gpu_worker(monkeypatch)

    scope_names: list[str] = []
    monkeypatch.setattr(
        gpu_worker_mod,
        "record_function_or_nullcontext",
        _recorded_scope_factory(scope_names),
    )

    worker = SimpleNamespace(
        model_runner=SimpleNamespace(sample_tokens=lambda _: "sampled")
    )

    assert gpu_worker_mod.Worker.sample_tokens(worker, None) == "sampled"
    assert scope_names == ["sampling"]


@pytest.mark.parametrize(
    "scheduler_output,expected",
    [
        (
            _make_scheduler_output(
                num_scheduled_tokens={"req": 2},
                new_req_ids=["req"],
            ),
            "prefill_batch",
        ),
        (
            _make_scheduler_output(
                num_scheduled_tokens={"req": 1},
                cached_generation_req_ids=["req"],
            ),
            "decode_batch",
        ),
    ],
)
def test_worker_execute_model_emits_batch_stage_scope(
    monkeypatch,
    scheduler_output: SchedulerOutput,
    expected: str,
):
    gpu_worker_mod = _import_gpu_worker(monkeypatch)

    scope_names: list[str] = []
    monkeypatch.setattr(
        gpu_worker_mod,
        "record_function_or_nullcontext",
        _recorded_scope_factory(scope_names),
    )
    monkeypatch.setattr(
        gpu_worker_mod,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True),
    )

    worker = SimpleNamespace(
        _pp_send_work=[],
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(
                pass_config=SimpleNamespace(enable_sp=False)
            ),
            parallel_config=SimpleNamespace(
                pipeline_parallel_size=1,
                distributed_executor_backend="mp",
            ),
        ),
        use_v2_model_runner=False,
        annotate_profile=lambda _: nullcontext(),
        model_runner=SimpleNamespace(execute_model=lambda *_: None),
    )

    assert gpu_worker_mod.Worker.execute_model(worker, scheduler_output) is None
    assert scope_names == [expected]
