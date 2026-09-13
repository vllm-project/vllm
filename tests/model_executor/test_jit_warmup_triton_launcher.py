# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Any

import pytest
import torch

from vllm.model_executor.warmup import jit_warmup_triton_helper
from vllm.model_executor.warmup.jit_warmup import (
    WarmupChoices,
    WarmupIntRange,
    kernel_launcher,
)
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    LaunchSpec,
    TritonJitKey,
    TritonWarmupTensor,
    VllmTritonJitKernel,
    triton_kernel_dispatcher_with_warmup,
    triton_warmup_inputs,
)
from vllm.triton_utils import tl, triton


@triton.jit
def _binder_test_kernel(x, value, BLOCK: tl.constexpr):
    pass


@triton.jit
def _second_binder_test_kernel(x, value, BLOCK: tl.constexpr):
    pass


@triton.jit
def _pointer_group_test_kernel(x_ptr, value, BLOCK: tl.constexpr):
    pass


class _FakeTritonKernel:
    arg_names: tuple[str, ...] = ("first", "second", "CONST")

    def __init__(self) -> None:
        self.warmup_calls: list[dict[str, Any]] = []
        self.runtime_calls: list[tuple[Any, tuple[Any, ...], dict[str, Any]]] = []

    def warmup(self, **kwargs: Any) -> None:
        self.warmup_calls.append(kwargs)

    def __getitem__(self, grid: Any) -> Any:
        def launch(*args: Any, **kwargs: Any) -> None:
            self.runtime_calls.append((grid, args, kwargs))

        return launch


def _patch_key_deriver(
    monkeypatch: pytest.MonkeyPatch,
    derive: Any,
) -> None:
    monkeypatch.setattr(
        jit_warmup_triton_helper,
        "_triton_key_deriver",
        lambda kernel: lambda kwargs: derive(kernel, kwargs),
    )


class _TestTritonKernel(VllmTritonJitKernel["_TestTritonKernel.CompileKey"]):
    kernel = _FakeTritonKernel()

    @dataclass(frozen=True)
    class CompileKey:
        value: int

    def dispatch(self, *, value: int) -> CompileKey:
        return self.CompileKey(value=value)

    def get_warmup_keys(self) -> list[CompileKey]:
        return self._trace_dispatch(self.dispatch)(value=1)

    def warmup_inputs(self, compile_key: CompileKey) -> dict[str, Any]:
        return dict(first="warmup", second=compile_key.value, runtime_launcher=None)

    @kernel_launcher
    def __call__(
        self,
        first: str,
        second: int,
        runtime_launcher: Any,
    ) -> LaunchSpec:
        return (2,), dict(
            CONST=7,
            _runtime_launcher=runtime_launcher,
            _runtime_launcher_arg_count=2,
        )


def test_triton_launcher_supports_compile_and_runtime_adapters() -> None:
    owner = _TestTritonKernel()
    owner.kernel.warmup_calls.clear()

    owner.compile(owner.CompileKey(value=1))
    assert owner.kernel.warmup_calls == [
        {"grid": (1,), "first": "warmup", "second": 1, "CONST": 7}
    ]

    runtime_calls: list[tuple[Any, ...]] = []

    def runtime_launcher(kernel: Any, grid: Any, *args: Any, **kwargs: Any) -> None:
        runtime_calls.append((kernel, grid, args, kwargs))

    owner("runtime", 2, runtime_launcher)
    assert runtime_calls == [(owner.kernel, (2,), ("runtime", 2), {"CONST": 7})]


def test_triton_launcher_supports_cpu_function_wrappers() -> None:
    calls: list[tuple[Any, ...]] = []

    def kernel(first: str, second: int, CONST: int) -> None:
        calls.append((first, second, CONST))

    class FuncWrapper:
        def __init__(self) -> None:
            self.func = kernel

        def __getitem__(self, _grid: Any) -> Any:
            return self.func

    class TestCpuKernel(_TestTritonKernel):
        kernel: Any = FuncWrapper()

    owner = TestCpuKernel()

    owner("runtime", 2, None)
    assert calls == [("runtime", 2, 7)]


def test_triton_kernel_decorator_returns_launcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel = _FakeTritonKernel()
    kernel.arg_names = (
        "FIRST_PTR",
        "FIRST_STRIDE0",
        "aliased_ptr",
        "aliased_stride_0",
        "SECOND",
        "CONST",
    )

    def warmup_inputs() -> dict[str, Any]:
        return dict(
            first=TritonWarmupTensor(torch.float32, shape=(2, 3), strides=(5, 1)),
            second=WarmupChoices(1, 2),
        )

    @triton_kernel_dispatcher_with_warmup(kernel=kernel, warmup_inputs=warmup_inputs)
    def launch(first: str, second: int = 2, config: int = 7) -> LaunchSpec:
        return (2,), dict(aliased_ptr=first, CONST=config)

    def fake_keys(kernel: Any, kwargs: Any) -> set[TritonJitKey]:
        return {TritonJitKey(id(kernel), "fake", 0, kwargs["SECOND"])}

    _patch_key_deriver(monkeypatch, fake_keys)

    keys = launch.get_warmup_keys()
    keys_with_config = launch.get_warmup_keys(vllm_config=object())
    assert keys_with_config == keys
    assert [dict(key.inputs)["second"] for key in keys] == [1, 2]
    launch.compile(keys[0])
    assert kernel.warmup_calls == [
        {
            "grid": (1,),
            "FIRST_PTR": TritonWarmupTensor(
                torch.float32, shape=(2, 3), strides=(5, 1)
            ),
            "FIRST_STRIDE0": 5,
            "aliased_ptr": TritonWarmupTensor(
                torch.float32, shape=(2, 3), strides=(5, 1)
            ),
            "aliased_stride_0": 5,
            "SECOND": 1,
            "CONST": 7,
        }
    ]
    first = torch.empty_strided((2, 3), (5, 1))
    launch(first)
    assert kernel.runtime_calls == [
        (
            (2,),
            (),
            {
                "FIRST_PTR": first,
                "FIRST_STRIDE0": 5,
                "aliased_ptr": first,
                "aliased_stride_0": 5,
                "SECOND": 2,
                "CONST": 7,
            },
        )
    ]
    assert launch.__name__ == "launch"
    with pytest.raises(TypeError, match="unexpected keyword"):
        launch(first, 1, 7, stale_constexpr=True)


def test_triton_kernel_decorator_exhausts_large_ranges_before_deduplication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel = _FakeTritonKernel()
    dispatched: list[int] = []

    def warmup_inputs() -> dict[str, Any]:
        tokens: Any = WarmupIntRange(1, 8193)
        return dict(first="warmup", second=tokens)

    def specialization(second: int) -> int:
        return 7 if second <= 97 else 8

    @triton_kernel_dispatcher_with_warmup(kernel=kernel, warmup_inputs=warmup_inputs)
    def dispatch(first: str, second: int) -> LaunchSpec:
        dispatched.append(second)
        return (second,), dict(CONST=specialization(second))

    def fake_keys(kernel: Any, kwargs: Any) -> set[TritonJitKey]:
        return {TritonJitKey(id(kernel), "fake", 0, kwargs["CONST"])}

    _patch_key_deriver(monkeypatch, fake_keys)

    keys = dispatch.get_warmup_keys()
    assert dispatched == list(range(1, 8193))
    assert len(keys) == 2
    assert {specialization(dict(key.inputs)["second"]) for key in keys} == {7, 8}


def test_triton_kernel_decorator_propagates_dispatch_assertions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel = _FakeTritonKernel()

    def warmup_inputs() -> dict[str, Any]:
        return dict(first="warmup", second=WarmupChoices(1, 2))

    @triton_kernel_dispatcher_with_warmup(kernel=kernel, warmup_inputs=warmup_inputs)
    def dispatch(first: str, second: int) -> LaunchSpec:
        assert second != 2, "broken dispatch"
        return (1,), dict(CONST=second)

    _patch_key_deriver(
        monkeypatch,
        lambda kernel, kwargs: {TritonJitKey(id(kernel), "fake", 0, kwargs["CONST"])},
    )

    with pytest.raises(AssertionError, match="broken dispatch"):
        dispatch.get_warmup_keys()


def test_triton_kernel_dispatch_uses_cuda_fake_tensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel = _FakeTritonKernel()

    def warmup_inputs() -> dict[str, Any]:
        return dict(
            first=TritonWarmupTensor(torch.float32, shape=(2, 3)),
            second=1,
        )

    @triton_kernel_dispatcher_with_warmup(kernel=kernel, warmup_inputs=warmup_inputs)
    def dispatch(first: torch.Tensor, second: int) -> LaunchSpec:
        assert isinstance(first, torch.Tensor)
        assert first.is_cuda
        assert first[0].is_contiguous()
        return (first.shape[0],), dict(CONST=first[0].numel())

    _patch_key_deriver(
        monkeypatch,
        lambda kernel, kwargs: {TritonJitKey(id(kernel), "fake", 0, kwargs["CONST"])},
    )

    keys = dispatch.get_warmup_keys()
    assert len(keys) == 1
    assert dict(keys[0].inputs)["first"].device.type == "cuda"


def test_triton_kernel_decorates_native_launchers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel = _FakeTritonKernel()

    def warmup_inputs(kernel: Any = kernel) -> dict[str, Any]:
        return triton_warmup_inputs(
            kernel,
            "warmup",
            WarmupChoices(1, 2),
            grid=(2,),
            CONST=7,
        )

    launcher = triton_kernel_dispatcher_with_warmup(warmup_inputs=warmup_inputs)(kernel)
    launcher[(3,)]("runtime", 4, CONST=8)

    assert kernel.runtime_calls == [((3,), ("runtime", 4), {"CONST": 8})]

    _patch_key_deriver(
        monkeypatch,
        lambda kernel, kwargs: {
            TritonJitKey(id(kernel), "fake", 0, (kwargs["second"], kwargs["CONST"]))
        },
    )

    keys = launcher.get_warmup_keys()
    assert len(keys) == 2
    launcher.compile(keys[0])
    assert kernel.warmup_calls == [
        {"grid": (1,), "first": "warmup", "second": 1, "CONST": 7}
    ]


def test_triton_warmup_inputs_expands_explicit_pointer_dtypes() -> None:
    inputs = triton_warmup_inputs(
        _pointer_group_test_kernel,
        grid=(1,),
        pointer_dtypes={torch.float32: ("x_ptr",)},
        value=2,
        BLOCK=16,
    )

    assert inputs["x_ptr"] == TritonWarmupTensor(torch.float32)
    assert inputs["value"] == 2

    with pytest.raises(ValueError, match="not a pointer: value"):
        triton_warmup_inputs(
            _pointer_group_test_kernel,
            grid=(1,),
            pointer_dtypes={torch.float32: ("value",)},
            x_ptr=TritonWarmupTensor(torch.float32),
            BLOCK=16,
        )
    with pytest.raises(ValueError, match="Missing Triton pointer inputs: x_ptr"):
        triton_warmup_inputs(
            _pointer_group_test_kernel,
            grid=(1,),
            pointer_dtypes={},
            value=2,
            BLOCK=16,
        )


def test_triton_key_derivation_applies_wrappers_and_runtime_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    knobs = triton.knobs
    driver = triton.runtime.driver

    device = "test-device"
    monkeypatch.setattr(driver.active, "get_current_device", lambda: device)
    calls: list[dict[str, Any]] = []

    def binder(x: Any, value: int, BLOCK: int, **options: Any) -> Any:
        calls.append(dict(BLOCK=BLOCK, **options))
        specialization = [("pointer", x.dtype), ("i32", value), ("constexpr", BLOCK)]
        return {}, specialization, options

    cache: dict[Any, Any] = {}
    _binder_test_kernel.device_caches[device] = ({}, cache, None, None, binder)
    heuristic_kernel = triton.heuristics(
        {"BLOCK": lambda args: 16 if args["value"] <= 16 else 32}
    )(_binder_test_kernel)

    keys = jit_warmup_triton_helper._triton_key_deriver(heuristic_kernel)(
        {"x": TritonWarmupTensor(tl.float32), "value": 2}
    )

    assert len(keys) == 1
    expected_options = {
        "debug": knobs.runtime.debug,
        "instrumentation_mode": knobs.compilation.instrumentation_mode,
    }
    if hasattr(knobs.compilation, "fpsan_homomorphic_casts"):
        expected_options["fpsan_homomorphic_casts"] = (
            knobs.compilation.fpsan_homomorphic_casts
        )
    assert calls == [{"BLOCK": 16, **expected_options}]


def test_triton_key_derivation_covers_autotune_configs_and_jit_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver = triton.runtime.driver

    device = "test-device"
    monkeypatch.setattr(driver.active, "get_current_device", lambda: device)

    def binder(x: Any, value: int, BLOCK: int, **options: Any) -> Any:
        specialization = [("pointer", x.dtype), ("i32", value), ("constexpr", BLOCK)]
        return {}, specialization, options

    for kernel in (_binder_test_kernel, _second_binder_test_kernel):
        kernel.device_caches[device] = ({}, {}, None, None, binder)
    autotuned = triton.autotune(
        configs=[
            triton.Config({"BLOCK": 16}, num_warps=2),
            triton.Config({"BLOCK": 32}, num_warps=4),
        ],
        key=["value"],
    )(_binder_test_kernel)
    inputs = {"x": TritonWarmupTensor(tl.float32), "value": 2}

    autotune_keys = jit_warmup_triton_helper._triton_key_deriver(autotuned)(inputs)
    other_kernel_keys = jit_warmup_triton_helper._triton_key_deriver(
        _second_binder_test_kernel
    )(inputs | {"BLOCK": 16})

    assert len(autotune_keys) == 2
    assert {key.jit_function_key for key in autotune_keys} == {
        _binder_test_kernel.cache_key
    }
    assert {key.jit_function_id for key in autotune_keys} == {id(_binder_test_kernel)}
    assert autotune_keys.isdisjoint(other_kernel_keys)


def test_compute_slot_mapping_uses_named_launcher_inputs(monkeypatch) -> None:
    from vllm.v1.attention.backends.utils import PAD_SLOT_ID
    from vllm.v1.worker.block_table import ComputeSlotMappingKernel

    owner = ComputeSlotMappingKernel()
    compile_key = owner.CompileKey(
        kv_cache_block_size=16,
        blocks_per_kv_block=1,
        total_cp_world_size=2,
        total_cp_rank=1,
        cp_kv_cache_interleave_size=1,
        block_table_stride=128,
        block_size=16,
    )
    launches: list[tuple[Any, ...]] = []

    def launch(launch_spec: Any, inputs: Any) -> None:
        launches.append((launch_spec, inputs))

    monkeypatch.setattr(owner, "launch", launch)
    owner.compile(compile_key)

    (grid, kwargs), inputs = launches[0]
    assert grid == (2,)
    assert inputs["num_tokens"] == 2
    assert inputs["block_table_stride"] == 128
    assert kwargs == {
        "KV_CACHE_BLOCK_SIZE": 16,
        "BLOCKS_PER_KV_BLOCK": 1,
        "TOTAL_CP_WORLD_SIZE": 2,
        "TOTAL_CP_RANK": 1,
        "CP_KV_CACHE_INTERLEAVE_SIZE": 1,
        "PAD_ID": PAD_SLOT_ID,
        "BLOCK_SIZE": owner.triton_block_size,
    }
