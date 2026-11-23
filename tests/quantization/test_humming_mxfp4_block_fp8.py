# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Humming MXFP4-weight GEMM paired with a block-FP8 activation.

The Humming kernel already implements MXFP4 weight (group-32 e8m0) with a
grouped FP8 activation via WGMMA software dequant -- see the grouped-fp8
`test_mxfp4.py` cases in the humming repo, which run on SM90 (H200). Enabling
it in vLLM comes down to two things, both exercised here:

  1. A grouped FP8 activation must classify to a float32-scaled key
     (kFp8Dynamic128Sym), not a uint8 microscale key. MX microscale
     activations (group size 32) keep their e8m0 (uint8) scale.
  2. (kMxfp4Static, kFp8Dynamic128Sym) must be an allowed MoE
     (weight, activation) pair.
"""

import asyncio
import contextlib
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import pytest
import torch

from vllm.model_executor.layers.fused_moe.experts.fused_humming_moe import (
    HummingExpertsBase,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic128Sym,
    kFp8DynamicTokenSym,
    kMxfp4Static,
    kMxfp8Dynamic,
    kNvfp4Dynamic,
)
from vllm.utils.import_utils import has_humming

from ..utils import multi_gpu_test

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

# make_humming_moe_quant_config stores per-sublayer humming LayerConfigs but the
# config-plumbing asserts below never read them, so a placeholder satisfies the
# required arg without pulling in the humming package.
_PLACEHOLDER_HUMMING_CONFIGS = {"w13": None, "w2": None}


def test_mxfp4_weight_with_block_fp8_activation_is_supported():
    """MXFP4 weight + block-FP8 (group-128) activation is an allowed pair."""
    assert HummingExpertsBase._supports_quant_scheme(kMxfp4Static, kFp8Dynamic128Sym)


def test_mxfp4_weight_with_per_token_fp8_activation_still_supported():
    """The pre-existing per-token FP8 pairing must keep working."""
    assert HummingExpertsBase._supports_quant_scheme(kMxfp4Static, kFp8DynamicTokenSym)


@pytest.mark.skipif(not has_humming(), reason="humming is not installed")
@pytest.mark.parametrize(
    "group_size,expected_key",
    [
        # per-token FP8 -> float32 per-token scale
        (0, kFp8DynamicTokenSym),
        # block FP8 -> float32 group-128 scale (the newly-fixed case)
        (128, kFp8Dynamic128Sym),
        # MXFP8 microscale -> e8m0 (uint8) group-32 scale (must be preserved)
        (32, kMxfp8Dynamic),
    ],
)
def test_humming_fp8_input_schema_to_quant_key(group_size, expected_key):
    from vllm.model_executor.layers.quantization.utils.humming import (
        input_schema_to_quant_key,
    )
    from vllm.utils.humming import HummingInputSchema
    from vllm.utils.humming import dtypes as humming_dtypes

    schema = HummingInputSchema(
        a_dtype=humming_dtypes.float8e4m3,
        input_scale_group_size=group_size,
    )
    assert input_schema_to_quant_key(schema, param_dtype=torch.bfloat16) == expected_key


@pytest.mark.skipif(not has_humming(), reason="humming is not installed")
def test_humming_bf16_input_schema_is_unquantized():
    from vllm.model_executor.layers.quantization.utils.humming import (
        input_schema_to_quant_key,
    )
    from vllm.utils.humming import HummingInputSchema

    # No a_dtype -> unquantized (bf16/fp16) inputs -> None.
    assert (
        input_schema_to_quant_key(HummingInputSchema(), param_dtype=torch.bfloat16)
        is None
    )


def test_block_fp8_activation_quant_config_is_block_quantized():
    """A block-FP8 activation group shape yields a block-quantized MoE config
    (FP8 dtype, [1, 128] block, not per-act-token). This is what makes the
    DeepEP prepare/finalize step quantize activations to block FP8 *before* the
    all-to-all dispatch instead of deferring to Humming -- and what makes
    HummingExpertsBase.expects_unquantized_inputs return False."""
    from vllm.model_executor.layers.quantization.utils.humming import (
        make_humming_moe_quant_config,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
    from vllm.platforms import current_platform

    qc = make_humming_moe_quant_config(
        quant_dtype=current_platform.fp8_dtype(),
        weight_dtype="float4e2m1",
        activation_group_shape=GroupShape(row=1, col=128),
        humming_configs=_PLACEHOLDER_HUMMING_CONFIGS,
    )
    assert qc.is_block_quantized
    assert qc.block_shape == [1, 128]
    assert not qc.per_act_token_quant
    assert qc.quant_dtype == current_platform.fp8_dtype()


def test_default_activation_quant_config_defers_to_humming():
    """Without a block activation shape the config stays per-token (the deferred
    path): Humming quantizes internally, preserving pre-existing behavior for
    every non-block-FP8 scheme."""
    from vllm.model_executor.layers.quantization.utils.humming import (
        make_humming_moe_quant_config,
    )
    from vllm.platforms import current_platform

    qc = make_humming_moe_quant_config(
        quant_dtype=current_platform.fp8_dtype(),
        weight_dtype="float4e2m1",
        humming_configs=_PLACEHOLDER_HUMMING_CONFIGS,
    )
    assert not qc.is_block_quantized
    assert qc.per_act_token_quant


@pytest.mark.skipif(not has_humming(), reason="humming is not installed")
@pytest.mark.parametrize(
    "capability,user_override,expected_group_size",
    [
        (90, None, 128),
        (90, kMxfp8Dynamic, 32),
        (100, None, 32),
    ],
)
def test_checkpoint_mxfp8_runs_as_block_fp8_on_hopper(
    monkeypatch, capability, user_override, expected_group_size
):
    """A checkpoint's MXFP8 MoE activations (e.g. DeepSeek V4) run as block
    FP8 on Humming's SM90 kernels; an explicit user request is honored."""
    from vllm.model_executor.layers.fused_moe.oracle import mxfp4
    from vllm.platforms import current_platform

    # Patch the class, not the instance: before pytest 9.2, undoing an instance
    # patch leaves the class attribute shadowed in the instance's __dict__.
    monkeypatch.setattr(
        type(current_platform),
        "is_device_capability_family",
        classmethod(lambda cls, cap: cap == capability),
    )
    monkeypatch.setattr(mxfp4, "_user_moe_activation_override", lambda: user_override)
    monkeypatch.delenv("VLLM_HUMMING_INPUT_QUANT_CONFIG", raising=False)

    schema = mxfp4._humming_input_schema(kMxfp8Dynamic)

    assert str(schema.a_dtype) == "float8e4m3"
    assert schema.input_scale_group_size == expected_group_size


@pytest.mark.skipif(not has_humming(), reason="humming is not installed")
@pytest.mark.parametrize(
    "capability,strict,honors",
    [
        # Hopper has no FP4 activations: falls back, and strict cannot run it.
        ((9, 0), False, False),
        ((9, 0), True, False),
        # Slow on SM10x, so it falls back unless strict mode forbids that.
        ((10, 3), False, False),
        ((10, 3), True, True),
        ((12, 1), False, True),
    ],
)
def test_humming_nvfp4_linear_declares_executed_activation(
    monkeypatch, capability, strict, honors
):
    """Humming picks its NVFP4 input schema at weight load, per device. Kernel
    selection needs the same answer up front so VLLM_STRICT_QUANT_SCHEME can
    move on to another kernel instead of failing at load."""
    import humming.schema.humming as humming_schema

    from vllm.model_executor.kernels.linear import HummingNvFp4LinearKernel
    from vllm.platforms import current_platform
    from vllm.platforms.interface import DeviceCapability

    monkeypatch.setenv("VLLM_STRICT_QUANT_SCHEME", str(int(strict)))
    # Patch the class, not the instance: before pytest 9.2, undoing an instance
    # patch leaves a bound `@cache` classmethod in current_platform.__dict__,
    # which cloudpickle cannot pickle when a later test calls LLM.apply_model.
    monkeypatch.setattr(
        type(current_platform),
        "get_device_capability",
        classmethod(lambda cls, device_id=0: DeviceCapability(*capability)),
    )
    monkeypatch.setattr(
        type(humming_schema.current_device),
        "sm_version",
        property(lambda self: capability[0] * 10 + capability[1]),
    )

    executed = HummingNvFp4LinearKernel.executed_activation_key(kNvfp4Dynamic)
    assert executed == (kNvfp4Dynamic if honors else None)


def _mxfp6_weight_schema():
    from vllm.utils.humming import BaseWeightSchema

    return BaseWeightSchema.from_config(
        {
            "quant_method": "humming",
            "dtype": "float6e2m3",
            "scale_dtype": "float8e8m0",
            "group_size": 32,
            "weight_scale_type": "group",
        }
    )


@contextlib.contextmanager
def _mock_sm_version(capability):
    from unittest.mock import patch

    from vllm.platforms.interface import DeviceCapability

    class _Device:
        sm_version = capability

    with (
        patch(
            "vllm.model_executor.layers.quantization.utils.humming.schema"
            ".current_platform"
        ) as platform_mock,
        patch("humming.schema.humming.current_device", _Device()),
    ):
        platform_mock.get_device_capability.return_value = DeviceCapability(
            capability // 10, capability % 10
        )
        yield


def _resolve_input_schema(capability, activation_key, **kwargs):
    from vllm.model_executor.layers.quantization.utils.humming import (
        check_and_fallback_input_schema,
        quant_key_to_input_schema,
    )

    with _mock_sm_version(capability):
        return check_and_fallback_input_schema(
            _mxfp6_weight_schema(),
            quant_key_to_input_schema(activation_key),
            param_dtype=torch.bfloat16,
            **kwargs,
        )


@pytest.mark.skipif(not has_humming(), reason="humming is not installed")
@pytest.mark.parametrize(
    "capability,activation_key,kwargs,expected",
    [
        # SM90 WGMMA has no block-scaled MX path: keep the 16-bit fallback.
        (90, kMxfp8Dynamic, {}, "16bit"),
        (100, kMxfp8Dynamic, {}, "fp8"),
        (103, kMxfp8Dynamic, {}, "fp8"),
        (120, kMxfp8Dynamic, {}, "fp8"),
        # MoE grouped-GEMM FP8 is unvalidated on SM100-SM119: keep the gate.
        (100, kMxfp8Dynamic, {"allow_fp8_sm100": False}, "16bit"),
        # FP4 requests keep their gate on SM100.
        (100, kNvfp4Dynamic, {}, "not_fp4"),
    ],
)
def test_input_schema_honors_sm_activation_gates(
    capability, activation_key, kwargs, expected
):
    """Honor MXFP8 requests while preserving architecture and MoE gates."""
    from vllm.utils.humming import dtypes as humming_dtypes

    resolved = _resolve_input_schema(capability, activation_key, **kwargs)
    if expected == "fp8":
        assert resolved.a_dtype == humming_dtypes.float8e4m3
        assert resolved.input_scale_group_size == 32
    elif expected == "16bit":
        assert resolved.a_dtype is None or resolved.a_dtype.num_bits == 16
    else:
        assert resolved.a_dtype is None or resolved.a_dtype.num_bits != 4


class HummingDispatchTestExtension:
    """Read bindings and qualify real forward calls on every DP worker."""

    model_runner: "GPUModelRunner"
    vllm_config: "VllmConfig"

    def init_worker_extension(self):
        """Observe initial graph capture before compilation and model warmup."""
        import sys
        import threading
        from collections import Counter

        from torch._subclasses.fake_tensor import FakeTensor

        from vllm.model_executor.layers.fused_moe.prepare_finalize.naive_dp_ep import (
            MoEPrepareAndFinalizeNaiveDPEPModular,
        )
        from vllm.utils.humming import may_process_input

        counts: Counter[str] = Counter()
        errors: list[str] = []
        active: dict[int, tuple[Any, ...]] = {}
        self.dispatch_capture_trace = counts, errors, active
        self.dispatch_capture_thread = threading.get_ident()
        self.dispatch_capture_observer = None
        model_config = self.vllm_config.model_config
        assert model_config is not None
        if model_config.enforce_eager:
            return
        assert sys.getprofile() is None and threading.getprofile() is None
        enabled = self.vllm_config.kernel_config.humming_fp8_dispatch
        prepare_code = MoEPrepareAndFinalizeNaiveDPEPModular.prepare.__code__
        process_code = HummingExpertsBase.process_input.__code__
        producer_code = may_process_input.__code__
        observed = {prepare_code, process_code, producer_code}

        def observe(frame, event, result):
            if frame.f_code not in observed:
                return
            values = frame.f_locals
            inputs = values.get("a1" if frame.f_code is prepare_code else "inputs")
            if (
                not isinstance(inputs, torch.Tensor)
                or isinstance(inputs, FakeTensor)
                or inputs.shape[0] == 0
                or not torch.cuda.is_current_stream_capturing()
            ):
                return
            try:
                if frame.f_code is prepare_code and event == "return":
                    q, scale = result[:2]
                    if enabled:
                        assert q.dtype == torch.float8_e4m3fn
                        assert scale.dtype == torch.float32
                        assert scale.shape == (q.shape[0], 1)
                    else:
                        assert q.dtype == torch.bfloat16 and scale is None
                    counts["prepare"] += 1
                elif frame.f_code is process_code:
                    key = id(frame)
                    if event == "call" and values["sublayer_name"] == "w13":
                        active[key] = inputs, values["input_scale"], 0
                    elif event == "return" and key in active:
                        inputs, scale, calls = active.pop(key)
                        if enabled:
                            assert inputs.dtype == torch.float8_e4m3fn
                            assert scale.dtype == torch.float32
                            assert result[0] is inputs and result[1] is scale
                            assert calls == 0
                        else:
                            assert inputs.dtype == torch.bfloat16 and scale is None
                            assert calls > 0
                        counts["w13"] += 1
                        counts["receiver_producer"] += calls
                elif frame.f_code is producer_code and event == "call":
                    parent = frame.f_back
                    if parent is not None and id(parent) in active:
                        inputs, scale, calls = active[id(parent)]
                        active[id(parent)] = inputs, scale, calls + 1
                    elif (
                        parent is not None
                        and parent.f_code.co_name == "quantize_input"
                        and parent.f_globals.get("__name__")
                        == "vllm.model_executor.layers.quantization.utils.humming.moe"
                    ):
                        counts["source_producer"] += 1
            except Exception as error:
                if len(errors) < 16:
                    errors.append(repr(error))

        self.dispatch_capture_observer = observe
        sys.setprofile(observe)

    def destroy_worker_extension(self):
        import sys

        if sys.getprofile() is self.dispatch_capture_observer:
            sys.setprofile(None)

    def humming_dispatch_capture_report(self):
        import sys
        import threading

        assert threading.get_ident() == self.dispatch_capture_thread
        try:
            counts, errors, active = self.dispatch_capture_trace
        finally:
            if sys.getprofile() is self.dispatch_capture_observer:
                sys.setprofile(None)
        return {
            "counts": dict(counts),
            "errors": errors,
            "unfinished": len(active),
            "profiler_off": sys.getprofile() is None and threading.getprofile() is None,
        }

    def humming_dispatch_contract(self):
        from vllm.distributed import (
            get_dp_group,
            get_ep_group,
            get_pcp_group,
            get_tp_group,
        )

        layers = []
        for name, module in self.model_runner.get_model().named_modules():
            kernel = getattr(getattr(module, "quant_method", None), "moe_kernel", None)
            if kernel is None:
                continue
            experts, prepare = kernel.fused_experts, kernel.prepare_finalize
            callback = getattr(prepare, "_input_quantizer", None)
            layers.append(
                {
                    "name": name,
                    "quant_method": type(module.quant_method).__name__,
                    "experts": type(experts).__name__,
                    "prepare": type(prepare).__name__,
                    "callback": None
                    if callback is None
                    else {
                        "module": callback.__module__,
                        "qualname": callback.__qualname__,
                    },
                }
            )
        group = get_dp_group()
        local = {
            "rank": get_ep_group().rank_in_group,
            "dp_rank": group.rank_in_group,
            "tp_rank": get_tp_group().rank_in_group,
            "pcp_rank": get_pcp_group().rank_in_group,
            "capture": self.humming_dispatch_capture_report(),
            "feature": self.vllm_config.kernel_config.humming_fp8_dispatch,
            "layers": layers,
        }
        return local

    def humming_dispatch_watch_begin(self):
        import inspect
        import sys
        import threading
        from collections import Counter

        from torch._subclasses.fake_tensor import FakeTensor

        from vllm.config.compilation import CUDAGraphMode
        from vllm.forward_context import get_forward_context
        from vllm.utils.humming import may_process_input

        assert sys.getprofile() is None and threading.getprofile() is None
        prepares, experts, callbacks = set(), set(), set()
        for module in self.model_runner.get_model().modules():
            kernel = getattr(getattr(module, "quant_method", None), "moe_kernel", None)
            if kernel is None:
                continue
            prepares.add(kernel.prepare_finalize.prepare.__func__.__code__)
            experts.add(id(kernel.fused_experts))
            callback = getattr(kernel.prepare_finalize, "_input_quantizer", None)
            if callback is not None:
                callbacks.add(callback.__code__)
        runner = self.model_runner
        execute_code = inspect.unwrap(runner.execute_model).__code__
        batch_code = inspect.unwrap(runner.gather_batch_req_state).__code__
        graph_manager = runner.cudagraph_manager
        graph_codes = {}
        if graph_manager is not None:
            graph_codes = {
                inspect.unwrap(
                    graph_manager.run_fullgraph
                ).__code__: CUDAGraphMode.FULL,
                inspect.unwrap(
                    graph_manager.run_pw_graph
                ).__code__: CUDAGraphMode.PIECEWISE,
            }
        counts: Counter[str] = Counter()
        errors: list[str] = []
        active: dict[int, tuple[Any, ...]] = {}
        graph_active: dict[int, CUDAGraphMode] = {}
        self.dispatch_trace = counts, errors, active, graph_active
        enabled = self.vllm_config.kernel_config.humming_fp8_dispatch
        process_code = HummingExpertsBase.process_input.__code__
        producer_code = may_process_input.__code__
        observed_codes = prepares | {
            execute_code,
            batch_code,
            process_code,
            producer_code,
            *graph_codes,
        }
        pure_decode = False
        has_prefill = False

        def dispatch_scope(inputs):
            if not isinstance(inputs, torch.Tensor) or isinstance(inputs, FakeTensor):
                return None
            if inputs.shape[0] == 0 or torch.cuda.is_current_stream_capturing():
                return None
            context = get_forward_context()
            metadata = context.dp_metadata
            assert metadata is not None
            global_rows = sum(metadata.get_chunk_sizes_across_dp_rank())
            if global_rows == 0:
                return None
            if has_prefill:
                counts["prefill_min_global_rows"] = min(
                    counts.get("prefill_min_global_rows", global_rows), global_rows
                )
                assert global_rows >= 256, (
                    "Qualification prefill must exceed small steps"
                )
            bucket = "small_decode" if pure_decode and global_rows < 256 else "other"
            prequantized = enabled
            return bucket, prequantized

        def observe(frame, event, result):
            nonlocal pure_decode, has_prefill
            code = frame.f_code
            if code not in observed_codes:
                return
            values = frame.f_locals
            try:
                if code is execute_code and values.get("self") is runner:
                    # Clear on every step, including dummy and zero-work steps.
                    pure_decode = False
                    has_prefill = False
                elif code is batch_code and values.get("self") is runner:
                    if event == "return":
                        batch, _ = result
                        has_prefill = batch is not None and batch.has_prefill
                        counts["prefill_steps"] += int(has_prefill)
                        # The real CPU batch state distinguishes even a one-token
                        # prefill extension from a pure decode.
                        pure_decode = (
                            batch is not None
                            and bool(batch.req_ids)
                            and not batch.has_prefill
                        )
                elif code in graph_codes and values.get("self") is graph_manager:
                    key = id(frame)
                    if event == "call" and pure_decode:
                        mode = graph_codes[code]
                        actual_mode = (
                            values["desc"].cg_mode
                            if mode == CUDAGraphMode.FULL
                            else get_forward_context().cudagraph_runtime_mode
                        )
                        assert actual_mode == mode
                        graph_active[key] = mode
                    elif event == "return" and key in graph_active:
                        mode = graph_active.pop(key)
                        counts[f"graph_decode_{mode.name.lower()}"] += 1
                elif code in prepares and event == "return":
                    if not isinstance(result, tuple):
                        return
                    scope = dispatch_scope(result[0])
                    if scope is None:
                        return
                    bucket, prequantized = scope
                    q, scale = result[:2]
                    if prequantized:
                        assert q.dtype == torch.float8_e4m3fn
                        assert scale.dtype == torch.float32
                        assert scale.shape == (q.shape[0], 1)
                    else:
                        assert q.dtype == torch.bfloat16 and scale is None
                    counts[f"{bucket}_prepare"] += 1
                elif code is process_code and id(values["self"]) in experts:
                    key = id(frame)
                    if event == "call" and values["sublayer_name"] == "w13":
                        scope = dispatch_scope(values["inputs"])
                        if scope is not None:
                            active[key] = (
                                *scope,
                                values["inputs"],
                                values["input_scale"],
                                0,
                            )
                    elif event == "return" and key in active:
                        bucket, prequantized, inputs, scale, calls = active.pop(key)
                        if prequantized:
                            assert inputs.dtype == torch.float8_e4m3fn
                            assert scale.dtype == torch.float32
                            assert result[0] is inputs and result[1] is scale
                            assert calls == 0
                        else:
                            assert inputs.dtype == torch.bfloat16 and scale is None
                            assert calls > 0
                        counts[f"{bucket}_w13"] += 1
                        counts[f"{bucket}_receiver_producer"] += calls
                elif code is producer_code and event == "call":
                    parent = frame.f_back
                    if parent is not None and id(parent) in active:
                        bucket, prequantized, inputs, scale, calls = active[id(parent)]
                        active[id(parent)] = (
                            bucket,
                            prequantized,
                            inputs,
                            scale,
                            calls + 1,
                        )
                    if parent is not None and parent.f_code in callbacks:
                        scope = dispatch_scope(values["inputs"])
                        if scope is not None:
                            bucket, prequantized = scope
                            assert prequantized
                            counts[f"{bucket}_source_producer"] += 1
                            counts[f"{bucket}_source_rows"] += values["inputs"].shape[0]
            except Exception as error:
                if len(errors) < 16:
                    errors.append(repr(error))

        # RPCs and model forwards share the worker loop thread (also on Python 3.11).
        self.dispatch_trace_thread = threading.get_ident()
        sys.setprofile(observe)

    def humming_dispatch_watch_end(self):
        import sys
        import threading

        from vllm.distributed import get_ep_group

        assert threading.get_ident() == self.dispatch_trace_thread
        sys.setprofile(None)
        counts, errors, active, graph_active = self.dispatch_trace
        group = get_ep_group()
        local = {
            "rank": group.rank_in_group,
            "counts": dict(counts),
            "errors": errors,
            "unfinished": len(active) + len(graph_active),
            "profiler_off": sys.getprofile() is None and threading.getprofile() is None,
        }
        active.clear()
        graph_active.clear()
        return local


@multi_gpu_test(num_gpus=2)
@pytest.mark.skipif(not has_humming(), reason="humming is not installed")
@pytest.mark.parametrize("enforce_eager", [True, False], ids=["eager", "graphs"])
def test_humming_fp8_dispatch_public_dp_generation(
    monkeypatch,
    enforce_eager: bool,
    tensor_parallel_size: int = 1,
    prefill_context_parallel_size: int = 1,
):
    """The opt-in preserves BI greedy output through real model construction.

    This tests deployment/configuration compatibility, not model accuracy.
    """
    from vllm import SamplingParams, TokensPrompt
    from vllm.distributed import cleanup_dist_env_and_memory
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.platforms import current_platform
    from vllm.transformers_utils.utils import maybe_model_redirect
    from vllm.v1.engine.async_llm import AsyncLLM

    if not current_platform.is_cuda():
        pytest.skip("Requires CUDA")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setenv("VLLM_HUMMING_MOE_GEMM_TYPE", "indexed")
    monkeypatch.setenv("VLLM_HUMMING_INPUT_QUANT_CONFIG", '{"dtype":"float8e4m3"}')

    async def run(enabled: bool) -> Sequence[tuple[int, ...]]:
        engine = AsyncLLM.from_engine_args(
            AsyncEngineArgs(
                model=maybe_model_redirect("openai/gpt-oss-20b"),
                dtype="bfloat16",
                enforce_eager=enforce_eager,
                tensor_parallel_size=tensor_parallel_size,
                prefill_context_parallel_size=prefill_context_parallel_size,
                data_parallel_size=2,
                enable_expert_parallel=True,
                all2all_backend="allgather_reducescatter",
                max_model_len=1024,
                max_num_batched_tokens=512,
                max_num_seqs=8,
                kv_cache_memory_bytes=1 << 30,
                enable_prefix_caching=False,
                seed=0,
                compilation_config={
                    "mode": 3,
                    "cudagraph_mode": "FULL_AND_PIECEWISE",
                    "cudagraph_capture_sizes": [1, 2, 4, 8],
                },
                worker_extension_cls=(
                    "tests.quantization.test_humming_mxfp4_block_fp8."
                    "HummingDispatchTestExtension"
                ),
                kernel_config={
                    "moe_backend": "humming",
                    "humming_fp8_dispatch": enabled,
                },
            )
        )

        async def rpc(method: str) -> list[Any]:
            """Collect every DP engine's local result without a worker collective."""
            per_engine = await asyncio.wait_for(
                engine.engine_core.call_utility_all_async(
                    "collective_rpc", method, None, (), None
                ),
                120,
            )
            return [result for results in per_engine for result in results]

        async def generate(rank: int, phase: str) -> tuple[int, ...]:
            prompt_ids = [1000 + rank] * 512
            final = None
            async for output in engine.generate(
                TokensPrompt(prompt_token_ids=prompt_ids),
                SamplingParams(temperature=0, max_tokens=8, ignore_eos=True),
                request_id=f"humming-{enabled}-{rank}-{phase}",
                data_parallel_rank=rank,
            ):
                final = output
            assert final is not None
            assert final.prompt_token_ids == prompt_ids
            assert len(final.outputs[0].token_ids) == 8
            return tuple(final.outputs[0].token_ids)

        async def check_binding():
            rows = await rpc("humming_dispatch_contract")
            world_size = 2 * tensor_parallel_size * prefill_context_parallel_size
            assert len(rows) == world_size
            assert {row["rank"] for row in rows} == set(range(world_size))
            expected = {
                "module": "vllm.model_executor.layers.quantization.utils.humming.moe",
                "qualname": (
                    "_make_humming_fp8_dispatch_quantizer.<locals>.quantize_input"
                ),
            }
            for row in rows:
                capture = row["capture"]
                assert not capture["errors"] and capture["unfinished"] == 0, row
                assert capture["profiler_off"], row
                if not enforce_eager:
                    # These are capture-time events, independently of replay counts.
                    counts = capture["counts"]
                    assert counts.get("prepare", 0) > 0, row
                    assert counts.get("w13", 0) > 0, row
                    if enabled:
                        assert counts.get("source_producer", 0) > 0, row
                        assert counts.get("receiver_producer", 0) == 0, row
                    else:
                        assert counts.get("source_producer", 0) == 0, row
                        assert counts.get("receiver_producer", 0) > 0, row
                assert row["feature"] is enabled
                assert len(row["layers"]) == 24
                assert len({layer["name"] for layer in row["layers"]}) == 24
                for layer in row["layers"]:
                    assert layer["quant_method"] in {
                        "GptOssMxfp4MoEMethod",
                        "HummingMoEMethod",
                    }
                    assert layer["experts"] == "HummingIndexedExperts"
                    assert layer["prepare"] == "MoEPrepareAndFinalizeNaiveDPEPModular"
                    assert layer["callback"] == (expected if enabled else None)

        try:
            await check_binding()
            await asyncio.gather(generate(0, "warmup"), generate(1, "warmup"))
            await rpc("humming_dispatch_watch_begin")
            try:
                outputs = await asyncio.gather(
                    generate(0, "qualification"), generate(1, "qualification")
                )
            finally:
                rows = await rpc("humming_dispatch_watch_end")
            world_size = 2 * tensor_parallel_size * prefill_context_parallel_size
            assert len(rows) == world_size
            assert {row["rank"] for row in rows} == set(range(world_size))
            for row in rows:
                assert not row["errors"] and row["unfinished"] == 0, row
                assert row["profiler_off"], row
                counts = row["counts"]
                assert counts.get("prefill_steps", 0) > 0, row
                assert counts.get("prefill_min_global_rows", 0) >= 256, row
                assert counts.get("other_prepare", 0) > 0, row
                assert counts.get("other_w13", 0) > 0, row
                if enabled:
                    assert counts.get("other_source_producer", 0) > 0, row
                    assert counts.get("other_receiver_producer", 0) == 0, row
                else:
                    assert counts.get("other_source_producer", 0) == 0, row
                    assert counts.get("other_receiver_producer", 0) > 0, row
                if enforce_eager:
                    # Large prefill counters cannot satisfy these decode assertions.
                    assert counts.get("small_decode_prepare", 0) > 0, row
                    assert counts.get("small_decode_w13", 0) > 0, row
                    if enabled:
                        assert counts.get("small_decode_source_producer", 0) > 0, row
                        assert counts.get("small_decode_receiver_producer", 0) == 0, row
                    else:
                        assert counts.get("small_decode_source_producer", 0) == 0, row
                        assert counts.get("small_decode_receiver_producer", 0) > 0, row
                else:
                    assert counts.get("graph_decode_full", 0) > 0, row
            await check_binding()
            return outputs
        finally:
            engine.shutdown()

    try:
        baseline = asyncio.run(run(False))
        candidate = asyncio.run(run(True))
        assert candidate == baseline
    finally:
        cleanup_dist_env_and_memory()
