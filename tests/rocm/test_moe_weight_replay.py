# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay MoE weights after sleep/wake and verify deterministic generation.

This test mirrors the load-then-post-load sequence used by RL frameworks when
they replay model weights into a sleeping vLLM engine:

1. Generate with the initial model.
2. Put the engine into level-2 sleep and wake it back up.
3. Replay the original weights into each worker.
4. Run the same greedy prompts and compare token IDs.

The replay helper is intentionally local to this test instead of importing a
specific RL framework. Frameworks such as VERL follow the same relevant order:
load one or more weight buckets, then run process_weights_after_loading once
after all buckets have arrived.
"""

import gc
import importlib.util
import os
from collections.abc import Iterable

import pytest
import torch

from vllm.platforms import current_platform

MODEL_PATH_ENV = "VLLM_AITER_MOE_REPLAY_MODEL_PATH"
DEFAULT_MODEL = "Qwen/Qwen3.5-35B-A3B"
TP_ENV = "VLLM_AITER_MOE_REPLAY_TP"

aiter_available = importlib.util.find_spec("aiter") is not None

pytestmark = pytest.mark.skipif(
    not (current_platform.is_rocm() and aiter_available),
    reason="MoE replay test requires ROCm with AITER installed",
)


def _visible_gpu_count() -> int:
    visible_devices = os.environ.get("HIP_VISIBLE_DEVICES")
    if visible_devices:
        return len([d for d in visible_devices.split(",") if d.strip()])
    return torch.accelerator.device_count()


def _model_path() -> str:
    return os.environ.get(MODEL_PATH_ENV, DEFAULT_MODEL)


def _sampling_params():
    from vllm import SamplingParams

    return SamplingParams(temperature=0.0, max_tokens=1)


def _cumem_allocator_available() -> bool:
    try:
        from vllm.device_allocator.cumem import cumem_available
    except ImportError:
        return False
    return cumem_available


def _set_backend_env(monkeypatch: pytest.MonkeyPatch, backend: str) -> None:
    monkeypatch.setenv("VLLM_USE_V1", "1")
    monkeypatch.setenv("VLLM_DISABLE_COMPILE_CACHE", "1")
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    monkeypatch.setenv("TORCHDYNAMO_DISABLE", "1")
    if backend == "aiter":
        monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
        monkeypatch.setenv("VLLM_ROCM_USE_AITER_MOE", "1")
    elif backend == "triton":
        monkeypatch.delenv("VLLM_ROCM_USE_AITER", raising=False)
        monkeypatch.setenv("VLLM_ROCM_USE_AITER_MOE", "0")
    else:
        raise AssertionError(f"unknown backend {backend}")

    from vllm._aiter_ops import rocm_aiter_ops

    rocm_aiter_ops.refresh_env_variables()


def _generated_token_ids(outputs) -> list[list[int]]:
    return [list(request_output.outputs[0].token_ids) for request_output in outputs]


def _generated_text(outputs) -> list[str]:
    return [request_output.outputs[0].text for request_output in outputs]


def _iter_buckets(
    weights: Iterable[tuple[str, torch.Tensor]],
    bucket_size_bytes: int,
) -> Iterable[list[tuple[str, torch.Tensor]]]:
    bucket: list[tuple[str, torch.Tensor]] = []
    bucket_bytes = 0
    for name, tensor in weights:
        tensor_bytes = tensor.numel() * tensor.element_size()
        if bucket and bucket_bytes + tensor_bytes > bucket_size_bytes:
            yield bucket
            bucket = []
            bucket_bytes = 0
        bucket.append((name, tensor))
        bucket_bytes += tensor_bytes
    if bucket:
        yield bucket


def _replay_cached_weights(worker, bucket_size_bytes: int = 256 << 20) -> dict:
    from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
        UnquantizedFusedMoEMethod,
    )
    from vllm.model_executor.model_loader import get_model_loader
    from vllm.model_executor.model_loader.utils import process_weights_after_loading

    model = worker.model_runner.model
    model_loader = get_model_loader(worker.model_runner.load_config)
    weights = model_loader.get_all_weights(worker.model_config, model)
    loaded_counts = []
    pad_events = []
    original_maybe_pad_weight = UnquantizedFusedMoEMethod._maybe_pad_weight

    def _recording_maybe_pad_weight(self, weight):
        before = {
            "data_ptr": weight.data_ptr(),
            "shape": tuple(weight.shape),
            "stride": tuple(weight.stride()),
            "element_size": weight.element_size(),
        }
        padded = original_maybe_pad_weight(self, weight)
        after = {
            "data_ptr": padded.data_ptr(),
            "shape": tuple(padded.shape),
            "stride": tuple(padded.stride()),
            "element_size": padded.element_size(),
        }
        if before["data_ptr"] != after["data_ptr"]:
            backend = str(
                getattr(self.unquantized_backend, "value", self.unquantized_backend)
            )
            pad_events.append(
                {
                    "backend": backend,
                    "before": before,
                    "after": after,
                }
            )
        return padded

    with torch.device(worker.device):
        for bucket in _iter_buckets(weights, bucket_size_bytes):
            device_bucket = [
                (name, tensor.to(worker.device, non_blocking=False))
                for name, tensor in bucket
            ]
            loaded = model.load_weights(device_bucket)
            loaded_counts.append(len(loaded) if loaded is not None else 0)
            del device_bucket
            torch.accelerator.empty_cache()
        before_post_load = _moe_weight_fingerprints(worker)
        UnquantizedFusedMoEMethod._maybe_pad_weight = _recording_maybe_pad_weight
        try:
            process_weights_after_loading(model, worker.model_config, worker.device)
        finally:
            UnquantizedFusedMoEMethod._maybe_pad_weight = original_maybe_pad_weight
        after_post_load = _moe_weight_fingerprints(worker)
    pointer_changes = [
        {
            "name": name,
            "before": before_post_load[name],
            "after": after_post_load[name],
        }
        for name in sorted(before_post_load)
        if before_post_load[name]["data_ptr"] != after_post_load[name]["data_ptr"]
    ]
    return {
        "num_buckets": len(loaded_counts),
        "loaded_counts": loaded_counts,
        "pad_events": pad_events,
        "pointer_changes": pointer_changes,
    }


def _moe_weight_fingerprints(worker) -> dict[str, dict]:
    fingerprints: dict[str, dict] = {}
    for module_name, module in worker.model_runner.model.named_modules():
        quant_method = getattr(module, "quant_method", None)
        backend = getattr(quant_method, "unquantized_backend", None)
        if backend is None:
            continue
        for weight_name in ("w13_weight", "w2_weight"):
            weight = getattr(module, weight_name, None)
            if weight is None:
                continue
            key = f"{module_name}.{weight_name}"
            fingerprints[key] = {
                "data_ptr": weight.data_ptr(),
                "shape": tuple(weight.shape),
                "stride": tuple(weight.stride()),
                "storage_offset": weight.storage_offset(),
                "backend": str(getattr(backend, "value", backend)),
            }
    return fingerprints


def _selected_unquantized_moe_backends(worker) -> list[str]:
    backends = set()
    for module in worker.model_runner.model.modules():
        quant_method = getattr(module, "quant_method", None)
        backend = getattr(quant_method, "unquantized_backend", None)
        if backend is not None:
            backends.add(str(getattr(backend, "value", backend)))
    return sorted(backends)


def _assert_backend_selected(llm, backend: str) -> None:
    backends = llm.llm_engine.collective_rpc(_selected_unquantized_moe_backends)
    flattened = {item for worker_backends in backends for item in worker_backends}
    if backend == "aiter":
        assert "ROCm AITER" in flattened
    else:
        assert "TRITON" in flattened


def _shutdown_llm(llm) -> None:
    if llm is None:
        return
    try:
        if hasattr(llm, "shutdown"):
            llm.shutdown()
        elif hasattr(llm.llm_engine, "shutdown"):
            llm.llm_engine.shutdown()
        elif hasattr(llm.llm_engine, "engine_core"):
            llm.llm_engine.engine_core.shutdown()
    finally:
        del llm
        gc.collect()
        torch.accelerator.empty_cache()


@pytest.mark.parametrize("backend", ["triton", "aiter"])
def test_moe_weight_replay_after_sleep_wake_is_deterministic(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
) -> None:
    tp_size = int(os.environ.get(TP_ENV, "2"))
    if _visible_gpu_count() < tp_size:
        pytest.skip(
            f"requires at least {tp_size} visible GPUs, found {_visible_gpu_count()}"
        )
    if not _cumem_allocator_available():
        pytest.skip("sleep mode replay requires the cumem allocator extension")
    if not (hasattr(torch.ops, "_moe_C") and hasattr(torch.ops._moe_C, "topk_softmax")):
        pytest.skip("MoE C extension with topk_softmax is required")

    _set_backend_env(monkeypatch, backend)

    from vllm import LLM

    llm = None
    try:
        llm = LLM(
            model=_model_path(),
            tensor_parallel_size=tp_size,
            dtype="bfloat16",
            trust_remote_code=True,
            enable_sleep_mode=True,
            enforce_eager=False,
            gpu_memory_utilization=float(
                os.environ.get("VLLM_AITER_MOE_REPLAY_GPU_MEMORY_UTILIZATION", "0.6")
            ),
            disable_custom_all_reduce=True,
            compilation_config={
                "mode": 0,
                "cudagraph_mode": "FULL",
            },
        )
        _assert_backend_selected(llm, backend)

        prompts = [
            "Solve 1+1. Answer with only the number.",
            "Name the color of the sky on a clear day in one word.",
        ]
        sampling_params = _sampling_params()

        first_outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        first_token_ids = _generated_token_ids(first_outputs)
        first_text = _generated_text(first_outputs)

        llm.sleep(level=2)
        llm.wake_up()
        replay_stats = llm.llm_engine.collective_rpc(_replay_cached_weights)
        assert all(stats["num_buckets"] > 0 for stats in replay_stats)
        assert all(stats["pad_events"] for stats in replay_stats), (
            f"{backend} MoE replay did not enter the ROCm padding branch: "
            f"{replay_stats!r}"
        )
        assert all(not stats["pointer_changes"] for stats in replay_stats), (
            f"{backend} MoE replay replaced parameter storage during "
            f"post-load processing: {replay_stats!r}"
        )

        second_outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        second_token_ids = _generated_token_ids(second_outputs)
        second_text = _generated_text(second_outputs)

        assert second_token_ids == first_token_ids, (
            f"{backend} MoE replay changed greedy token IDs after sleep/wake.\n"
            f"first_text={first_text!r}\n"
            f"second_text={second_text!r}\n"
            f"replay_stats={replay_stats!r}"
        )
    finally:
        _shutdown_llm(llm)
