# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import MethodType
from typing import TYPE_CHECKING, Any

import numpy as np

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

from mtp_ep_load_balance_utils import (
    DEFAULT_DATASET,
    DEFAULT_DATASET_CONFIG,
    DEFAULT_DATASET_SPLIT,
    SCHEMA_VERSION,
    StepTiming,
    aggregate_worker_step_timings,
    average_step_histograms,
    count_layer_expert_histograms,
    select_dataset_indices,
    select_step_routing_data,
)

if TYPE_CHECKING:
    from argparse import Namespace


@dataclass
class ConditionRawData:
    batch_size: int
    draft_length: int
    selected_dataset_indices: np.ndarray
    prompt_lengths: np.ndarray
    condition_latency_ms: float
    step_histograms: np.ndarray
    step_total_tokens: np.ndarray
    step_total_ms: np.ndarray
    step_attention_ms: np.ndarray
    step_routing_ms: np.ndarray
    step_prepare_ms: np.ndarray
    step_finalize_ms: np.ndarray
    step_ffn_ms: np.ndarray
    captured_step_kinds: np.ndarray
    layers: np.ndarray
    avg_histograms: np.ndarray
    num_forward_steps_total: int
    num_captured_steps: int
    num_dropped_steps: int

    def to_npz_payload(self) -> dict[str, np.ndarray]:
        step_all2all_ms = self.step_prepare_ms + self.step_finalize_ms
        return {
            "schema_version": np.asarray([SCHEMA_VERSION], dtype=np.int64),
            "batch_size": np.asarray([self.batch_size], dtype=np.int64),
            "draft_length": np.asarray([self.draft_length], dtype=np.int64),
            "selected_dataset_indices": self.selected_dataset_indices,
            "prompt_lengths": self.prompt_lengths,
            "condition_latency_ms": np.asarray(
                [self.condition_latency_ms], dtype=np.float64
            ),
            "step_histograms": self.step_histograms,
            "step_total_tokens": self.step_total_tokens,
            "step_total_ms": self.step_total_ms,
            "step_attention_ms": self.step_attention_ms,
            "step_routing_ms": self.step_routing_ms,
            "step_prepare_ms": self.step_prepare_ms,
            "step_finalize_ms": self.step_finalize_ms,
            "step_all2all_ms": step_all2all_ms,
            "step_ffn_ms": self.step_ffn_ms,
            "captured_step_kinds": self.captured_step_kinds,
            "layers": self.layers,
            "avg_histograms": self.avg_histograms,
            "num_forward_steps_total": np.asarray(
                [self.num_forward_steps_total], dtype=np.int64
            ),
            "num_captured_steps": np.asarray(
                [self.num_captured_steps], dtype=np.int64
            ),
            "num_dropped_steps": np.asarray(
                [self.num_dropped_steps], dtype=np.int64
            ),
        }


@dataclass
class CollectedConditionSummary:
    batch_size: int
    draft_length: int
    raw_path: str
    condition_latency_ms: float
    num_forward_steps_total: int
    num_captured_steps: int
    num_dropped_steps: int


@dataclass
class StepAccumulator:
    attention_ms: float = 0.0
    routing_ms: float = 0.0
    prepare_ms: float = 0.0
    finalize_ms: float = 0.0
    ffn_ms: float = 0.0


@dataclass
class WorkerInstrumentationState:
    enabled: bool = False
    pending_step_timings: deque[dict[str, float]] = field(default_factory=deque)
    current_step: StepAccumulator | None = None
    enter_step_logs: int = 0
    queued_step_logs: int = 0


_WORKER_STATE = WorkerInstrumentationState()
_ORIGINAL_WORKER_EXECUTE_MODEL = None
_ORIGINAL_ROUTER_SELECT_EXPERTS = None
_ORIGINAL_QWEN2_MLP_FORWARD = None
_ORIGINAL_QWEN3_MLP_FORWARD = None
_ORIGINAL_QWEN3_NEXT_ATTN_FORWARD = None
_ORIGINAL_QWEN_GDN_FORWARD = None
_ORIGINAL_MODULAR_PREPARE = None
_ORIGINAL_MODULAR_FUSED_EXPERTS = None
_ORIGINAL_MODULAR_FINALIZE = None
_ORIGINAL_MONOLITHIC_APPLY = None


def condition_name(batch_size: int, draft_length: int) -> str:
    return f"batch_{batch_size:03d}_draft_{draft_length:02d}"


def default_output_dir() -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    return Path("results") / f"qwen3_6_mtp_ep_{timestamp}"


def ensure_collect_dirs(output_dir: Path) -> dict[str, Path]:
    raw_dir = output_dir / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    return {"root": output_dir, "raw": raw_dir}


def prompt_cache_path(output_dir: Path) -> Path:
    return output_dir / "prompt_cache.json"


def save_run_metadata(output_dir: Path, args: Any) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "dataset_split": args.dataset_split,
        "batch_sizes": list(args.batch_sizes),
        "draft_lengths": list(args.draft_lengths),
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "layers": list(args.layers),
        "num_experts": args.num_experts,
        "enforce_eager": args.enforce_eager,
        "warmup_rounds": args.warmup_rounds,
        "vllm_enable_v1_multiprocessing": os.environ.get(
            "VLLM_ENABLE_V1_MULTIPROCESSING"
        ),
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, ensure_ascii=False, indent=2)


def save_collect_manifest(
    output_dir: Path,
    args: Any,
    condition_summaries: list[CollectedConditionSummary],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "dataset_split": args.dataset_split,
        "batch_sizes": list(args.batch_sizes),
        "draft_lengths": list(args.draft_lengths),
        "max_tokens": args.max_tokens,
        "layers": list(args.layers),
        "warmup_rounds": args.warmup_rounds,
        "conditions": [
            {
                "batch_size": summary.batch_size,
                "draft_length": summary.draft_length,
                "raw_path": summary.raw_path,
                "condition_latency_ms": summary.condition_latency_ms,
                "num_forward_steps_total": summary.num_forward_steps_total,
                "num_captured_steps": summary.num_captured_steps,
                "num_dropped_steps": summary.num_dropped_steps,
            }
            for summary in condition_summaries
        ],
    }
    with (output_dir / "collect_manifest.json").open("w", encoding="utf-8") as fp:
        json.dump(manifest, fp, ensure_ascii=False, indent=2)


def load_condition_summary(raw_path: Path) -> CollectedConditionSummary:
    with np.load(raw_path, allow_pickle=False) as data:
        batch_size = int(data["batch_size"][0])
        draft_length = int(data["draft_length"][0])
        condition_latency_ms = float(data["condition_latency_ms"][0])
        num_forward_steps_total = int(data["num_forward_steps_total"][0])
        num_captured_steps = int(data["num_captured_steps"][0])
        num_dropped_steps = int(data["num_dropped_steps"][0])
    return CollectedConditionSummary(
        batch_size=batch_size,
        draft_length=draft_length,
        raw_path=str(raw_path.relative_to(raw_path.parent.parent)),
        condition_latency_ms=condition_latency_ms,
        num_forward_steps_total=num_forward_steps_total,
        num_captured_steps=num_captured_steps,
        num_dropped_steps=num_dropped_steps,
    )


def load_prompt_items(args: Namespace) -> list[dict[str, list[int]]]:
    cache_path = getattr(args, "prompt_cache_path", None)
    if cache_path is not None:
        return load_prompt_items_from_cache(Path(cache_path))

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
    )
    dataset = load_dataset(
        args.dataset,
        args.dataset_config,
        split=args.dataset_split,
    )
    if hasattr(args, "batch_sizes"):
        max_samples = max(args.batch_sizes)
    else:
        max_samples = args.batch_size
    prompt_items: list[dict[str, list[int]]] = []
    for item in dataset.select(range(min(max_samples, len(dataset)))):
        token_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": item["question"]}],
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
        ).input_ids
        prompt_items.append({"prompt_token_ids": token_ids})
    return prompt_items


def save_prompt_items_cache(
    prompt_items: list[dict[str, list[int]]],
    cache_path: Path,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "prompt_token_ids": [item["prompt_token_ids"] for item in prompt_items],
    }
    with cache_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp)


def load_prompt_items_from_cache(cache_path: Path) -> list[dict[str, list[int]]]:
    with cache_path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    prompt_token_ids = payload["prompt_token_ids"]
    return [{"prompt_token_ids": list(token_ids)} for token_ids in prompt_token_ids]


def prepare_prompt_cache(args: Namespace, output_dir: Path) -> Path:
    cache_path = prompt_cache_path(output_dir)
    if cache_path.exists():
        return cache_path
    prompt_items = load_prompt_items(args)
    save_prompt_items_cache(prompt_items, cache_path)
    return cache_path


def create_llm(args: Namespace, batch_size: int, draft_length: int):
    from vllm import LLM

    speculative_config = None
    if draft_length > 0:
        speculative_config = {
            "method": "mtp",
            "num_speculative_tokens": draft_length,
            "max_model_len": args.max_model_len,
        }

    return LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        async_scheduling=False,
        enable_expert_parallel=True,
        enable_return_routed_experts=True,
        enable_eplb=False,
        max_model_len=args.max_model_len,
        max_num_seqs=batch_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        speculative_config=speculative_config,
        enforce_eager=args.enforce_eager,
    )


def get_inproc_handles(llm: Any) -> tuple[Any, Any]:
    engine_core_client = llm.llm_engine.engine_core
    if not hasattr(engine_core_client, "engine_core"):
        raise RuntimeError(
            "This experiment requires in-proc V1 execution. "
            "Set VLLM_ENABLE_V1_MULTIPROCESSING=0 before running."
        )
    engine_core = engine_core_client.engine_core
    return engine_core.scheduler, engine_core.model_executor


def _synchronize_device() -> None:
    from vllm.platforms import current_platform

    synchronize = current_platform.synchronize
    if synchronize is not None:
        synchronize()


def _measure_worker_section(label: str, fn: Callable, *args: Any, **kwargs: Any):
    current_step = _WORKER_STATE.current_step
    if not _WORKER_STATE.enabled or current_step is None:
        return fn(*args, **kwargs)

    _synchronize_device()
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    _synchronize_device()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    if label == "prepare":
        current_step.prepare_ms += elapsed_ms
    elif label == "finalize":
        current_step.finalize_ms += elapsed_ms
    elif label == "attention":
        current_step.attention_ms += elapsed_ms
    elif label == "routing":
        current_step.routing_ms += elapsed_ms
    elif label == "ffn":
        current_step.ffn_ms += elapsed_ms
    else:
        raise ValueError(f"Unknown measured label: {label}")
    return result


def _install_worker_hooks() -> None:
    global _ORIGINAL_WORKER_EXECUTE_MODEL
    global _ORIGINAL_ROUTER_SELECT_EXPERTS
    global _ORIGINAL_QWEN2_MLP_FORWARD
    global _ORIGINAL_QWEN3_MLP_FORWARD
    global _ORIGINAL_QWEN3_NEXT_ATTN_FORWARD
    global _ORIGINAL_QWEN_GDN_FORWARD
    global _ORIGINAL_MODULAR_PREPARE
    global _ORIGINAL_MODULAR_FUSED_EXPERTS
    global _ORIGINAL_MODULAR_FINALIZE
    global _ORIGINAL_MONOLITHIC_APPLY

    if _ORIGINAL_WORKER_EXECUTE_MODEL is not None:
        return

    from types import NoneType

    from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
    )
    from vllm.model_executor.layers.fused_moe.modular_kernel import (
        FusedMoEKernelModularImpl,
        FusedMoEKernelMonolithicImpl,
    )
    from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP
    from vllm.model_executor.models.qwen3_moe import Qwen3MoeMLP
    from vllm.model_executor.models.qwen3_next import Qwen3NextAttention
    from vllm.v1.worker.gpu_worker import Worker

    _ORIGINAL_WORKER_EXECUTE_MODEL = Worker.execute_model
    _ORIGINAL_ROUTER_SELECT_EXPERTS = BaseRouter.select_experts
    _ORIGINAL_QWEN2_MLP_FORWARD = Qwen2MoeMLP.forward
    _ORIGINAL_QWEN3_MLP_FORWARD = Qwen3MoeMLP.forward
    _ORIGINAL_QWEN3_NEXT_ATTN_FORWARD = Qwen3NextAttention.forward
    _ORIGINAL_QWEN_GDN_FORWARD = QwenGatedDeltaNetAttention.forward
    _ORIGINAL_MODULAR_PREPARE = FusedMoEKernelModularImpl._prepare
    _ORIGINAL_MODULAR_FUSED_EXPERTS = FusedMoEKernelModularImpl._fused_experts
    _ORIGINAL_MODULAR_FINALIZE = FusedMoEKernelModularImpl._finalize
    _ORIGINAL_MONOLITHIC_APPLY = FusedMoEKernelMonolithicImpl.apply

    def patched_worker_execute_model(self, scheduler_output):
        if (
            not _WORKER_STATE.enabled
            or scheduler_output.total_num_scheduled_tokens <= 0
        ):
            return _ORIGINAL_WORKER_EXECUTE_MODEL(self, scheduler_output)

        if _WORKER_STATE.enter_step_logs < 3:
            print(
                "[worker_timing] execute begin "
                f"scheduled_tokens={scheduler_output.total_num_scheduled_tokens}",
                flush=True,
            )
            _WORKER_STATE.enter_step_logs += 1

        _WORKER_STATE.current_step = StepAccumulator()
        _synchronize_device()
        start = time.perf_counter()
        try:
            output = _ORIGINAL_WORKER_EXECUTE_MODEL(self, scheduler_output)
            _synchronize_device()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            assert _WORKER_STATE.current_step is not None
            _WORKER_STATE.pending_step_timings.append(
                {
                    "total_ms": elapsed_ms,
                    "attention_ms": _WORKER_STATE.current_step.attention_ms,
                    "routing_ms": _WORKER_STATE.current_step.routing_ms,
                    "prepare_ms": _WORKER_STATE.current_step.prepare_ms,
                    "finalize_ms": _WORKER_STATE.current_step.finalize_ms,
                    "ffn_ms": _WORKER_STATE.current_step.ffn_ms,
                }
            )
            if _WORKER_STATE.queued_step_logs < 3:
                print(
                    "[worker_timing] queued "
                    f"total_ms={elapsed_ms:.3f} "
                    f"attention_ms={_WORKER_STATE.current_step.attention_ms:.3f} "
                    f"routing_ms={_WORKER_STATE.current_step.routing_ms:.3f} "
                    f"prepare_ms={_WORKER_STATE.current_step.prepare_ms:.3f} "
                    f"finalize_ms={_WORKER_STATE.current_step.finalize_ms:.3f} "
                    f"ffn_ms={_WORKER_STATE.current_step.ffn_ms:.3f}",
                    flush=True,
                )
                _WORKER_STATE.queued_step_logs += 1
            return output
        finally:
            _WORKER_STATE.current_step = None

    def patched_router_select_experts(self, *args, **kwargs):
        return _measure_worker_section(
            "routing",
            _ORIGINAL_ROUTER_SELECT_EXPERTS,
            self,
            *args,
            **kwargs,
        )

    def patched_qwen2_mlp_forward(self, *args, **kwargs):
        return _measure_worker_section(
            "ffn",
            _ORIGINAL_QWEN2_MLP_FORWARD,
            self,
            *args,
            **kwargs,
        )

    def patched_qwen3_mlp_forward(self, *args, **kwargs):
        return _measure_worker_section(
            "ffn",
            _ORIGINAL_QWEN3_MLP_FORWARD,
            self,
            *args,
            **kwargs,
        )

    def patched_qwen3_next_attention_forward(self, *args, **kwargs):
        return _measure_worker_section(
            "attention",
            _ORIGINAL_QWEN3_NEXT_ATTN_FORWARD,
            self,
            *args,
            **kwargs,
        )

    def patched_qwen_gdn_forward(self, *args, **kwargs):
        return _measure_worker_section(
            "attention",
            _ORIGINAL_QWEN_GDN_FORWARD,
            self,
            *args,
            **kwargs,
        )

    def patched_modular_prepare(self, *args, **kwargs):
        return _measure_worker_section(
            "prepare",
            _ORIGINAL_MODULAR_PREPARE,
            self,
            *args,
            **kwargs,
        )

    def patched_modular_fused_experts(self, *args, **kwargs):
        return _measure_worker_section(
            "ffn",
            _ORIGINAL_MODULAR_FUSED_EXPERTS,
            self,
            *args,
            **kwargs,
        )

    def patched_modular_finalize(self, *args, **kwargs):
        return _measure_worker_section(
            "finalize",
            _ORIGINAL_MODULAR_FINALIZE,
            self,
            *args,
            **kwargs,
        )

    def patched_monolithic_apply(
        self,
        hidden_states,
        w1,
        w2,
        router_logits,
        activation,
        global_num_experts,
        expert_map,
        apply_router_weight_on_input,
        num_expert_group=None,
        e_score_correction_bias=None,
        routed_scaling_factor=None,
        topk_group=None,
    ):
        if not _WORKER_STATE.enabled or _WORKER_STATE.current_step is None:
            return _ORIGINAL_MONOLITHIC_APPLY(
                self,
                hidden_states,
                w1,
                w2,
                router_logits,
                activation,
                global_num_experts,
                expert_map,
                apply_router_weight_on_input,
                num_expert_group=num_expert_group,
                e_score_correction_bias=e_score_correction_bias,
                routed_scaling_factor=routed_scaling_factor,
                topk_group=topk_group,
            )

        a1q, a1q_scale, router_logits = _measure_worker_section(
            "prepare",
            self.prepare_finalize.prepare,
            hidden_states,
            router_logits=router_logits,
            quant_config=self.fused_experts.quant_config,
            defer_input_quant=self.fused_experts.expects_unquantized_inputs,
        )
        fused_out = _measure_worker_section(
            "ffn",
            self.fused_experts.apply,
            hidden_states=a1q,
            w1=w1,
            w2=w2,
            router_logits=router_logits,
            activation=activation,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            apply_router_weight_on_input=apply_router_weight_on_input,
            a1q_scale=a1q_scale,
            num_expert_group=num_expert_group,
            e_score_correction_bias=e_score_correction_bias,
            routed_scaling_factor=routed_scaling_factor,
            topk_group=topk_group,
        )
        return _measure_worker_section(
            "finalize",
            self.prepare_finalize.finalize,
            fused_out,
        )

    Worker.execute_model = patched_worker_execute_model
    BaseRouter.select_experts = patched_router_select_experts
    Qwen2MoeMLP.forward = patched_qwen2_mlp_forward
    Qwen3MoeMLP.forward = patched_qwen3_mlp_forward
    Qwen3NextAttention.forward = patched_qwen3_next_attention_forward
    QwenGatedDeltaNetAttention.forward = patched_qwen_gdn_forward
    FusedMoEKernelModularImpl._prepare = patched_modular_prepare
    FusedMoEKernelModularImpl._fused_experts = patched_modular_fused_experts
    FusedMoEKernelModularImpl._finalize = patched_modular_finalize
    FusedMoEKernelMonolithicImpl.apply = patched_monolithic_apply


def install_experiment_hooks_worker(worker: Any) -> bool:
    _install_worker_hooks()
    _WORKER_STATE.enabled = False
    _WORKER_STATE.pending_step_timings.clear()
    _WORKER_STATE.current_step = None
    _WORKER_STATE.enter_step_logs = 0
    _WORKER_STATE.queued_step_logs = 0
    return True


def start_condition_collection_worker(worker: Any) -> bool:
    _WORKER_STATE.enabled = True
    _WORKER_STATE.pending_step_timings.clear()
    _WORKER_STATE.current_step = None
    _WORKER_STATE.enter_step_logs = 0
    _WORKER_STATE.queued_step_logs = 0
    return True


def stop_condition_collection_worker(worker: Any) -> dict[str, int]:
    _WORKER_STATE.enabled = False
    pending = len(_WORKER_STATE.pending_step_timings)
    _WORKER_STATE.pending_step_timings.clear()
    _WORKER_STATE.current_step = None
    return {"pending_timings": pending}


def pop_step_timing_worker(
    worker: Any,
    timeout_s: float = 5.0,
    poll_s: float = 0.01,
) -> dict[str, float] | None:
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        if _WORKER_STATE.pending_step_timings:
            return _WORKER_STATE.pending_step_timings.popleft()
        time.sleep(poll_s)
    if _WORKER_STATE.pending_step_timings:
        return _WORKER_STATE.pending_step_timings.popleft()
    return None


def _close_ffn_component(step_timing: StepTiming, tol_ms: float = 1e-3) -> float:
    unattributed_ms = step_timing.unattributed_ms
    if unattributed_ms < -tol_ms:
        raise RuntimeError(
            "Step timing components exceeded total step time: "
            f"total_ms={step_timing.total_ms:.6f}, "
            f"attention_ms={step_timing.attention_ms:.6f}, "
            f"routing_ms={step_timing.routing_ms:.6f}, "
            f"all2all_ms={step_timing.all2all_ms:.6f}, "
            f"ffn_ms={step_timing.ffn_ms:.6f}, "
            f"unattributed_ms={unattributed_ms:.6f}"
        )
    return step_timing.ffn_ms + max(unattributed_ms, 0.0)


class SchedulerStepRecorder:
    def __init__(
        self,
        scheduler: Any,
        model_executor: Any,
        *,
        use_spec_decode: bool,
        layers: tuple[int, ...],
        num_experts: int,
    ) -> None:
        self.scheduler = scheduler
        self.model_executor = model_executor
        self.use_spec_decode = use_spec_decode
        self.layers = layers
        self.num_experts = num_experts
        self._original_update = None
        self.step_histograms: list[np.ndarray] = []
        self.step_total_tokens: list[int] = []
        self.step_total_ms: list[float] = []
        self.step_attention_ms: list[float] = []
        self.step_routing_ms: list[float] = []
        self.step_prepare_ms: list[float] = []
        self.step_finalize_ms: list[float] = []
        self.step_ffn_ms: list[float] = []
        self.step_kinds: list[str] = []
        self.num_forward_steps_total = 0
        self.num_dropped_steps = 0
        self.debug_update_logs = 0

    def __enter__(self) -> "SchedulerStepRecorder":
        self._original_update = self.scheduler.update_from_output

        def wrapped_update(
            scheduler_self: Any,
            scheduler_output: Any,
            model_runner_output: Any,
        ) -> Any:
            if self.debug_update_logs < 5:
                print(
                    "[recorder] update begin "
                    f"use_spec_decode={self.use_spec_decode} "
                    f"num_forward_steps_total={self.num_forward_steps_total}",
                    flush=True,
                )
            worker_timings = self.model_executor.collective_rpc(
                pop_step_timing_worker,
                timeout=30,
            )
            step_timing = aggregate_worker_step_timings(worker_timings)
            self.num_forward_steps_total += 1

            captured_step = select_step_routing_data(
                scheduler_output,
                model_runner_output,
                use_spec_decode=self.use_spec_decode,
            )
            if captured_step is None:
                self.num_dropped_steps += 1
            else:
                self.step_histograms.append(
                    count_layer_expert_histograms(
                        captured_step.routing_data,
                        layers=self.layers,
                        num_experts=self.num_experts,
                    )
                )
                self.step_total_tokens.append(captured_step.total_scheduled_tokens)
                self.step_total_ms.append(step_timing.total_ms)
                self.step_attention_ms.append(step_timing.attention_ms)
                self.step_routing_ms.append(step_timing.routing_ms)
                self.step_prepare_ms.append(step_timing.prepare_ms)
                self.step_finalize_ms.append(step_timing.finalize_ms)
                self.step_ffn_ms.append(_close_ffn_component(step_timing))
                self.step_kinds.append(captured_step.step_kind)

            if self.debug_update_logs < 5:
                print(
                    "[recorder] update end "
                    f"use_spec_decode={self.use_spec_decode} "
                    f"captured={captured_step is not None} "
                    f"step_kind={captured_step.step_kind if captured_step else 'none'} "
                    f"total_ms={step_timing.total_ms:.3f}",
                    flush=True,
                )
                self.debug_update_logs += 1

            return self._original_update(scheduler_output, model_runner_output)

        self.scheduler.update_from_output = MethodType(wrapped_update, self.scheduler)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._original_update is not None:
            self.scheduler.update_from_output = self._original_update


def collect_condition(
    args: Namespace,
    prompt_items: list[dict[str, list[int]]],
    raw_dir: Path,
    *,
    batch_size: int,
    draft_length: int,
) -> CollectedConditionSummary:
    from vllm import SamplingParams

    print(
        f"[collect] start batch_size={batch_size} draft_length={draft_length}",
        flush=True,
    )
    llm = create_llm(args, batch_size, draft_length)
    print(
        f"[collect] llm created batch_size={batch_size} draft_length={draft_length}",
        flush=True,
    )
    scheduler, model_executor = get_inproc_handles(llm)
    print(
        f"[collect] inproc handles ready batch_size={batch_size} "
        f"draft_length={draft_length} model_executor={type(model_executor).__name__}",
        flush=True,
    )
    print(
        f"[collect] install hooks begin batch_size={batch_size} "
        f"draft_length={draft_length}",
        flush=True,
    )
    model_executor.collective_rpc(install_experiment_hooks_worker, timeout=30)
    print(
        f"[collect] install hooks end batch_size={batch_size} "
        f"draft_length={draft_length}",
        flush=True,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        ignore_eos=True,
    )
    use_spec_decode = draft_length > 0
    selected_indices = select_dataset_indices(batch_size, len(prompt_items))
    prompt_batch = [prompt_items[idx] for idx in selected_indices]
    prompt_lengths = np.asarray(
        [len(item["prompt_token_ids"]) for item in prompt_batch],
        dtype=np.int64,
    )
    if args.warmup_rounds > 0:
        print(
            f"[collect] warmup begin batch_size={batch_size} "
            f"draft_length={draft_length} rounds={args.warmup_rounds}",
            flush=True,
        )
        for warmup_idx in range(args.warmup_rounds):
            llm.generate(prompt_batch, sampling_params=sampling_params, use_tqdm=False)
            print(
                f"[collect] warmup done batch_size={batch_size} "
                f"draft_length={draft_length} round={warmup_idx + 1}",
                flush=True,
            )
    print(
        f"[collect] start collection begin batch_size={batch_size} "
        f"draft_length={draft_length}",
        flush=True,
    )
    model_executor.collective_rpc(start_condition_collection_worker, timeout=30)
    print(
        f"[collect] start collection end batch_size={batch_size} "
        f"draft_length={draft_length}",
        flush=True,
    )

    try:
        with SchedulerStepRecorder(
            scheduler,
            model_executor,
            use_spec_decode=use_spec_decode,
            layers=tuple(args.layers),
            num_experts=args.num_experts,
        ) as recorder:
            print(
                f"[collect] generate begin batch_size={batch_size} "
                f"draft_length={draft_length}",
                flush=True,
            )
            start = time.perf_counter()
            llm.generate(prompt_batch, sampling_params=sampling_params, use_tqdm=False)
            condition_latency_ms = (time.perf_counter() - start) * 1000.0
            print(
                f"[collect] generate end batch_size={batch_size} "
                f"draft_length={draft_length}",
                flush=True,
            )

        pending_counts = model_executor.collective_rpc(stop_condition_collection_worker)
        pending_values = [item["pending_timings"] for item in pending_counts]
        if any(pending_values):
            if len(set(pending_values)) != 1:
                raise RuntimeError(
                    "Worker timing queues ended with inconsistent leftover counts: "
                    f"{pending_counts}"
                )
            print(
                "[collect] warning leftover worker timings were discarded "
                f"batch_size={batch_size} draft_length={draft_length} "
                f"pending_per_worker={pending_values[0]}",
                flush=True,
            )

        if not recorder.step_histograms:
            raise RuntimeError(
                "No routed-expert steps were captured for "
                f"batch_size={batch_size}, draft_length={draft_length}."
            )

        step_histograms = np.stack(recorder.step_histograms, axis=0)
        step_total_tokens = np.asarray(recorder.step_total_tokens, dtype=np.int64)
        step_total_ms = np.asarray(recorder.step_total_ms, dtype=np.float64)
        step_attention_ms = np.asarray(recorder.step_attention_ms, dtype=np.float64)
        step_routing_ms = np.asarray(recorder.step_routing_ms, dtype=np.float64)
        step_prepare_ms = np.asarray(recorder.step_prepare_ms, dtype=np.float64)
        step_finalize_ms = np.asarray(recorder.step_finalize_ms, dtype=np.float64)
        step_ffn_ms = np.asarray(recorder.step_ffn_ms, dtype=np.float64)
        avg_histograms = average_step_histograms(step_histograms)
        raw_data = ConditionRawData(
            batch_size=batch_size,
            draft_length=draft_length,
            selected_dataset_indices=selected_indices,
            prompt_lengths=prompt_lengths,
            condition_latency_ms=condition_latency_ms,
            step_histograms=step_histograms,
            step_total_tokens=step_total_tokens,
            step_total_ms=step_total_ms,
            step_attention_ms=step_attention_ms,
            step_routing_ms=step_routing_ms,
            step_prepare_ms=step_prepare_ms,
            step_finalize_ms=step_finalize_ms,
            step_ffn_ms=step_ffn_ms,
            captured_step_kinds=np.asarray(recorder.step_kinds),
            layers=np.asarray(args.layers, dtype=np.int64),
            avg_histograms=avg_histograms,
            num_forward_steps_total=recorder.num_forward_steps_total,
            num_captured_steps=len(recorder.step_histograms),
            num_dropped_steps=recorder.num_dropped_steps,
        )
        raw_path = raw_dir / f"{condition_name(batch_size, draft_length)}.npz"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(raw_path, **raw_data.to_npz_payload())
        print(
            "[collect] finished "
            f"batch_size={batch_size} draft_length={draft_length} "
            f"captured_steps={len(recorder.step_histograms)} "
            f"latency_ms={condition_latency_ms:.3f}",
            flush=True,
        )
        return CollectedConditionSummary(
            batch_size=batch_size,
            draft_length=draft_length,
            raw_path=str(raw_path.relative_to(raw_dir.parent)),
            condition_latency_ms=condition_latency_ms,
            num_forward_steps_total=recorder.num_forward_steps_total,
            num_captured_steps=len(recorder.step_histograms),
            num_dropped_steps=recorder.num_dropped_steps,
        )
    finally:
        print(
            f"[collect] teardown begin batch_size={batch_size} "
            f"draft_length={draft_length}",
            flush=True,
        )
        try:
            llm.llm_engine.engine_core.shutdown()
            print(
                f"[collect] shutdown done batch_size={batch_size} "
                f"draft_length={draft_length}",
                flush=True,
            )
        except Exception:
            print(
                f"[collect] shutdown failed batch_size={batch_size} "
                f"draft_length={draft_length}",
                flush=True,
            )
        del llm
        print(
            f"[collect] llm deleted batch_size={batch_size} "
            f"draft_length={draft_length}",
            flush=True,
        )
        print(
            f"[collect] teardown done batch_size={batch_size} "
            f"draft_length={draft_length}",
            flush=True,
        )


def collect_one_condition(args: Namespace, output_dir: Path) -> CollectedConditionSummary:
    dirs = ensure_collect_dirs(output_dir)
    prompt_items = load_prompt_items(args)
    return collect_condition(
        args,
        prompt_items,
        dirs["raw"],
        batch_size=args.batch_size,
        draft_length=args.draft_length,
    )


def _build_collect_one_command(
    args: Namespace,
    output_dir: Path,
    entrypoint: Path,
    *,
    batch_size: int,
    draft_length: int,
    prompt_cache: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(entrypoint),
        "collect-one",
        "--model",
        args.model,
        "--dataset",
        args.dataset,
        "--dataset-config",
        args.dataset_config,
        "--dataset-split",
        args.dataset_split,
        "--batch-size",
        str(batch_size),
        "--draft-length",
        str(draft_length),
        "--max-tokens",
        str(args.max_tokens),
        "--max-model-len",
        str(args.max_model_len),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--num-experts",
        str(args.num_experts),
        "--output-dir",
        str(output_dir),
        "--prompt-cache-path",
        str(prompt_cache),
        "--warmup-rounds",
        str(args.warmup_rounds),
    ]
    command.extend(["--layers", *(str(layer) for layer in args.layers)])
    if args.enforce_eager:
        command.append("--enforce-eager")
    else:
        command.append("--no-enforce-eager")
    return command


def collect_experiment(args: Namespace, output_dir: Path, entrypoint: Path) -> None:
    dirs = ensure_collect_dirs(output_dir)
    save_run_metadata(dirs["root"], args)
    prompts_cache = prepare_prompt_cache(args, dirs["root"])
    condition_summaries: list[CollectedConditionSummary] = []
    for batch_size in args.batch_sizes:
        for draft_length in args.draft_lengths:
            print(
                f"[collect-parent] launching batch_size={batch_size} "
                f"draft_length={draft_length}",
                flush=True,
            )
            command = _build_collect_one_command(
                args,
                dirs["root"],
                entrypoint,
                batch_size=batch_size,
                draft_length=draft_length,
                prompt_cache=prompts_cache,
            )
            subprocess.run(
                command,
                check=True,
                cwd=entrypoint.parent.parent.parent.parent,
                env=os.environ.copy(),
            )
            raw_path = dirs["raw"] / f"{condition_name(batch_size, draft_length)}.npz"
            summary = load_condition_summary(raw_path)
            condition_summaries.append(summary)

    save_collect_manifest(dirs["root"], args, condition_summaries)
