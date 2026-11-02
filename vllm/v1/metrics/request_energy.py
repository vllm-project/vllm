# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal request snapshots; hardware collection lives outside vLLM."""

import os
import uuid


class SchedulerEnergyTrace:
    def __init__(self, config):
        self.lifecycle = None
        self.step = 0
        self.ranges = {}
        self.config = config
        if os.environ.get("REQUEST_ENERGY_TRACE_DIR"):
            from request_energy.instrumentation import LifecycleTracer

            self.lifecycle = LifecycleTracer("vllm", "scheduler")
        self.session = uuid.uuid4().hex

    @staticmethod
    def client_id(request):
        params = request.sampling_params
        extra = params.extra_args if params is not None else None
        return extra.get("request_energy_client_id") if extra else None

    def arrive(self, request):
        if self.lifecycle:
            self.lifecycle.request(
                request.request_id, "arrive", client_id=self.client_id(request)
            )

    def finish(self, request):
        if self.lifecycle:
            self.lifecycle.request(
                request.request_id,
                "cancel" if "ABORT" in request.status.name else "finish",
                outcome=request.status.name,
            )
            self.ranges.pop(request.request_id, None)

    def release(self, request):
        if self.lifecycle:
            self.lifecycle.request(request.request_id, "release")

    def snapshot(self, scheduler, output):
        if not self.lifecycle:
            return
        self.step += 1
        execution_id = f"{self.session}:{self.step}"
        if output.total_num_scheduled_tokens:
            self.lifecycle.hold(execution_id)
        owners = []
        supported = (
            self.config.speculative_config is None
            and self.config.parallel_config.pipeline_parallel_size == 1
            and self.config.parallel_config.data_parallel_size == 1
            and not scheduler.has_mamba_layers
            and scheduler.connector is None
            and not scheduler.is_encoder_decoder
            and not self.config.model_config.is_multimodal_model
            and not getattr(self.config.offload_config.uva, "cpu_offload_gb", 0)
            and not self.config.offload_config.prefetch.offload_group_size
            and self.config.model_config.runner_type == "generate"
        )
        for rid, count in output.num_scheduled_tokens.items():
            request = scheduler.requests[rid]
            before = request.num_computed_tokens
            prompt = max(0, min(count, request.num_prompt_tokens - before))
            decode = count - prompt
            previous = self.ranges.get(rid, [])
            repeated = sum(
                max(0, min(before + count, right) - max(before, left))
                for left, right in previous
            )
            merged: list[tuple[int, int]] = []
            for left, right in sorted([*previous, (before, before + count)]):
                if merged and left <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(right, merged[-1][1]))
                else:
                    merged.append((left, right))
            self.ranges[rid] = merged
            phase = "mixed" if prompt and decode else "prefill" if prompt else "decode"
            self.lifecycle.admit(rid, client_id=self.client_id(request))
            owners.append(
                {
                    "request_id": rid,
                    "phase": phase,
                    "computed_tokens": count,
                    "prefill_tokens": prompt,
                    "decode_tokens": decode,
                    "context_tokens": before + count,
                    "recompute_tokens": repeated,
                    "kv_read_token_pairs": (
                        None
                        if getattr(
                            self.config.model_config.hf_text_config,
                            "sliding_window",
                            None,
                        )
                        else count * before + count * (count + 1) // 2
                    ),
                    "cache_reused_tokens": (
                        before if rid not in self.ranges or not previous else None
                    ),
                    "flops_estimate": None,
                    "read_bytes_estimate": None,
                    "write_bytes_estimate": None,
                }
            )
        output.request_energy_trace = {
            "execution_id": execution_id,
            "owners": owners,
            "supported": supported,
        }

    def complete(self, output):
        if self.lifecycle and output.request_energy_trace:
            self.lifecycle.release_hold(output.request_energy_trace["execution_id"])


class WorkerEnergyTrace:
    def __init__(self, device, rank):
        self.tracer = None
        self.context = None
        if os.environ.get("REQUEST_ENERGY_TRACE_DIR"):
            from request_energy.instrumentation import CudaStepTracer

            self.tracer = CudaStepTracer("vllm", f"worker-{rank}", device=device)

    def begin_forward(self, output):
        self.context = getattr(output, "request_energy_trace", None)
        if self.tracer and self.context and output.total_num_scheduled_tokens:
            key = self.context["execution_id"] + ":forward"
            self.tracer.begin(
                key,
                self.context["owners"],
                supported=self.context["supported"],
                logical_step_id=self.context["execution_id"],
            )
            return key
        return None

    def begin_sample(self):
        if self.tracer and self.context:
            from request_energy.instrumentation import sampling_owners

            key = self.context["execution_id"] + ":sampling"
            owners = sampling_owners(self.context["owners"])
            self.tracer.begin(
                key,
                owners,
                supported=self.context["supported"],
                work_kind="sampling",
                logical_step_id=self.context["execution_id"],
            )
            return key
        return None

    def end(self, key):
        if self.tracer and key:
            self.tracer.end(key)
