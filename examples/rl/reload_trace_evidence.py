# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker extension for inspecting production reload tracing in day0 tests."""

import hashlib

import torch

from vllm.model_executor.model_loader.reload.integration import get_model_reload_tracer


class ReloadTraceEvidence:
    def inspect_reload_trace(self, arm=False):
        trace = get_model_reload_tracer(self.model_runner.model)
        result = {}
        for key, state in trace.states.items():
            method = getattr(state.module, "quant_method", None)
            if arm and method is not None:

                def forbidden(*args, **kwargs):
                    raise AssertionError("Post-load processing called during reload")

                method.process_weights_after_loading = forbidden
            tensors = {}
            for role, target in state.targets.items():
                value = target.tensor
                raw = value.detach().contiguous().reshape(-1).view(torch.uint8)
                tensors[role] = {
                    "id": id(value),
                    "ptr": value.data_ptr(),
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "hash": hashlib.sha256(raw.cpu().numpy().tobytes()).hexdigest(),
                }
            result[key] = {
                "policy": type(state.policy).__name__,
                "method": id(method),
                "kernel": id(getattr(method, "moe_kernel", None)),
                "config": id(getattr(method, "moe_quant_config", None)),
                "complete": state.complete,
                "staging": bool(state.checkpoint),
                "tensors": tensors,
            }
        assert result
        return result
