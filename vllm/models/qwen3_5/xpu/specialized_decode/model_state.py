# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.5 XPU batch semantics over the standard hybrid ModelState."""

from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState


class Qwen3_5XPUModelState(MambaHybridModelState):
    """Retain stock KV/SSM handling and plan only the optional MoE stage."""

    def __init__(self, vllm_config, model, encoder_cache, device):
        super().__init__(vllm_config, model, encoder_cache, device)
        language_model = getattr(model, "language_model", model)
        self.decode_runtime = language_model._xpu_moe_decode_runtime

    def supports_decode_cudagraph_batch(self, *, has_prefill: bool) -> bool:
        # The stock runner can reinterpret a one-token prompt tail as decode.
        # Such a step must not replay a graph captured with specialized MoE.
        return not has_prefill

    def prepare_inputs(self, input_batch, req_states):
        inputs = super().prepare_inputs(input_batch, req_states)
        self.decode_runtime.prepare_step(input_batch)
        return inputs

    def prepare_dummy_inputs(self, num_reqs, num_tokens):
        inputs = super().prepare_dummy_inputs(num_reqs, num_tokens)
        self.decode_runtime.prepare_capture(num_reqs, num_tokens)
        return inputs
