# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm import LLM, SamplingParams
from vllm.plugins.observation import (
    ObservationAction,
    ObservationPlugin,
    ObservationResult,
    RequestContext,
)


class DummyObservationPlugin(ObservationPlugin):
    def __init__(self, vllm_config=None):
        self.prefill_chunks = []
        self.decode_steps = 0
        self.request_started = False
        self.request_completed = False

    def get_observation_layers(self) -> list[int]:
        return [-1]

    def on_request_start(self, request_id: str, prompt: str | None = None) -> None:
        self.request_started = True

    def on_step_batch(
        self,
        batch_hidden_states: dict[int, torch.Tensor],
        request_contexts: list[RequestContext],
    ) -> list[ObservationResult]:
        for ctx in request_contexts:
            # You can slice the tensor for this specific request like so:
            start = ctx.batch_offset
            end = start + ctx.num_tokens
            _req_tensor = batch_hidden_states[-1][start:end]

            if ctx.is_prefill:
                self.prefill_chunks.append(ctx.num_cached_tokens)
            else:
                self.decode_steps += 1

            # If decode steps reach 5, let's abort it to test ABORT logic
            if self.decode_steps == 5:
                return [ObservationResult(action=ObservationAction.ABORT)] * len(
                    request_contexts
                )

        return [ObservationResult(action=ObservationAction.CONTINUE)] * len(
            request_contexts
        )

    def on_request_complete(self, request_id: str) -> None:
        self.request_completed = True


dummy_plugin = DummyObservationPlugin()

print("Initializing LLM with V1 Architecture...")
# Assuming Llama-2-7B or similar is available, or opt-125m for fast testing
llm = LLM(
    model="facebook/opt-125m",
    observation_plugins=[dummy_plugin],
    max_num_seqs=1,
    enforce_eager=True,
)

prompts = [
    "Hello, my name is",
]
sampling_params = SamplingParams(temperature=0.0, max_tokens=10)
# noqa: E501
print("Running offline inference...")
outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    finish_reason = output.outputs[0].finish_reason
    print(
        f"Prompt: {prompt!r}, Generated text: {generated_text!r}, "
        f"Finish reason: {finish_reason!r}"
    )

print(f"Plugin received request_start: {dummy_plugin.request_started}")
print(f"Plugin received request_complete: {dummy_plugin.request_completed}")
print(f"Plugin observed {len(dummy_plugin.prefill_chunks)} prefill chunk(s)")
print(f"Plugin observed {dummy_plugin.decode_steps} decode step(s)")
