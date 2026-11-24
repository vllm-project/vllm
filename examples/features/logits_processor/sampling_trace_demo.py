# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run from the repository root with a supported GPU and Model Runner V2."""

from vllm import LLM, SamplingParams


def main():
    llm = LLM(
        model="facebook/opt-125m",
        distributed_executor_backend="uni",
        enforce_eager=True,
        logits_processors=[
            "examples.features.logits_processor.sampling_trace:SamplingTraceProcessor"
        ],
        worker_extension_cls=(
            "examples.features.logits_processor.sampling_trace.SamplingTraceWorkerExtension"
        ),
        additional_config={"sampling_trace": {"max_steps": 2, "max_bytes": 67108864}},
    )
    outputs = llm.generate(
        ["Hello, my name is", "The capital of France is"],
        SamplingParams(
            temperature=0, max_tokens=2, extra_args={"sampling_trace_label": "demo"}
        ),
    )
    for output in outputs:
        print(output.outputs[0].text)
    # All requests have completed. Host inspection occurs outside generation.
    for processors in llm.collective_rpc("take_sampling_traces"):
        for snapshots in processors:
            for snapshot in snapshots:
                print(snapshot.positions.cpu().tolist(), snapshot.logits.shape)


if __name__ == "__main__":
    main()
