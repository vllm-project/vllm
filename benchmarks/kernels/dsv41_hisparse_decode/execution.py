# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small worker-side receipts of actual graph dispatch, without GPU timing hooks."""


def execution_counts(worker, action):
    runner = worker.model_runner
    if action == "install":
        assert not hasattr(runner, "_decode_execution_records")
        records = []
        runner._decode_execution_records = records
        original = runner.step_timing.record_batch

        def record(input_batch, full_cudagraph):
            records.append(
                (bool(full_cudagraph), input_batch.num_tokens, input_batch.num_reqs)
            )
            return original(input_batch, full_cudagraph)

        runner.step_timing.record_batch = record
        return dict(installed=True)
    assert action in ("reset", "read")
    records = runner._decode_execution_records
    if action == "reset":
        records.clear()
        return dict(reset=True)
    return dict(rank=worker.rank, steps=list(records))
