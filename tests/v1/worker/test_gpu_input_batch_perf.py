# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.worker.gpu_input_batch import InputBatch


@pytest.mark.cpu_test
def test_update_async_output_token_ids_handles_long_outputs_quickly() -> None:
    num_reqs = 128
    output_len = 10_000
    output_token_ids = [list(range(output_len)) + [-1] for _ in range(num_reqs)]
    event = SimpleNamespace(synchronize=lambda: None)
    input_batch = SimpleNamespace(
        sampling_metadata=SimpleNamespace(output_token_ids=output_token_ids),
        sampled_token_ids_cpu=torch.full(
            (num_reqs, 1), output_len + 1, dtype=torch.long
        ),
        prev_req_id_to_index={f"req{i}": i for i in range(num_reqs)},
        req_ids=[f"req{i}" for i in range(num_reqs)],
        async_copy_ready_event=event,
    )

    start = time.perf_counter()
    InputBatch.update_async_output_token_ids(input_batch)
    elapsed_s = time.perf_counter() - start

    for output_ids in output_token_ids:
        assert output_ids == list(range(output_len)) + [output_len + 1]

    # The previous implementation scanned from the beginning of each request's
    # long output history. The fixed path should only inspect trailing
    # placeholders, so this synthetic long-generation case should stay tiny.
    assert elapsed_s < 0.005
