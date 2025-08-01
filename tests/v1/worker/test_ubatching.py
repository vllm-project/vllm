# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import threading

import torch
import torch.nn.functional as F

from vllm.v1.worker.ubatching import (make_ubatch_contexts,
                                      yield_and_switch_from_comm_to_compute,
                                      yield_and_switch_from_compute_to_comm)


def layer(static_input, weight, static_output):
    intermediate_output_1 = F.linear(static_input, weight)
    intermediate_output_2 = F.linear(static_input, weight)
    static_output.copy_(intermediate_output_1 + intermediate_output_2)


def ubatch_layer(static_input, weight, static_output):
    intermediate_output_1 = F.linear(static_input, weight)
    yield_and_switch_from_compute_to_comm()
    intermediate_output_2 = F.linear(static_input, weight)
    yield_and_switch_from_comm_to_compute()
    static_output.copy_(intermediate_output_1 + intermediate_output_2)


def ubatch_thread_fn(static_input, weight, static_output, ubatch_context):
    with ubatch_context:
        ubatch_layer(static_input, weight, static_output)


def test_ubatch_context():
    M = 128
    N = 256
    K = 512

    input_ids = torch.randn(M, K, device='cuda')
    weight = torch.randn(N, K, device='cuda')

    # Static tensors for graph
    outputs = torch.empty(M, N, device='cuda')
    test_outputs = torch.empty(M, N, device='cuda')

    num_ubatches = 2

    compute_stream: torch.cuda.Stream = torch.cuda.Stream()  #type: ignore
    comm_stream: torch.cuda.Stream = torch.cuda.Stream()  # type: ignore

    ubatch_contexts = make_ubatch_contexts(num_ubatches, compute_stream,
                                           comm_stream, None)
    ubatch_inputs = [input_ids[:M // 2], input_ids[M // 2:]]
    ubatch_outputs = [outputs[:M // 2], outputs[M // 2:]]

    threads = []
    for i, ubatch_context in enumerate(ubatch_contexts):
        thread = threading.Thread(target=ubatch_thread_fn,
                                  args=(
                                      ubatch_inputs[i],
                                      weight,
                                      ubatch_outputs[i],
                                      ubatch_context,
                                  ))
        thread.start()
        threads.append(thread)

    ubatch_contexts[0].cpu_wait_event.set()
    for thread in threads:
        thread.join()

    layer(input_ids, weight, test_outputs)
    max_diff = torch.max(torch.abs(test_outputs - outputs)).item()
    print(f"Max Diff: {max_diff}")
    rtol = 1e-5
    atol = 1e-5
    all_close = torch.allclose(test_outputs, outputs, rtol=rtol, atol=atol)
    assert all_close
