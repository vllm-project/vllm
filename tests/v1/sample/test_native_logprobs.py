# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform


@pytest.mark.parametrize("vocab", [49, 2048, 2049, 129280])
@pytest.mark.parametrize("strided", [False, True])
def test_native_logprobs_row_identity_gradient_and_replay(vocab, strided):
    """Native LP rows retain decode/teacher identity and the native VJP."""
    if not current_platform.is_cuda():
        pytest.skip("Native CUDA log probabilities")
    from torch._subclasses.fake_tensor import FakeTensorMode

    from vllm.model_executor.kernels.batch_invariant import native_log_softmax

    generator = torch.Generator(device="cuda").manual_seed(951)
    storage = torch.randn(
        257,
        vocab * (2 if strided else 1),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    values = storage[:, ::2] if strided else storage
    values.requires_grad_()
    actual = native_log_softmax(values)
    for rows in (1, 4, 17):
        prefix = native_log_softmax(values[:rows].detach())
        assert torch.equal(actual[:rows], prefix)
    cpu_values = values.detach().cpu().double().requires_grad_()
    reference = torch.nn.functional.log_softmax(cpu_values, dim=-1)
    torch.testing.assert_close(
        actual, reference.float().to(values.device), atol=3e-6, rtol=1e-6
    )
    incoming = torch.randn(actual.shape, device="cuda", generator=generator)
    (gradient,) = torch.autograd.grad(actual, values, incoming)
    (expected_grad,) = torch.autograd.grad(
        reference, cpu_values, incoming.cpu().double()
    )
    torch.testing.assert_close(
        gradient, expected_grad.float().to(values.device), atol=5e-7, rtol=1e-5
    )
    with FakeTensorMode():
        sample = torch.empty((17, vocab), device="cuda", dtype=torch.float32)
        fake = native_log_softmax(sample)
        assert fake.shape == sample.shape and fake.dtype == sample.dtype
    compiled = torch.compile(native_log_softmax, backend="aot_eager", fullgraph=True)
    eager = native_log_softmax(values.detach())
    assert torch.equal(compiled(values.detach()), eager)

    replay_input = values.detach()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        native_log_softmax(replay_input)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = native_log_softmax(replay_input)
    replay_input.zero_()
    replayed.fill_(float("nan"))
    graph.replay()
    expected = native_log_softmax(replay_input[:1]).expand_as(replayed)
    assert torch.equal(replayed, expected)
