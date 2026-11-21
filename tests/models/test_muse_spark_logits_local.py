# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch
from torch import nn

from vllm.models.muse_spark.amd.model import MuseSparkForCausalLM
from vllm.models.muse_spark.amd.model_state import MuseSparkModelState
from vllm.models.muse_spark.common.layers import (
    MuseSparkNormPreservingResidualConnection,
)
from vllm.sampling_params import SamplingParams


def test_compute_logits_local_applies_output_transform() -> None:
    model = MuseSparkForCausalLM.__new__(MuseSparkForCausalLM)
    nn.Module.__init__(model)
    model.model = nn.Module()
    model.model.mono = None
    model.lm_head = nn.Linear(4, 8, bias=False, dtype=torch.bfloat16)
    model.logits_processor = lambda lm_head, h, skip_gather=False: lm_head(h)
    model.metap_output_multiplier = 0.5
    model.output_soft_cap_temp = 20.0
    model.mul_tanh_soft_cap = None
    hidden = torch.randn(3, 4, dtype=torch.bfloat16) * 50

    torch.testing.assert_close(
        model.compute_logits_local(hidden), model.compute_logits(hidden)
    )


def test_residual_coefficients_refilled_in_place() -> None:
    """Every load rebuilds the gated-residual coefficients in place."""
    connection = MuseSparkNormPreservingResidualConnection(4)
    buffers = []
    for gate in (0.1, 0.5):  # cold start, reload
        with torch.no_grad():
            connection.gate.fill_(gate)
        connection.materialize_coefficients()
        buffers.append((connection.alpha, connection.beta))
        alpha, beta = connection.coefficients(connection.gate / connection.temperature)
        assert torch.equal(connection.alpha, alpha)
        assert torch.equal(connection.beta, beta)
    assert buffers[0][0] is buffers[1][0] and buffers[0][1] is buffers[1][1]


def test_model_state_passes_every_row_while_a_request_reads_them() -> None:
    """Token-sharded steps gather only the sampled rows, except while a request
    in the batch asks for prompt logprobs."""
    state = MuseSparkModelState.__new__(MuseSparkModelState)
    state.mm_prefix_ranges = state.rope_state = state.prompt_embeds_state = None
    state.min_sharded_tokens = 8
    state.logits_indices = torch.zeros(16, dtype=torch.int64)
    state.all_rows_req_ids = set()
    batch = SimpleNamespace(
        num_tokens_after_padding=8,
        req_ids=["a", "b"],
        logits_indices=torch.tensor([3, 7]),
    )
    for req_id, params in (("a", SamplingParams()), ("b", None)):
        state.add_request(0, SimpleNamespace(req_id=req_id, sampling_params=params))

    def rows():
        return state.prepare_inputs(batch, None).get("logits_indices")

    assert torch.equal(rows(), batch.logits_indices)
    state.add_request(
        2,
        SimpleNamespace(req_id="c", sampling_params=SamplingParams(prompt_logprobs=0)),
    )
    assert torch.equal(rows(), batch.logits_indices)
    batch.req_ids.append("c")
    assert torch.equal(rows(), torch.arange(8))
    state.remove_request("c")
    assert torch.equal(rows(), batch.logits_indices)
    batch.num_tokens_after_padding = 7
    assert rows() is None
