# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.models.exaone import ExaoneForCausalLM, ExaoneModel


def test_embed_input_ids_uses_transformer() -> None:
    transformer = ExaoneModel.__new__(ExaoneModel)
    torch.nn.Module.__init__(transformer)
    transformer.wte = torch.nn.Embedding(32, 16)

    model = ExaoneForCausalLM.__new__(ExaoneForCausalLM)
    torch.nn.Module.__init__(model)
    model.transformer = transformer

    input_ids = torch.tensor([1, 2, 3, 4])
    torch.testing.assert_close(
        model.embed_input_ids(input_ids),
        transformer.embed_input_ids(input_ids),
    )
