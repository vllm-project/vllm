# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock, patch

from vllm.v1.spec_decode.draft_model import DraftModelProposer


def test_draft_model_declares_the_root_it_loads_under():
    """The root given to the quant config must be the one the model loads under.

    Draft layer names carry a synthetic root that the draft checkpoint's
    quantization config knows nothing about. If the two drift apart, layer-name
    targets silently stop matching and the drafter loads unquantized.
    """
    proposer = DraftModelProposer.__new__(DraftModelProposer)
    quant_config = SimpleNamespace(model_root_prefix="")
    draft_vllm_config = Mock(quant_config=quant_config)

    with (
        patch.object(
            DraftModelProposer,
            "_create_draft_vllm_config",
            return_value=draft_vllm_config,
        ),
        patch("vllm.v1.spec_decode.draft_model.get_model") as mock_get_model,
    ):
        proposer._get_model()

    assert quant_config.model_root_prefix == mock_get_model.call_args.kwargs["prefix"]
