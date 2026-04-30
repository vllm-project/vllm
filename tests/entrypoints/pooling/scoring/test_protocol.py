# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from pydantic import ValidationError

from vllm.entrypoints.pooling.scoring.protocol import RerankRequest


class TestRerankRequestValidation:
    def test_top_n_defaults_to_zero(self):
        request = RerankRequest(query="capital of France", documents=["Paris"])

        assert request.top_n == 0

    @pytest.mark.parametrize("top_n", [0, 1, 2])
    def test_non_negative_top_n_is_accepted(self, top_n: int):
        request = RerankRequest(
            query="capital of France",
            documents=["Paris", "Berlin"],
            top_n=top_n,
        )

        assert request.top_n == top_n

    def test_negative_top_n_is_rejected(self):
        with pytest.raises(ValidationError):
            RerankRequest(
                query="capital of France",
                documents=["Paris", "Berlin"],
                top_n=-1,
            )
