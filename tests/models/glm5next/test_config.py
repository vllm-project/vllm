# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash config options this implementation does not support.

`Glm5NextTextConfig` accepts these, so vLLM has to reject them itself rather
than serve a checkpoint it would silently get wrong.
"""

import pytest
from transformers import Glm5NextTextConfig

from vllm.models.glm5next.common.model import _validate_supported_config


def test_rejects_dropping_the_incomplete_kpool_tail():
    config = Glm5NextTextConfig(
        num_hidden_layers=2, index_topk=2048, index_kpool_always_select_tail=False
    )

    with pytest.raises(NotImplementedError, match="index_kpool_always_select_tail"):
        _validate_supported_config(config)
