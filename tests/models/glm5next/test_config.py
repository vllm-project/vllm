# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash config options this implementation does not support.

`Glm5NextTextConfig` accepts these, so vLLM has to reject them itself rather
than serve a checkpoint it would silently get wrong.
"""

import pytest
from transformers import Glm5NextTextConfig

from vllm.models.glm5next.common.model import _validate_supported_config


def _config(**kwargs) -> Glm5NextTextConfig:
    return Glm5NextTextConfig(num_hidden_layers=2, **kwargs)


@pytest.mark.parametrize(
    ("kwargs", "option"),
    [
        (
            {"index_topk": 2048, "index_dsa_use_layernorm": False},
            "index_dsa_use_layernorm",
        ),
        ({"index_topk": 2048, "index_kpool_compress": False}, "index_kpool_compress"),
        (
            {"index_topk": 2048, "index_kpool_always_select_tail": False},
            "index_kpool_always_select_tail",
        ),
        ({"mhc": True, "hres_vwnstyle": False}, "hres_vwnstyle"),
        ({"mhc": True, "mhc_no_norm_weight": True}, "mhc_no_norm_weight"),
    ],
)
def test_rejects_unimplemented_config_options(kwargs, option):
    with pytest.raises(NotImplementedError, match=option):
        _validate_supported_config(_config(**kwargs))


def test_accepts_the_shipped_checkpoint_options():
    """The values GLM-5.3-Flash actually ships must pass."""
    _validate_supported_config(
        _config(index_topk=2048, index_kpool_compress=True, mhc=True)
    )
