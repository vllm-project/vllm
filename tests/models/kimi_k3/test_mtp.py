# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.models.kimi_k3.amd.mtp import (
    _get_draft_text_config as get_amd_draft_text_config,
)
from vllm.models.kimi_k3.nvidia.mtp import (
    _get_draft_text_config as get_nvidia_draft_text_config,
)


@pytest.mark.parametrize(
    "get_draft_text_config",
    [get_amd_draft_text_config, get_nvidia_draft_text_config],
)
def test_kimi_k3_mtp_uses_draft_text_config(get_draft_text_config) -> None:
    target_text_config = SimpleNamespace(num_nextn_predict_layers=0)
    draft_text_config = SimpleNamespace(num_nextn_predict_layers=5)
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=target_text_config),
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_text_config=draft_text_config)
        ),
    )

    # Before the fix, Kimi-K3 reads the target config and profile-time MTP
    # step selection divides by zero.
    with pytest.raises(ZeroDivisionError):
        0 % vllm_config.model_config.hf_text_config.num_nextn_predict_layers

    selected_config = get_draft_text_config(vllm_config)
    assert selected_config is draft_text_config
    assert 0 % selected_config.num_nextn_predict_layers == 0
