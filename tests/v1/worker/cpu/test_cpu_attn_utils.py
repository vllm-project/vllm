# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch

from vllm.config import VllmConfig
from vllm.platforms import CpuArchEnum, current_platform
from vllm.triton_utils import HAS_TRITON
from vllm.v1.worker.cpu import attn_utils
from vllm.v1.worker.cpu.attn_utils import FastPrefillHelper

pytestmark = pytest.mark.skipif(
    not current_platform.is_cpu()
    or HAS_TRITON
    or current_platform.get_cpu_architecture() == CpuArchEnum.X86,
    reason="requires a non-x86 CPU platform without Triton",
)


@pytest.mark.parametrize(
    ("layer_names", "shared_layers", "enabled", "expected"),
    [
        (["t0", "t1"], {}, True, set()),
        (
            ["t0", "t1", "t2", "t3"],
            {"t2": "t1", "t3": "t1"},
            True,
            {"t2", "t3"},
        ),
        (
            ["t0", "t1", "t2", "t3"],
            {"t1": "t0", "t3": "t0"},
            True,
            {"t3"},
        ),
        (["t0"], {"t0": "t0"}, False, set()),
    ],
)
def test_fast_prefill_selects_only_trailing_shared_layers(
    monkeypatch: pytest.MonkeyPatch,
    layer_names: list[str],
    shared_layers: dict[str, str],
    enabled: bool,
    expected: set[str],
) -> None:
    monkeypatch.setattr(
        attn_utils,
        "get_layers_from_vllm_config",
        lambda *args, **kwargs: dict.fromkeys(layer_names),
    )
    monkeypatch.setattr(
        attn_utils,
        "get_shared_kv_cache_layers",
        lambda config: shared_layers,
    )
    config = VllmConfig()
    config.cache_config.kv_sharing_fast_prefill = enabled

    assert attn_utils.get_kv_sharing_fast_prefill_eligible_layers(config) == expected


def test_fast_prefill_uses_unpadded_cpu_logits_indices() -> None:
    helper = FastPrefillHelper()
    logits_indices = torch.tensor([3, 9], dtype=torch.int32)
    cu_num_logits = np.array([0, 1, 2], dtype=np.int32)

    assert helper.prepare(logits_indices, cu_num_logits, has_prefill=False) is None

    metadata = helper.prepare(logits_indices, cu_num_logits, has_prefill=True)

    assert metadata is not None
    assert metadata.logits_indices_padded is logits_indices
    assert metadata.num_logits_indices == 2
    assert metadata.max_logits_per_req == 1
