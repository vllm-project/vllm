# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash MonoKernel spec, model side, without a GPU."""

import pytest
import torch

from vllm.models.common.mono.spec import MonoContext, MonoStep
from vllm.models.glm5next.amd.mono import kernel, spec


def _ctx(tp: int) -> MonoContext:
    return MonoContext(None, tp, 0, None, torch.device("cpu"))


def _step(width: int) -> MonoStep:
    return MonoStep(width, width, torch.zeros(width, dtype=torch.long), {}, {})


@pytest.mark.parametrize("tp", spec.TP_SIZES)
@pytest.mark.parametrize("width", spec.WIDTHS)
def test_eligible_caps_width_per_tp(tp: int, width: int):
    cap = spec.TP_MAX_WIDTH.get(tp, max(spec.WIDTHS))
    reason = kernel.eligible(_ctx(tp), _step(width))
    assert (reason is None) == (width <= cap), (tp, width, reason)
