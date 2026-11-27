# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The ROCm CSA layers reach the Step5 scored-prefill warmup."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="Step5 CSA is the ROCm path"
)


def _worker(context):
    return SimpleNamespace(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(model_type="step5")
            ),
            compilation_config=SimpleNamespace(static_forward_context=context),
        )
    )


def test_step5_warmup_dispatches_rocm_csa_topk():
    from vllm.model_executor.warmup.step5_dsa_warmup import step5_dsa_warmup
    from vllm.models.step5.amd import warmup
    from vllm.models.step5.amd.model import Step5Attention

    def layer(indexer):
        mock = MagicMock(indexer=indexer)
        mock.__class__ = Step5Attention
        return mock

    sparse = layer(SimpleNamespace(topk=512))
    dense = layer(None)
    worker = _worker({"swa": dense, "other": object(), "csa": sparse})

    with patch.object(warmup, "run_step5_scored_prefill_warmup") as run:
        step5_dsa_warmup(worker)
    run.assert_called_once_with(worker, 512)

    with patch.object(warmup, "run_step5_scored_prefill_warmup") as run:
        step5_dsa_warmup(_worker({"swa": dense}))
    run.assert_not_called()
