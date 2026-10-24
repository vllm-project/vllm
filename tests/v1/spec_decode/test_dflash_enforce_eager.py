# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``--speculative-config.enforce_eager`` must reach the drafter (#61337).

The override was accepted by the CLI but never read: the draft
``ModelConfig`` unconditionally inherited the target's ``enforce_eager``, and
the DFlash speculator sized its CUDA-graph manager off the target runner's
mode. A drafter-only eager run therefore still captured draft graphs.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from transformers import PreTrainedConfig

from vllm.config.compilation import CUDAGraphMode
from vllm.config.parallel import ParallelConfig
from vllm.config.speculative import SpeculativeConfig
from vllm.v1.worker.gpu.spec_decode.dflash import speculator as dflash_speculator
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator


def _make_speculative_config(enforce_eager, target_enforce_eager):
    hf_config = SpeculativeConfig.hf_config_override(
        PreTrainedConfig(
            architectures=["LlamaForCausalLM"],
            model_type="deepseek_v4",
            num_hidden_layers=64,
            num_nextn_predict_layers=3,
            dspark_block_size=5,
        )
    )
    draft = MagicMock(
        hf_config=hf_config, architectures=hf_config.architectures, max_model_len=128
    )
    draft.registry.inspect_model_cls.return_value = (None, "DSparkDraftModel")
    target = MagicMock(
        max_model_len=128,
        quantization=None,
        hf_overrides={},
        enforce_eager=target_enforce_eager,
    )
    target.hf_config.model_type = "deepseek_v4"

    with patch(
        "vllm.config.speculative.ModelConfig", return_value=draft
    ) as model_config_cls:
        config = SpeculativeConfig(
            model="draft",
            method="dspark",
            enforce_eager=enforce_eager,
            target_model_config=target,
            target_parallel_config=ParallelConfig(),
        )
    return config, model_config_cls


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("override", "target_enforce_eager", "expected"),
    [
        (True, False, True),
        (False, True, False),
        (None, True, True),
        (None, False, False),
    ],
)
def test_draft_enforce_eager_override(override, target_enforce_eager, expected):
    _, model_config_cls = _make_speculative_config(override, target_enforce_eager)
    assert model_config_cls.call_args.kwargs["enforce_eager"] is expected


def _fake_dflash_self(draft_enforce_eager: bool) -> SimpleNamespace:
    return SimpleNamespace(
        draft_model_config=SimpleNamespace(enforce_eager=draft_enforce_eager),
        attn_cg_support=SimpleNamespace(
            min_cg_support=dflash_speculator.AttentionCGSupport.ALWAYS,
            min_cg_attn_backend="FLASH_ATTN",
        ),
        _speculator_name="DFlash",
        vllm_config=SimpleNamespace(),
        device="cpu",
        num_query_per_req=4,
    )


@pytest.mark.cpu_test
def test_dflash_draft_enforce_eager_disables_capture() -> None:
    fake_self = _fake_dflash_self(draft_enforce_eager=True)
    with patch.object(dflash_speculator, "DFlashCudaGraphManager") as manager_cls:
        DFlashSpeculator.init_cudagraph_manager(
            fake_self, CUDAGraphMode.FULL_AND_PIECEWISE
        )
    assert manager_cls.call_args.args[2] == CUDAGraphMode.NONE


@pytest.mark.cpu_test
def test_dflash_target_mode_still_reaches_capture_without_override() -> None:
    fake_self = _fake_dflash_self(draft_enforce_eager=False)
    with patch.object(dflash_speculator, "DFlashCudaGraphManager") as manager_cls:
        DFlashSpeculator.init_cudagraph_manager(
            fake_self, CUDAGraphMode.FULL_AND_PIECEWISE
        )
    assert manager_cls.call_args.args[2] == CUDAGraphMode.FULL_DECODE_ONLY
