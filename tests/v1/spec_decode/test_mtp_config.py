# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import logging
from pathlib import Path

import pytest
from transformers import Qwen3_5TextConfig, Step3p7TextConfig

from vllm.config import ModelConfig, ParallelConfig, SpeculativeConfig

_MTP_CONFIG_LOGGER = "vllm.v1.worker.gpu.spec_decode.mtp.config"

# model: (config class, MTP layer count key, target architecture)
_MTP_TARGETS = {
    "qwen3_5": (Qwen3_5TextConfig, "mtp_num_hidden_layers", "Qwen3_5ForCausalLM"),
    "step3p5": (Step3p7TextConfig, "num_nextn_predict_layers", "Step3p5ForCausalLM"),
}


def _save_target(path: Path, model: str, num_mtp_modules: int) -> str:
    # Config only: no weights are loaded, so default model sizes are fine.
    config_cls, layers_key, arch = _MTP_TARGETS[model]
    config_cls(**{layers_key: num_mtp_modules}, architectures=[arch]).save_pretrained(
        path
    )
    return str(path)


def _mtp_spec(target: str, **kwargs) -> SpeculativeConfig:
    return SpeculativeConfig(
        method="mtp",
        target_model_config=ModelConfig(
            model=target, tokenizer_mode="skip", max_model_len=2048
        ),
        target_parallel_config=ParallelConfig(),
        **kwargs,
    )


def _mtp_config_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == _MTP_CONFIG_LOGGER and record.levelno >= logging.WARNING
    ]


@pytest.mark.parametrize(
    ("model", "num_mtp_modules", "num_speculative_tokens", "reused"),
    [
        ("qwen3_5", 1, 1, False),
        ("qwen3_5", 1, 2, True),
        # Multi-module draft whose model type was not special-cased.
        ("qwen3_5", 2, 2, False),
        ("step3p5", 3, 3, False),
        # Special-cased model type whose modules are cycled twice.
        ("step3p5", 3, 6, True),
    ],
)
def test_mtp_module_reuse_warning_follows_module_count(
    tmp_path: Path,
    caplog_vllm: pytest.LogCaptureFixture,
    model: str,
    num_mtp_modules: int,
    num_speculative_tokens: int,
    reused: bool,
):
    target = _save_target(tmp_path, model, num_mtp_modules)
    spec = _mtp_spec(target, num_speculative_tokens=num_speculative_tokens)
    assert spec.num_mtp_modules() == num_mtp_modules
    assert bool(_mtp_config_warnings(caplog_vllm)) == reused


@pytest.mark.parametrize("num_mtp_modules", [1, 3])
def test_mtp_num_speculative_tokens_defaults_to_module_count(
    tmp_path: Path, caplog_vllm: pytest.LogCaptureFixture, num_mtp_modules: int
):
    # With an explicit draft model, num_speculative_tokens may be omitted and
    # defaults to the draft's n_predict.
    target = _save_target(tmp_path, "step3p5", num_mtp_modules)
    spec = _mtp_spec(target, model=target)
    assert spec.num_speculative_tokens == num_mtp_modules
    assert spec.use_multi_module_mtp() == (num_mtp_modules > 1)
    assert not _mtp_config_warnings(caplog_vllm)
