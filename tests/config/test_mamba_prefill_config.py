# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.config.mamba import (
    MambaBackendEnum,
    MambaConfig,
    MambaPrefillBackendEnum,
)
from vllm.engine.arg_utils import EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser


def test_mamba_prefill_backend_defaults_to_triton_independently():
    config = MambaConfig(backend=MambaBackendEnum.FLASHINFER)
    assert config.backend == MambaBackendEnum.FLASHINFER
    assert config.prefill_backend == MambaPrefillBackendEnum.TRITON


def test_mamba_prefill_backend_parses_explicit_flashinfer():
    config = MambaConfig(prefill_backend="flashinfer")
    assert config.backend == MambaBackendEnum.TRITON
    assert config.prefill_backend == MambaPrefillBackendEnum.FLASHINFER


def test_mamba_prefill_backend_rejects_unknown_value():
    with pytest.raises(ValueError, match="Unknown Mamba prefill backend"):
        MambaConfig(prefill_backend="not-a-backend")


def test_mamba_prefill_cli_is_separate_from_decode_backend():
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = parser.parse_args(
        [
            "--model",
            "dummy",
            "--mamba-backend",
            "flashinfer",
            "--mamba-prefill-backend",
            "triton",
        ]
    )
    assert args.mamba_backend == "flashinfer"
    assert args.mamba_prefill_backend == "triton"
