# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.config.mamba import (
    MambaBackendEnum,
    MambaConfig,
    MambaDecodeBackendEnum,
    MambaPrefillBackendEnum,
)
from vllm.engine.arg_utils import EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser


def test_mamba_prefill_backend_defaults_to_triton_independently():
    config = MambaConfig(decode_backend=MambaDecodeBackendEnum.FLASHINFER)
    assert config.decode_backend == MambaDecodeBackendEnum.FLASHINFER
    assert config.prefill_backend == MambaPrefillBackendEnum.TRITON


def test_mamba_prefill_backend_parses_explicit_flashinfer():
    config = MambaConfig(prefill_backend="flashinfer")
    assert config.decode_backend == MambaDecodeBackendEnum.TRITON
    assert config.prefill_backend == MambaPrefillBackendEnum.FLASHINFER


def test_mamba_prefill_backend_rejects_unknown_value():
    with pytest.raises(ValueError, match="Unknown Mamba prefill backend"):
        MambaConfig(prefill_backend="not-a-backend")


def test_legacy_mamba_backend_config_alias():
    config = MambaConfig(**{"backend": "flashinfer"})

    assert MambaBackendEnum is MambaDecodeBackendEnum
    assert config.decode_backend == MambaDecodeBackendEnum.FLASHINFER
    assert config.backend == MambaDecodeBackendEnum.FLASHINFER


@pytest.mark.parametrize(
    ("decode_backend", "expected_decode_backend"),
    [
        (None, MambaDecodeBackendEnum.FLASHINFER),
        (MambaDecodeBackendEnum.TRITON, MambaDecodeBackendEnum.TRITON),
    ],
)
def test_engine_args_mamba_decode_backend_precedence(
    decode_backend: MambaDecodeBackendEnum | None,
    expected_decode_backend: MambaDecodeBackendEnum,
):
    config = EngineArgs(
        mamba_config=MambaConfig(
            decode_backend=MambaDecodeBackendEnum.FLASHINFER,
            prefill_backend=MambaPrefillBackendEnum.FLASHINFER,
        ),
        mamba_decode_backend=decode_backend,
    ).create_engine_config()

    assert config.mamba_config.decode_backend == expected_decode_backend
    assert config.mamba_config.prefill_backend == MambaPrefillBackendEnum.FLASHINFER


def test_mamba_backend_cli_options_are_unset_when_omitted():
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = parser.parse_args(["--model", "dummy"])

    assert args.mamba_decode_backend is None
    assert args.mamba_backend is None
    assert args.mamba_prefill_backend is None


def test_mamba_prefill_cli_is_separate_from_decode_backend():
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = parser.parse_args(
        [
            "--model",
            "dummy",
            "--mamba-decode-backend",
            "flashinfer",
            "--mamba-prefill-backend",
            "triton",
        ]
    )
    assert args.mamba_decode_backend == "flashinfer"
    assert args.mamba_backend is None
    assert args.mamba_prefill_backend == "triton"


def test_legacy_mamba_backend_cli_alias():
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = parser.parse_args(["--mamba-backend", "flashinfer"])
    engine_args = EngineArgs.from_cli_args(args)

    assert engine_args.mamba_decode_backend == "flashinfer"


def test_mamba_decode_backend_aliases_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        EngineArgs(
            mamba_decode_backend=MambaDecodeBackendEnum.TRITON,
            mamba_backend=MambaDecodeBackendEnum.FLASHINFER,
        )
