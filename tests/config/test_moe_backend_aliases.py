# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The single-variant MXFP4 MoE backend aliases are accepted on the CLI."""

import pytest

from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    Mxfp4MoeBackend,
    map_mxfp4_backend,
)
from vllm.utils.argparse_utils import FlexibleArgumentParser

ALIASES = {
    "aiter_mxfp4_fp8": Mxfp4MoeBackend.AITER_MXFP4_FP8,
    "aiter_mxfp4_mxfp4": Mxfp4MoeBackend.AITER_MXFP4_MXFP4,
    "flashinfer_trtllm_afp8": Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8,
    "flashinfer_cutlass_afp8": Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_MXFP8,
}


@pytest.mark.parametrize("alias", list(ALIASES))
def test_alias_parses_from_cli(alias: str):
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = parser.parse_args(["--moe-backend", alias.replace("_", "-")])
    assert EngineArgs.from_cli_args(args).moe_backend == alias
    assert map_mxfp4_backend(alias) == [ALIASES[alias]]
