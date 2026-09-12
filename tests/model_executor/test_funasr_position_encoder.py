# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.models.funasr import SinusoidalPositionEncoder


def test_position_encoding_fp16_matches_fp32_reference() -> None:
    """fp16 position encodings must stay close to the fp32 reference.

    Regression test for a bug where FunASR-Nano produced degenerate
    repeated-token transcriptions under `--dtype float16`: the encoder's
    log/exp/sin/cos math was run directly in fp16, whose limited precision
    on the small timescale terms produced phase errors up to ~0.4 (out of
    a [-1, 1] range) once multiplied by position indices.
    """
    encoder = SinusoidalPositionEncoder()
    positions = torch.arange(1, 801)[None, :]
    depth = 560

    encoding_fp32 = encoder.encode(positions, depth, dtype=torch.float32)
    encoding_fp16 = encoder.encode(positions, depth, dtype=torch.float16)

    torch.testing.assert_close(
        encoding_fp16.float(), encoding_fp32, atol=1e-2, rtol=1e-2
    )
