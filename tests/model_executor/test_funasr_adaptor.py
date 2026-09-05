# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for FunASR adaptor padding isolation."""

import pytest
import torch

from vllm.model_executor.models.funasr import Transformer
from vllm.platforms import current_platform


def _seed_weights(module: torch.nn.Module) -> None:
    """Initialize weights for a forward pass without a checkpoint."""
    for name, param in module.named_parameters():
        if "norm" in name:
            continue
        if param.dim() >= 2:
            torch.nn.init.xavier_uniform_(param)
        else:
            torch.nn.init.zeros_(param)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="MMEncoderAttention requires a CUDA-like device",
)
@pytest.mark.parametrize("seed", range(100, 105))
def test_adaptor_batch_invariant(dist_init, seed: int) -> None:
    """A short sample's output must not depend on its batch mates."""
    torch.manual_seed(seed)
    short_len, long_len, dim = 5, 40, 32
    device = current_platform.device_type

    adaptor = Transformer(
        downsample_rate=1,
        encoder_dim=dim,
        llm_dim=dim,
        ffn_dim=64,
        n_layer=2,
        attention_heads=4,
    ).eval()
    adaptor = adaptor.to(device)
    _seed_weights(adaptor)

    short_features = torch.randn(short_len, dim, device=device)
    padding = torch.randn(long_len - short_len, dim, device=device) * 7 + 3
    padded_short = torch.cat([short_features, padding])
    batched = torch.stack([padded_short, torch.randn(long_len, dim, device=device)])
    ilens_batched = torch.tensor(
        [short_len, long_len], dtype=torch.int32, device=device
    )

    alone = short_features.unsqueeze(0)
    ilens_alone = torch.tensor([short_len], dtype=torch.int32, device=device)

    with torch.no_grad():
        out_batched, olens_batched = adaptor(batched, ilens_batched)
        out_alone, olens_alone = adaptor(alone, ilens_alone)

    assert olens_batched[0].item() == short_len
    assert olens_alone[0].item() == short_len
    torch.testing.assert_close(
        out_batched[0, :short_len],
        out_alone[0, :short_len],
        atol=1e-3,
        rtol=1e-3,
    )
