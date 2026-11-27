# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm only", allow_module_level=True)

from vllm.model_executor.layers.fusion.gemma4_ple_embed import gemma4_ple_embed_gather

VOCAB, PLE, LAYERS, SELF_LAYERS = 4096, 256, 42, 24


@pytest.mark.parametrize("tokens", [1, 17, 300, 2048])
@pytest.mark.parametrize("use_rows", [False, True])
def test_ple_embed_gather(tokens, use_rows):
    torch.manual_seed(0)
    table = torch.randn(VOCAB, LAYERS * PLE, device="cuda").bfloat16()
    scale = torch.tensor(PLE**0.5, device="cuda")
    ids = torch.randint(-3, VOCAB + 3, (tokens,), device="cuda", dtype=torch.int32)
    valid = (ids >= 0) & (ids < VOCAB)
    ref_all = (table[torch.where(valid, ids, 0).long()] * scale).reshape(
        tokens, LAYERS, PLE
    )

    out = torch.zeros(tokens, SELF_LAYERS, PLE, device="cuda").bfloat16()
    gemma4_ple_embed_gather(
        ids, None, table, scale, out.view(tokens, -1), 0, SELF_LAYERS * PLE, tokens
    )
    assert torch.equal(out, ref_all[:, :SELF_LAYERS])

    rows = torch.randint(0, tokens, (7,), device="cuda") if use_rows else None
    n = 7 if use_rows else tokens
    cross = torch.zeros(n, LAYERS - SELF_LAYERS, PLE, device="cuda").bfloat16()
    gemma4_ple_embed_gather(
        ids,
        rows,
        table,
        scale,
        cross.view(n, -1),
        SELF_LAYERS * PLE,
        (LAYERS - SELF_LAYERS) * PLE,
        n,
    )
    ref = ref_all if rows is None else ref_all[rows]
    assert torch.equal(cross, ref[:, SELF_LAYERS:])
