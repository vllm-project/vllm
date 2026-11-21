# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 large-batch sequence parallelism: all-gathering token shards of
the attention input keeps it MXFP8. The gathered E4M3 data and 128x4-swizzled
UE8M0 scales must equal quantizing the whole batch, bitwise."""

import pytest
import ray
import torch

from vllm.platforms import current_platform

from ...utils import (
    ensure_model_parallel_initialized,
    init_test_distributed_environment,
    multi_process_parallel,
)

HIDDEN = 6144


@ray.remote(num_gpus=1, max_calls=1)
def _gather_matches_whole_batch(
    monkeypatch, tp_size, pp_size, rank, distributed_init_port
):
    from vllm.model_executor.layers.fusion.quant_activation import (
        QuantizedActivation,
    )
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kMxfp8Dynamic,
    )
    from vllm.models.minimax_m3.nvidia.model import _sp_all_gather_input

    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
        ensure_model_parallel_initialized(tp_size, pp_size)

        for shard_rows in (128, 512):
            num_tokens = shard_rows * tp_size
            gen = torch.Generator(device=device).manual_seed(num_tokens)
            x = torch.randn(num_tokens, HIDDEN, generator=gen, device=device)
            x = (x * 8).to(torch.bfloat16)
            shard = x[rank * shard_rows : (rank + 1) * shard_rows]

            # BF16 shards are gathered as they are.
            torch.testing.assert_close(
                _sp_all_gather_input(shard, num_tokens), x, rtol=0, atol=0
            )

            data, scale = mxfp8_e4m3_quantize(shard, is_sf_swizzled_layout=True)
            gathered = _sp_all_gather_input(
                QuantizedActivation(
                    data=data,
                    scale=scale,
                    orig_dtype=torch.bfloat16,
                    orig_shape=shard.shape,
                    quant_key=kMxfp8Dynamic,
                ),
                num_tokens,
            )
            want_data, want_scale = mxfp8_e4m3_quantize(x, is_sf_swizzled_layout=True)
            assert gathered.orig_shape == x.shape
            assert torch.equal(
                gathered.data.view(torch.uint8), want_data.view(torch.uint8)
            )
            assert torch.equal(
                gathered.scale.view(torch.uint8).flatten(),
                want_scale.view(torch.uint8).flatten(),
            )


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="MXFP8 quantization needs SM100-family GPUs",
)
@pytest.mark.parametrize("tp_size", [2, 4])
def test_sp_all_gather_input_mxfp8(monkeypatch: pytest.MonkeyPatch, tp_size: int):
    if torch.accelerator.device_count() < tp_size:
        pytest.skip(f"Need at least {tp_size} GPUs.")
    multi_process_parallel(monkeypatch, tp_size, 1, _gather_matches_whole_batch)
