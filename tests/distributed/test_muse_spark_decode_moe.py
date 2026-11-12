# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark ROCm decode MoE under expert parallelism: the layer output with
and without it, all-reduced and on a token-sharded stream."""

import json

import pytest
import torch

from vllm.platforms import current_platform

from ..models.test_muse_spark_rocm_mono import TINY_TEXT, _checkpoint_config
from ..utils import multi_gpu_test
from .eplb_utils import distributed_run, set_env_vars_and_device

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="Muse Spark ROCm model path"
)

# One MoE layer whose experts take the FlyDSL MXFP4 kernels.
TEXT = TINY_TEXT | {
    "hidden_size": 1024,
    "moe_hidden_size": 1024,
    "moe_intermediate_size": 1024,
    "num_local_experts": 16,
}


def _vllm_config(checkpoint: str, world_size: int):
    from vllm.config import ModelConfig, ParallelConfig, VllmConfig

    return VllmConfig(
        model_config=ModelConfig(
            model=checkpoint,
            tokenizer_mode="skip",
            dtype="bfloat16",
            max_model_len=4096,
            language_model_only=True,
            enforce_eager=True,
        ),
        parallel_config=ParallelConfig(
            tensor_parallel_size=world_size,
            enable_expert_parallel=True,
            distributed_executor_backend="mp",
        ),
    )


def _moe_layer(vllm_config):
    from vllm.config.load import LoadConfig
    from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader
    from vllm.model_executor.model_loader.utils import process_weights_after_loading
    from vllm.models.muse_spark.amd.model import MuseSparkMoE
    from vllm.models.muse_spark.common.layers import (
        materialize_residual_path_constants,
    )
    from vllm.utils.torch_utils import set_default_torch_dtype

    device = torch.device("cuda", torch.accelerator.current_device_index())
    with set_default_torch_dtype(torch.bfloat16), device:
        moe = MuseSparkMoE(vllm_config, layer_idx=1, prefix="layers.1.feed_forward")
    model_config = vllm_config.model_config
    DummyModelLoader(LoadConfig(load_format="dummy")).load_weights(moe, model_config)
    # Spread the router scores: the reference runs the router GEMM at another
    # token count, which must not flip near-tied expert choices.
    moe.router.weight.data.mul_(100)
    materialize_residual_path_constants(moe)
    process_weights_after_loading(moe, model_config, device)
    return moe


def decode_moe_matches_the_per_route_path(
    env: dict[str, str], world_size: int, checkpoint: str, sizes: list[int]
) -> None:
    set_env_vars_and_device(env)
    from vllm.config import set_current_vllm_config
    from vllm.distributed.parallel_state import ensure_model_parallel_initialized
    from vllm.forward_context import set_forward_context
    from vllm.models.muse_spark.amd.decode_moe import MAX_TOKENS
    from vllm.models.muse_spark.amd.model import TokenShard
    from vllm.v1.worker.workspace import init_workspace_manager

    config = _vllm_config(checkpoint, world_size)
    with set_current_vllm_config(config):
        ensure_model_parallel_initialized(world_size, 1)
        init_workspace_manager(torch.device("cuda", int(env["LOCAL_RANK"])))
        moe = _moe_layer(config)
    assert moe.decode_moe is not None
    assert moe.decode_moe.kernels(moe, MAX_TOKENS + 1) is None

    def run(x, shard=None):
        with set_forward_context(None, config):
            return moe(x, shard)

    generator = torch.Generator(device="cuda").manual_seed(0)
    outputs = []
    with torch.inference_mode():
        for num_tokens in sizes:
            assert moe.decode_moe.kernels(moe, num_tokens) is not None
            x = torch.randn(
                MAX_TOKENS + 1, TEXT["hidden_size"], device="cuda", generator=generator
            ).to(torch.bfloat16)
            # A call over MAX_TOKENS runs the per-route path; each token's MoE
            # output depends only on its own row.
            expected = run(x)[:num_tokens]
            x = x[:num_tokens]
            shard = TokenShard(num_tokens)
            outputs.append((run(x), expected))
            outputs.append((run(shard.local(x), shard), shard.local(expected)))
    # After the collectives: a rank that fails does not leave the other waiting.
    # The reference's expert kernels run another tile config, so the MXFP4
    # requantization of a row's stage-1 output can round a block differently.
    for output, expected in outputs:
        error = (output.float() - expected.float()).norm()
        assert error <= 1e-2 * expected.float().norm()


@multi_gpu_test(num_gpus=2)
def test_muse_spark_decode_moe_matches_the_per_route_path(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All-reduced and token-sharded (reduce-scattered) MoE outputs of the decode
    MoE equal the per-route path's."""
    from vllm.platforms.rocm import on_gfx950

    if not on_gfx950():
        pytest.skip("the decode MoE runs on gfx950")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    (tmp_path / "config.json").write_text(json.dumps(_checkpoint_config(TEXT)))
    distributed_run(
        decode_moe_matches_the_per_route_path,
        2,
        str(tmp_path),
        [1, 8, 37, 512, 513, 2048],
    )
