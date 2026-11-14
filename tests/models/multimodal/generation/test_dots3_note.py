# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform

# The dots3_note vision tower is an NVIDIA-only implementation
# (``vllm/models/dots3_note/nvidia``), so importing it is only safe on CUDA.
pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="dots3_note vision tower is CUDA only"
)


def _tiny_config():
    """Smallest config that still exercises the full vision path.

    Leaving ``pyramid_num_routed`` unset selects the dense ``DotsSwiGLUFFN``
    FFN, which keeps MoE routing and the FP8 MoE kernel out of the test.
    """
    from vllm.models.dots3_note.nvidia.vision import DotsMoEVitConfig

    return DotsMoEVitConfig(
        embed_dim=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        adapter_in_dim=128,  # must match embed_dim
        adapter_out_dim=128,
        attn_implementation="eager",
        enable_torch_compile=False,
    )


@pytest.mark.core_model
@torch.inference_mode()
def test_dots3_note_vision_without_host_sync(dist_init):
    """The vision tower must run without blocking the host, including the mask.

    The vision tower is replicated across TP ranks (it uses plain ``nn.Linear``,
    not ``QKVParallelLinear``), so the copies covered here behave the same at
    any TP size and a single process is enough to exercise them.
    """
    # Imported here so collection on non-CUDA platforms stays safe.
    from vllm import envs
    from vllm.models.dots3_note.nvidia.vision import DotsMoEVitModel
    from vllm.utils.gpu_sync_debug import (
        enable_gpu_sync_check,
        with_gpu_sync_check,
    )
    from vllm.utils.torch_utils import set_default_torch_dtype

    # ``_SYNC_CHECK_MODE`` is captured when the module is imported, so the gate
    # must be armed from the environment (the CI image bakes it in) rather than
    # toggled here.
    if envs.VLLM_GPU_SYNC_CHECK is None:
        pytest.skip(
            "needs VLLM_GPU_SYNC_CHECK=error (set by the CI image); "
            "export it and re-run to exercise this test locally"
        )

    torch.manual_seed(0)
    grid_thw = [[2, 4, 6], [1, 6, 8]]
    config = _tiny_config()

    # bf16 matches production, but the property under test (host sync) is
    # dtype independent, so older cards can fall back to fp32.
    use_bf16 = torch.cuda.get_device_capability()[0] >= 8
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    with set_default_torch_dtype(dtype):
        model = DotsMoEVitModel(config).cuda().eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(mean=0.0, std=0.02)

    grid_thw_cpu = torch.tensor(grid_thw, dtype=torch.long)

    # Warm up before arming the gate, mirroring what gpu_worker does.
    num_patches = sum(t * h * w for t, h, w in grid_thw)
    pixels = torch.randn(
        num_patches, 3 * config.patch_size**2, device="cuda", dtype=model.dtype
    )
    meta = model.prepare_meta(grid_thw_cpu)
    model(pixels, meta)

    enable_gpu_sync_check()

    def _run():
        # Two segments, so the eager path really builds a block-diagonal mask.
        meta = model.prepare_meta(grid_thw_cpu)
        return model(pixels, meta)

    out = with_gpu_sync_check(_run)()

    merge_area = config.spatial_merge_size**2
    assert out.shape == (num_patches // merge_area, config.adapter_out_dim)
    assert out.isfinite().all()
