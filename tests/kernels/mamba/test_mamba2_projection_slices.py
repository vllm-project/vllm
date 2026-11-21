# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-slice and low-rank Mamba2 projections give the same math as one in_proj."""

import itertools

import pytest
import torch

from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.mamba.mamba_mixer2 import (
    MAMBA2_IN_PROJ_SLICES,
    MambaMixer2,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.platforms import current_platform

HIDDEN, HEADS, HEAD_DIM, GROUPS, STATE = 64, 4, 16, 2, 8
SIZES = dict(zip(MAMBA2_IN_PROJ_SLICES, (64, 64, 16, 16, 4)))
LAYER_IDS = itertools.count()  # each mixer registers a unique layer name

pytestmark = pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA only")


@pytest.fixture
def bf16_cuda(default_vllm_config, dist_init):
    dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("cuda"):
        yield
    torch.set_default_dtype(dtype)


class RecordingConfig(QuantizationConfig):
    """Unquantized linears; records each prefix the mixer asks a method for."""

    def __init__(self):
        super().__init__()
        self.prefixes = []

    def get_name(self):
        return "recording"

    def get_supported_act_dtypes(self):
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls):
        return 0

    @staticmethod
    def get_config_filenames():
        return []

    @classmethod
    def from_config(cls, config):
        return cls()

    def get_quant_method(self, layer, prefix):
        self.prefixes.append(prefix)
        return UnquantizedLinearMethod()


def make_mixer(quant_config=None, **kwargs):
    return MambaMixer2(
        hidden_size=HIDDEN,
        ssm_state_size=STATE,
        conv_kernel_size=4,
        intermediate_size=HEADS * HEAD_DIM,
        use_conv_bias=True,
        use_bias=False,
        n_groups=GROUPS,
        num_heads=HEADS,
        head_dim=HEAD_DIM,
        quant_config=quant_config,
        prefix=f"model.layers.{next(LAYER_IDS)}.mixer",
        **kwargs,
    )


def fill(mixer):
    generator = torch.Generator(device="cuda").manual_seed(0)
    for parameter in mixer.parameters():
        parameter.data.copy_(
            torch.randn(parameter.shape, generator=generator, device="cuda") * 0.1
        )


@pytest.mark.parametrize(
    "low_rank", [{}, {"x": 8}, {"z": 8, "dt": 2, "out": 16}], ids=str
)
def test_slices_match_the_fused_projection(bf16_cuda, low_rank):
    config = RecordingConfig()
    mixer = make_mixer(config, split_in_proj=True, low_rank=low_rank)
    fill(mixer)
    prefix = mixer.prefix
    names = [
        f"{name}_a_proj" if name in low_rank else f"{name}_proj"
        for name in MAMBA2_IN_PROJ_SLICES
    ]
    # Every slice and factor resolves its own quantization method by name.
    expected = []
    for name in MAMBA2_IN_PROJ_SLICES:
        expected += (
            [f"{prefix}.{name}_a_proj", f"{prefix}.{name}_b_proj"]
            if name in low_rank
            else [f"{prefix}.{name}_proj"]
        )
    expected += (
        [f"{prefix}.out_a_proj", f"{prefix}.out_b_proj"]
        if "out" in low_rank
        else [f"{prefix}.out_proj"]
    )
    assert config.prefixes == expected
    assert not hasattr(mixer, "in_proj")
    assert all(hasattr(mixer, name) for name in names)

    # Reference: one dense in_proj weight with the slices stacked in row order.
    def weight(name):
        if name in low_rank:
            return (
                getattr(mixer, f"{name}_b_proj").weight.float()
                @ getattr(mixer, f"{name}_a_proj").weight.float()
            )
        return getattr(mixer, f"{name}_proj").weight.float()

    fused = torch.cat([weight(name) for name in MAMBA2_IN_PROJ_SLICES])
    assert fused.shape == (sum(SIZES.values()), HIDDEN)
    x = torch.randn(5, HIDDEN)
    torch.testing.assert_close(
        mixer._project_in(x).float(), x.float() @ fused.T, rtol=0.03, atol=0.03
    )
    h = torch.randn(5, HEADS * HEAD_DIM)
    out = (
        mixer.out_b_proj.weight.float() @ mixer.out_a_proj.weight.float()
        if "out" in low_rank
        else mixer.out_proj.weight.float()
    )
    torch.testing.assert_close(
        mixer._project_out(h).float(), h.float() @ out.T, rtol=0.03, atol=0.03
    )


def test_default_layout_is_one_fused_in_proj(bf16_cuda):
    config = RecordingConfig()
    mixer = make_mixer(config)
    assert config.prefixes == [f"{mixer.prefix}.in_proj", f"{mixer.prefix}.out_proj"]
    assert mixer.in_proj.weight.shape == (sum(SIZES.values()), HIDDEN)
    split = make_mixer(split_in_proj=True)
    fill(mixer)
    for name, start, size in zip(
        MAMBA2_IN_PROJ_SLICES,
        [0, 64, 128, 144, 160],
        SIZES.values(),
    ):
        getattr(split, f"{name}_proj").weight.data.copy_(
            mixer.in_proj.weight[start : start + size]
        )
    x = torch.randn(3, HIDDEN)
    torch.testing.assert_close(split._project_in(x), mixer._project_in(x))


def test_invalid_low_rank_is_rejected(bf16_cuda):
    for low_rank in ({"y": 4}, {"x": 0}, {"x": 2.0}):
        with pytest.raises(ValueError, match="low-rank"):
            make_mixer(low_rank=low_rank)
