# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-slice Mamba2 in_proj: one linear per slice, same math as the fused in_proj."""

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


def test_each_slice_resolves_its_own_quant_method(bf16_cuda):
    config = RecordingConfig()
    mixer = make_mixer(config, split_in_proj=True)
    prefix = mixer.prefix
    assert config.prefixes == [
        *(f"{prefix}.{name}_proj" for name in MAMBA2_IN_PROJ_SLICES),
        f"{prefix}.out_proj",
    ]
    assert not hasattr(mixer, "in_proj")
    for name in MAMBA2_IN_PROJ_SLICES:
        assert getattr(mixer, f"{name}_proj").weight.shape == (SIZES[name], HIDDEN)


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
