# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mamba2 in_proj slices grouped by weight format: per-slice tensors loaded into the
groups, adjacent or not, give the same math as the fused in_proj."""

import itertools

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.mamba.mamba_mixer2 import (
    MAMBA2_IN_PROJ_SLICES,
    MambaMixer2,
    mamba2_in_proj_group_name,
    mamba2_in_proj_stacked_mapping,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptMixedPrecisionConfig,
    ModelOptNvFp4Config,
)
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper
from vllm.platforms import current_platform

HIDDEN, HEADS, HEAD_DIM, GROUPS, STATE = 64, 4, 16, 2, 8
SIZES = dict(zip(MAMBA2_IN_PROJ_SLICES, (64, 64, 16, 16, 4)))
LAYER_IDS = itertools.count()  # each mixer registers a unique layer name
# Slices grouped by format, in the order mamba2_in_proj_groups returns them.
GROUPINGS = [
    [(name,) for name in MAMBA2_IN_PROJ_SLICES],
    [("z",), ("x",), ("B", "C", "dt")],
    [("z", "x"), ("B", "C", "dt")],
    [("z", "B", "C", "dt"), ("x",)],
    [("z", "C"), ("x",), ("B", "dt")],
    [MAMBA2_IN_PROJ_SLICES],
]

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


class FormatConfig(RecordingConfig):
    """Gives each slice of a group the same format, and each group its own; other
    layer names have no format."""

    def __init__(self, groups):
        super().__init__()
        self.formats = {name: i for i, group in enumerate(groups) for name in group}

    def get_linear_format(self, prefix):
        return self.formats.get(prefix.rsplit(".", 1)[-1].removesuffix("_proj"))


def make_mixer(quant_config=None, prefix=None, **kwargs):
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
        prefix=prefix or f"model.layers.{next(LAYER_IDS)}.mixer",
        **kwargs,
    )


def fill(mixer):
    generator = torch.Generator(device="cuda").manual_seed(0)
    for parameter in mixer.parameters():
        parameter.data.copy_(
            torch.randn(parameter.shape, generator=generator, device="cuda") * 0.1
        )


@pytest.mark.parametrize("groups", GROUPINGS, ids=str)
def test_each_group_resolves_its_own_quant_method(bf16_cuda, groups):
    config = FormatConfig(groups)
    mixer = make_mixer(config)
    assert mixer.in_proj_groups == groups
    names = [mamba2_in_proj_group_name(group) for group in groups]
    assert config.prefixes == [
        *(f"{mixer.prefix}.{name}" for name in names),
        f"{mixer.prefix}.out_proj",
    ]
    for name, group in zip(names, groups):
        rows = sum(SIZES[slice_name] for slice_name in group)
        assert getattr(mixer, name).weight.shape == (rows, HIDDEN)


@pytest.mark.parametrize("config", [None, RecordingConfig()], ids=["none", "plain"])
def test_default_layout_is_one_fused_in_proj(bf16_cuda, config):
    mixer = make_mixer(config)
    assert mixer.in_proj_groups == [MAMBA2_IN_PROJ_SLICES]
    if config is not None:
        assert config.prefixes == [
            f"{mixer.prefix}.in_proj",
            f"{mixer.prefix}.out_proj",
        ]
        assert config.packed_modules_mapping == {}
    assert mixer.in_proj.weight.shape == (sum(SIZES.values()), HIDDEN)


@pytest.mark.parametrize("groups", GROUPINGS, ids=str)
def test_per_slice_tensors_load_into_groups(bf16_cuda, groups):
    fused = make_mixer()
    fill(fused)
    model = nn.Module()
    model.layer = nn.Module()
    model.layer.mixer = make_mixer(FormatConfig(groups))
    rows = fused.in_proj.weight.split(list(SIZES.values()))
    weights = [
        (f"layer.mixer.{name}_proj.weight", weight.clone())
        for name, weight in zip(MAMBA2_IN_PROJ_SLICES, rows)
    ]
    mapper = WeightsMapper(orig_to_new_stacked=mamba2_in_proj_stacked_mapping(groups))
    AutoWeightsLoader(model).load_weights(weights, mapper=mapper)
    x = torch.randn(3, HIDDEN)
    torch.testing.assert_close(model.layer.mixer._project_in(x), fused._project_in(x))


@pytest.mark.parametrize(
    "layers, groups",
    [
        (("in_proj",), [MAMBA2_IN_PROJ_SLICES]),  # a fused checkpoint
        ([f"{name}_proj" for name in MAMBA2_IN_PROJ_SLICES], [MAMBA2_IN_PROJ_SLICES]),
        (("z_proj", "B_proj", "C_proj", "dt_proj"), [("z", "B", "C", "dt"), ("x",)]),
    ],
    ids=["fused", "per-slice", "x-unquantized"],
)
def test_modelopt_mixed_precision_formats_per_slice(bf16_cuda, layers, groups):
    prefix = f"model.layers.{next(LAYER_IDS)}.mixer"
    nvfp4 = {"quant_algo": "W4A16_NVFP4", "group_size": 16}
    config = ModelOptMixedPrecisionConfig.from_config(
        {
            "quantization": {
                "quant_algo": "MIXED_PRECISION",
                "kv_cache_quant_algo": None,
                "exclude_modules": [],
                "group_size": 16,
                "quantized_layers": {f"{prefix}.{name}": nvfp4 for name in layers},
            }
        }
    )
    mixer = make_mixer(config, prefix=prefix)
    assert mixer.in_proj_groups == groups
    # Every quantized group gets the NVFP4 method, fused or per-slice.
    for group in groups:
        method = getattr(mixer, mamba2_in_proj_group_name(group)).quant_method
        assert isinstance(method, UnquantizedLinearMethod) == (group == ("x",))


def test_modelopt_wildcard_exclusions_cover_fused_groups(bf16_cuda):
    config = ModelOptNvFp4Config(
        is_checkpoint_nvfp4_serialized=True, exclude_modules=["*.z_proj", "*.x_proj"]
    )
    mixer = make_mixer(config)
    assert mixer.in_proj_groups == [("z", "x"), ("B", "C", "dt")]
    assert isinstance(mixer.z_x_proj.quant_method, UnquantizedLinearMethod)
    assert not isinstance(mixer.B_C_dt_proj.quant_method, UnquantizedLinearMethod)
