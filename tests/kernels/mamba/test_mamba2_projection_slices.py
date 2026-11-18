# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mamba2 in_proj slices grouped by weight format, and low-rank slices: per-slice
tensors loaded into the groups, adjacent or not, give the same math as the fused
in_proj, and low-rank slices give the dense product of their factors."""

import itertools

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.mamba.mamba_mixer2 import (
    MAMBA2_IN_PROJ_SLICES,
    MambaMixer2,
    mamba2_in_proj_group_name,
    mamba2_in_proj_groups,
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


@pytest.mark.parametrize(
    ("groups", "low_rank"),
    [
        (GROUPINGS[0], {"x": 8}),
        (GROUPINGS[-1], {"z": 8, "dt": 2, "out": 16}),
        (GROUPINGS[1], {"x": 8, "out": 16}),
        (GROUPINGS[3], {"x": 8}),
    ],
    ids=str,
)
def test_low_rank_slices_match_the_dense_product(bf16_cuda, groups, low_rank):
    config = FormatConfig(groups)
    mixer = make_mixer(config, low_rank=low_rank)
    fill(mixer)
    prefix = mixer.prefix
    # Every group and factor resolves its own quantization method by name.
    expected = []
    for group in mixer.in_proj_groups:
        if group[0] in low_rank:
            expected += [f"{prefix}.{group[0]}_a_proj", f"{prefix}.{group[0]}_b_proj"]
        else:
            expected.append(f"{prefix}.{mamba2_in_proj_group_name(group)}")
    expected += (
        [f"{prefix}.out_a_proj", f"{prefix}.out_b_proj"]
        if "out" in low_rank
        else [f"{prefix}.out_proj"]
    )
    assert config.prefixes == expected

    # Reference: one dense in_proj weight with the slices stacked in row order.
    def weight(group):
        if group[0] in low_rank:
            return (
                getattr(mixer, f"{group[0]}_b_proj").weight.float()
                @ getattr(mixer, f"{group[0]}_a_proj").weight.float()
            )
        return getattr(mixer, mamba2_in_proj_group_name(group)).weight.float()

    rows: dict[str, torch.Tensor] = {}
    for group in mixer.in_proj_groups:
        rows.update(zip(group, weight(group).split([SIZES[n] for n in group])))
    fused = torch.cat([rows[name] for name in MAMBA2_IN_PROJ_SLICES])
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


def test_low_rank_slices_are_their_own_format_group():
    config = FormatConfig([MAMBA2_IN_PROJ_SLICES])
    groups = mamba2_in_proj_groups(config, "model.layers.0.mixer", {"x": 8})
    assert groups == [("z", "B", "C", "dt"), ("x",)]
