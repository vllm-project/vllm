# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which weight forms each TP rank holds, by the builds a deployment serves
(vllm/models/glm53/nvidia/weights.py ``form_sets``, ``attention_forms``,
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.models.glm53.nvidia import switch, weights

WORLD = 8
PLAIN = {"gamma_x", "gamma_q", "absorb_weight", "gamma_kv"}
SELECTOR = {"gamma_x", "wis_weight", "key_gamma", "key_bias", "gamma_q"}
PROBE = {f"probe.{name}" for name in PLAIN}


def _fp8(*shape: int) -> torch.Tensor:
    return torch.randn(*shape).to(torch.float8_e4m3fn)


def _model(geometry) -> SimpleNamespace:
    def norm(width: int, bias: bool = False) -> SimpleNamespace:
        return SimpleNamespace(
            weight=torch.rand(width, dtype=torch.bfloat16) + 0.5,
            bias=torch.rand(width, dtype=torch.bfloat16) if bias else None,
        )

    def layer() -> SimpleNamespace:
        shape = (geometry.index_dim + geometry.index_heads, geometry.hidden)
        return SimpleNamespace(
            input_layernorm=norm(geometry.hidden),
            self_attn=SimpleNamespace(
                q_a_layernorm=norm(geometry.q_lora),
                kv_a_layernorm=norm(geometry.kv_lora),
                indexer=SimpleNamespace(
                    k_norm=norm(geometry.index_dim, bias=True),
                    wk_weights_proj=SimpleNamespace(
                        weight=SimpleNamespace(
                            data=torch.zeros(shape, dtype=torch.bfloat16)
                        )
                    ),
                ),
            ),
        )

    return SimpleNamespace(layers=[layer(), layer(), layer(), layer()])


def _stash(geometry, rank: int, narrow: bool, wide: bool) -> weights.CheckpointStash:
    g, world_heads = geometry, geometry.heads * WORLD
    stash = weights.CheckpointStash(g, rank, torch.device("cpu"), narrow, wide)
    stash._kept[(0, "self_attn.q_b_proj.weight_scale_inv")] = torch.ones(
        world_heads * g.qk_head // g.block, g.q_lora // g.block
    )
    stash._kept[(0, "self_attn.kv_b_proj.weight_scale_inv")] = torch.ones(
        world_heads * g.kv_b_stride // g.block, g.kv_lora // g.block
    )
    for head in weights.head_groups(g, 0, rank, narrow):
        stash._kept[(0, f"self_attn.q_b_proj.weight@{head}")] = _fp8(
            g.heads * g.qk_head, g.q_lora
        )
        stash._kept[(0, f"self_attn.kv_b_proj.weight@{head}")] = _fp8(
            g.heads * g.kv_b_stride, g.kv_lora
        )
    return stash


def _names(forms: dict[str, torch.Tensor], layer: int = 0) -> set[str]:
    prefix = f"layers.{layer}.attention."
    return {name.removeprefix(prefix) for name in forms if name.startswith(prefix)}


@pytest.mark.parametrize(
    ("tiers", "roles"),
    [
        ((), (True, False)),  # the one-request build
        ((4, 8), (True, True)),  # tier 8 selects on every rank too (request8)
        ((4, 8, 16), (True, True)),  # --max-num-seqs 4
        ((16,), (False, True)),
        ((8,), (False, True)),  # one request of 5-8 tokens fills tier 8: group8
        ((8, 16), (False, True)),
        ((16,), (False, True)),
    ],
)
def test_roles_follow_the_served_tiers(tiers, roles) -> None:
    assert switch.roles(tiers) == roles


@pytest.mark.parametrize(
    ("narrow", "wide"), [(True, False), (True, True), (False, True)]
)
def test_attention_forms_per_rank(geometry, narrow: bool, wide: bool) -> None:
    model = _model(geometry)
    rows = {}
    for rank in range(WORLD):
        stash = _stash(geometry, rank, narrow, wide)
        forms = weights.attention_forms(model, stash, 0, rank)
        assert not stash.remaining()
        names = _names(forms)
        selects = rank < geometry.selector_ranks or wide
        plain = narrow and rank >= geometry.selector_ranks
        expected = (PLAIN if plain else set()) | (
            SELECTOR | PROBE if selects else set()
        )
        assert names == expected, (rank, names ^ expected)
        pages = {name: t for name, t in forms.items() if name.endswith("absorb_weight")}
        rows[rank] = {name.split(".")[-2:][0]: t.shape[0] for name, t in pages.items()}
        if plain and selects:  # the probe worker serves the rank's own heads: a view
            plain_pages = forms["layers.0.attention.absorb_weight"]
            probe_pages = forms["layers.0.attention.probe.absorb_weight"]
            assert probe_pages.data_ptr() == plain_pages.data_ptr()
            assert (
                probe_pages.shape[0]
                * len(weights.head_groups(geometry, 0, rank, narrow))
                == plain_pages.shape[0]
            )
    one_group = min(min(r.values()) for r in rows.values())
    for rank, sizes in rows.items():
        pushes = narrow and geometry.is_push_rank(0, rank)
        assert sizes.get("attention", sizes.get("probe", one_group)) in (
            one_group,
            2 * one_group,
        )
        if "attention" in sizes:  # the plain pages: two head groups on a push rank
            assert sizes["attention"] == one_group * (2 if pushes else 1)
        if "probe" in sizes:
            assert sizes["probe"] == one_group


def test_a_shared_layer_has_the_plain_forms_everywhere(geometry) -> None:
    model = _model(geometry)
    assert geometry.is_shared(3)
    for narrow, wide in ((True, False), (True, True), (False, True)):
        for rank in range(WORLD):
            stash = weights.CheckpointStash(
                geometry, rank, torch.device("cpu"), narrow, wide
            )
            g = geometry
            stash._kept[(3, "self_attn.q_b_proj.weight_scale_inv")] = torch.ones(
                8 * g.heads * g.qk_head // g.block, g.q_lora // g.block
            )
            stash._kept[(3, "self_attn.kv_b_proj.weight_scale_inv")] = torch.ones(
                8 * g.heads * g.kv_b_stride // g.block, g.kv_lora // g.block
            )
            for head in weights.head_groups(g, 3, rank, narrow):
                stash._kept[(3, f"self_attn.q_b_proj.weight@{head}")] = _fp8(
                    g.heads * g.qk_head, g.q_lora
                )
                stash._kept[(3, f"self_attn.kv_b_proj.weight@{head}")] = _fp8(
                    g.heads * g.kv_b_stride, g.kv_lora
                )
            forms = weights.attention_forms(model, stash, 3, rank)
            assert _names(forms, 3) == PLAIN


class _Linear:
    """The vLLM linear's adopt: the module's own tensor and a scales tensor."""

    def __init__(self, key: str, operand: str) -> None:
        self.key, self.operand = key, operand
        self.scales_operand = operand.removesuffix("_weight") + "_scales"

    def adopt(self, module, layer):
        return module, torch.zeros(1)


@pytest.mark.parametrize(
    ("narrow", "wide"), [(True, False), (True, True), (False, True)]
)
def test_adopted_linears_follow_the_forms(
    monkeypatch, geometry, narrow: bool, wide: bool
) -> None:
    linears = {
        "input": _Linear("fused_qkv_a_proj", "input_weight"),
        "o": _Linear("o_proj", "o_weight"),
        "kvb": _Linear("kv_b_proj", "kvb_weight"),
    }
    monkeypatch.setattr(weights, "worker_linears", lambda g: linears)
    monkeypatch.setattr(
        weights, "selector_linear", lambda g: _Linear("indexer.wq_b", "wqb_weight")
    )
    layer = SimpleNamespace(
        self_attn=SimpleNamespace(
            fused_qkv_a_proj=torch.zeros(1),
            o_proj=torch.zeros(1),
            kv_b_proj=torch.zeros(1),
            indexer=SimpleNamespace(wq_b=torch.zeros(1)),
        )
    )
    model = SimpleNamespace(layers=[layer] * geometry.layers)
    for rank in range(WORLD):
        forms: dict[str, torch.Tensor] = {}
        weights.adopt_worker_linears(model, geometry, forms, rank, narrow, wide)
        weights.adopt_selector_linears(model, geometry, forms, rank, wide)
        names = _names(forms, 0)
        plain, selects = weights.form_sets(geometry, 0, rank, narrow, wide)
        worker = {
            f"{family}_{part}"
            for family in ("input", "o", "kvb")
            for part in ("weight", "scales")
        }
        expected = (worker if plain else set()) | (
            {f"probe.{n}" for n in worker} | {"wqb_weight", "wqb_scales"}
            if selects
            else set()
        )
        assert names == expected, (rank, names ^ expected)
        assert _names(forms, 3) == worker  # a shared layer: plain, no selector


_PLAIN = {
    "gamma_x", "gamma_q", "absorb_weight", "gamma_kv", "input_weight", "input_scales",
    "kvb_weight", "kvb_scales", "o_weight", "o_scales",
}  # fmt: skip
_SELECTOR = {
    "gamma_x", "wis_weight", "key_gamma", "key_bias", "gamma_q", "wqb_weight",
    "wqb_scales",
}  # fmt: skip


@pytest.mark.parametrize("k", range(8))
@pytest.mark.parametrize("seqs", range(1, 9))
def test_every_tier_build_binds_over_the_forms_its_roles_hold(
    geometry, seqs: int, k: int
) -> None:
    from test_kernels import _AnyTensors, _bound, _Placement

    from vllm.models.glm53.nvidia import providers

    per = k + 1
    try:
        tiers = switch.derived_tiers(seqs, per)
    except ValueError:
        pytest.skip("1 + K exceeds every build")
    if seqs == 1 and per <= switch.GROUP_ROWS:
        tiers = ()
    narrow, wide = switch.roles(tiers)
    builds = sorted(
        set(switch.tier_claims(seqs, per).values()) if tiers else (),
        key=lambda build: (build.tier, build.w8a16_rows),
    )

    class Held(_AnyTensors):
        def __init__(self, rank: int) -> None:
            super().__init__()
            self.rank = rank

        def __missing__(self, name: str) -> torch.Tensor:
            parts = name.split(".")
            if len(parts) >= 4 and parts[0] == "layers" and parts[2] == "attention":
                layer, probe, leaf = int(parts[1]), parts[3] == "probe", parts[-1]
                if not geometry.is_shared(layer) and leaf in _PLAIN | _SELECTOR:
                    plain, selects = weights.form_sets(
                        geometry, layer, self.rank, narrow, wide
                    )
                    own = (plain and leaf in _PLAIN) or (selects and leaf in _SELECTOR)
                    held = (probe and selects and leaf in _PLAIN) or (not probe and own)
                    assert held, (name, self.rank)
            return super().__missing__(name)

    for build in builds or [None]:
        if build is None:
            built = geometry
        else:
            packed = switch.packed_tier(build.tier, per)
            rows = per if packed else switch.lone_rows(per)
            built = geometry.for_tier(build.tier, rows - 1, packed, build.w8a16_rows)
        for rank in range(8):
            _bound(providers.program(built, rank, Held(rank), _Placement()))
