# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load plan on tiny CPU models.

Invariant, for these models (whose loaders set no module state): planning
(after ``start_reload``, as the NCCL engine does at the first update) changes
no registration, attribute, address or layout, and no
bytes outside checkpoint tensors, and a direct reload that copies each
name's boxes into its ``get_dest`` tensors (and sends fallbacks in full through
``load_weights``) leaves every tensor byte for byte as a cold start on the same
weights, at the same addresses, on the first update and on later ones that
reuse the plan.
"""

import contextlib
from types import SimpleNamespace

import pytest
import torch

from vllm.distributed.weight_transfer.base import ParamMeta
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.model_loader.reload import (
    finish_reload,
    load_planned,
    plan_loads,
    start_reload,
)
from vllm.model_executor.model_loader.reload.plan import LoadPlan, Region
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.utils import replace_parameter

_MODEL_CONFIG = SimpleNamespace(
    word_embeddings_untied_by_checkpoint=False, quantization=None, dtype=torch.bfloat16
)
_CPU = torch.device("cpu")
BF16 = torch.bfloat16


@pytest.fixture(autouse=True)
def _direct_mode(monkeypatch):
    monkeypatch.setenv("VLLM_RELOAD_MODE", "direct")


def _param(*shape, dtype=BF16, loader=None) -> torch.nn.Parameter:
    p = torch.nn.Parameter(torch.zeros(shape, dtype=dtype), requires_grad=False)
    if loader is not None:
        p.weight_loader = loader
    return p


class _Toy(torch.nn.Module):
    """Loads ``name`` through its parameter's ``weight_loader`` (or
    ``default_weight_loader``); ``stacked`` maps checkpoint names to a fused
    parameter and shard id, as models' ``stacked_params_mapping`` does."""

    stacked: dict[str, tuple[str, object]] = {}
    skipped: frozenset[str] = frozenset()

    def load_weights(self, weights):
        params = dict(self.named_parameters(remove_duplicate=False))
        for name, weight in weights:
            if name in self.skipped:
                continue
            if name in self.stacked:
                target, shard = self.stacked[name]
                param = params[target]
                param.weight_loader(param, weight, shard)
                continue
            param = params[name]
            getattr(param, "weight_loader", default_weight_loader)(param, weight)


def _checkpoint(shapes: dict, seed: int) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return {
        name: torch.randn(shape, generator=gen).to(dtype)
        for name, (shape, dtype) in shapes.items()
    }


def _metadata(shapes: dict) -> list[ParamMeta]:
    return [ParamMeta(n, dtype, tuple(shape)) for n, (shape, dtype) in shapes.items()]


def _cold_start(make, weights):
    model = make()
    model.load_weights(weights.items())
    process_weights_after_loading(model, _MODEL_CONFIG, _CPU)
    return model


def _planned_reload(model, plan: LoadPlan, weights) -> None:
    """What an engine does after ``start_reload``: boxes into their
    destinations, fallbacks in full, in batches of two as a packed transfer
    hands them over."""
    items = list(weights.items())
    for i in range(0, len(items), 2):
        load_planned(model, plan, items[i : i + 2])
    finish_reload(model, _MODEL_CONFIG, _CPU)


def _state(model):
    return {
        name: (t.dtype, tuple(t.shape), t.stride(), t.detach().clone(), t.data_ptr())
        for name, t in [*model.named_parameters(), *model.named_buffers()]
    }


def _registrations(model):
    """Every registered tensor object and every module attribute."""
    return {
        name: (
            {k: id(v) for k, v in m._parameters.items()},
            {k: id(v) for k, v in m._buffers.items()},
            {k: id(v) for k, v in vars(m).items()},
        )
        for name, m in model.named_modules()
    }


def _plan(model, shapes, mirrors=()) -> LoadPlan:
    """``plan_loads`` (after ``start_reload``), checking that planning changed
    no registration, module attribute, address or layout, and no bytes but
    those of checkpoint tensors (the bake runs loaders' constant writes and
    reads of their destination for real) and of ``mirrors``: other tensors a
    fallback's loader writes from its destination."""
    state, registrations = _state(model), _registrations(model)
    plan = plan_loads(model, _metadata(shapes))
    assert _registrations(model) == registrations
    for name, t in [*model.named_parameters(), *model.named_buffers()]:
        dtype, shape, stride, data, ptr = state[name]
        assert (t.dtype, tuple(t.shape), t.stride(), t.data_ptr()) == (
            dtype,
            shape,
            stride,
            ptr,
        ), name
        if not hasattr(t, "_ckpt") and name not in mirrors:
            assert torch.equal(t.view(torch.uint8), data.view(torch.uint8)), name
    return plan


def _check_plan_equals_cold_start(make, shapes, mirrors=()):
    """Plan at the first update's start; that update and a second one reusing
    the plan, with new random weights, must each equal a cold start on them,
    byte for byte, and keep every address."""
    model = _cold_start(make, _checkpoint(shapes, seed=0))
    ptrs = {n: s[4] for n, s in _state(model).items()}
    start_reload(model)
    plan = _plan(model, shapes, mirrors)
    for seed in (1, 2):
        if seed > 1:
            start_reload(model)
        new = _checkpoint(shapes, seed=seed)
        _planned_reload(model, plan, new)
        actual, expected = _state(model), _state(_cold_start(make, new))
        assert actual.keys() == expected.keys()
        for name, (dtype, shape, stride, data, ptr) in actual.items():
            assert (dtype, shape, stride) == expected[name][:3], name
            assert torch.equal(
                data.view(torch.uint8), expected[name][3].view(torch.uint8)
            )
            assert ptr == ptrs[name], f"{name} moved"
    return model, plan


def _offset(dest: torch.Tensor, param: torch.Tensor) -> int:
    """Where ``dest`` starts in ``param``'s storage, relative to ``param``."""
    return dest.storage_offset() - param.storage_offset()


# ------------------------------------------------------------ linear under TP

_LINEAR_SHAPES = {
    "q_proj.weight": ((8, 8), BF16),  # 4 heads of 2
    "k_proj.weight": ((4, 8), BF16),  # 2 kv heads
    "v_proj.weight": ((4, 8), BF16),
    "gate_proj.weight": ((6, 8), BF16),
    "up_proj.weight": ((6, 8), BF16),
    "down_proj.weight": ((8, 6), BF16),
}


def _linear_model():
    def make():
        model = _Toy()
        kw = dict(bias=False, params_dtype=BF16)
        model.qkv_proj = QKVParallelLinear(8, 2, 4, 2, **kw)
        model.gate_up_proj = MergedColumnParallelLinear(8, [6, 6], **kw)
        model.down_proj = RowParallelLinear(6, 8, **kw)
        model.stacked = {
            "q_proj.weight": ("qkv_proj.weight", "q"),
            "k_proj.weight": ("qkv_proj.weight", "k"),
            "v_proj.weight": ("qkv_proj.weight", "v"),
            "gate_proj.weight": ("gate_up_proj.weight", 0),
            "up_proj.weight": ("gate_up_proj.weight", 1),
        }
        return model

    return make


@pytest.mark.parametrize("tp_rank", [0, 1])
def test_qkv_and_merged_column_under_tp(default_vllm_config, monkeypatch, tp_rank):
    from vllm.model_executor import parameter
    from vllm.model_executor.layers import linear

    for module in (linear, parameter):
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: tp_rank)
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 2)
    model, plan = _check_plan_equals_cold_start(_linear_model(), _LINEAR_SHAPES)
    assert all(plan.get_dest(name) for name in _LINEAR_SHAPES)
    # k rows [K/2 * rank, ...) land after this rank's q rows of qkv_proj
    ((box, k_dest),) = plan.get_dest("k_proj.weight")
    assert box == ((2 * tp_rank, 2 * tp_rank + 2), (0, 8))
    assert _offset(k_dest, model.qkv_proj.weight) == 4 * 8
    assert (tuple(k_dest.shape), k_dest.stride()) == ((2, 8), (8, 1))
    # row parallel: this rank's columns, strided over the rows
    ((box, down_dest),) = plan.get_dest("down_proj.weight")
    assert box == ((0, 8), (3 * tp_rank, 3 * tp_rank + 3))
    assert down_dest.is_contiguous()


# ------------------------------------------------------------ toy loaders


def _transposing_loader(param, loaded):
    param.copy_(loaded.t())


def _step_loader(param, loaded):
    param.copy_(loaded[::2])


def _transpose_pwal(layer):
    replace_parameter(layer, "weight", layer.weight.t().contiguous())


class _PwalMethod(QuantizeMethodBase):
    """Post-load processing given as a function of the layer."""

    def __init__(self, fn):
        self.fn = fn

    def create_weights(self, *args, **kwargs):
        pass

    def apply(self, *args, **kwargs):
        raise NotImplementedError

    def process_weights_after_loading(self, layer):
        self.fn(layer)


class _Layer(torch.nn.Module):
    def __init__(self, pwal=None):
        super().__init__()
        if pwal is not None:
            self.quant_method = _PwalMethod(pwal)


_TOY_SHAPES = {
    "embed.weight": ((10, 4), BF16),
    "lm_head.weight": ((10, 4), BF16),  # tied: the loader skips it
    "proj.weight": ((4, 6), BF16),  # transposing loader, transposing PWAL
    "router.bias": ((5,), BF16),  # FP32 parameter: a cast
    "strided.weight": ((8, 3), BF16),  # every other row
}


def _toy_model():
    model = _Toy()
    model.embed = _Layer()
    model.embed.weight = _param(10, 4)
    model.lm_head = _Layer()
    model.lm_head.weight = model.embed.weight
    model.proj = _Layer(_transpose_pwal)
    model.proj.weight = _param(6, 4, loader=_transposing_loader)
    model.router = _Layer()
    model.router.bias = _param(5, dtype=torch.float32)
    model.strided = _Layer()
    model.strided.weight = _param(4, 3, loader=_step_loader)
    model.skipped = frozenset({"lm_head.weight"})
    return model


def test_toy_loaders():
    """One model for transpose, tied alias, cast and step slice, so that their
    targets share a model (and, for the tied pair, a storage)."""
    _, plan = _check_plan_equals_cold_start(_toy_model, _TOY_SHAPES)
    entries = plan.entries

    # P.copy_(W.t()): the transpose lives in the destination's strides
    ((box, dest),) = plan.get_dest("proj.weight")
    assert box == ((0, 4), (0, 6))
    assert (tuple(dest.shape), dest.stride(), dest.dtype) == ((4, 6), (1, 4), BF16)

    # the tied alias needs nothing; nothing to write
    assert plan.get_boxes("lm_head.weight") is None
    assert plan.get_dest("lm_head.weight") == []

    assert plan.get_dest("router.bias") is None
    assert "casts torch.bfloat16 to torch.float32" in entries["router.bias"].reason
    assert plan.get_boxes("router.bias") == (((0, 5),),)

    assert plan.get_dest("strided.weight") is None
    assert "step" in entries["strided.weight"].reason


# ------------------------------------------------------------ fused experts

_E, _I, _H = 4, 4, 3


def _experts_model(ep_rank: int, ep_size: int, tp_rank: int, tp_size: int):
    """``experts.gate_up_proj`` [E, 2I, H] and ``experts.down_proj`` [E, H, I]
    (transformers v5), loaded one expert at a time as RoutedExperts does."""
    local = range(ep_rank * _E // ep_size, (ep_rank + 1) * _E // ep_size)
    i_shard = _I // tp_size

    def w13_loader(param, loaded, shard_id=None):
        for j, e in enumerate(local):
            for half in (0, 1):  # gate rows, then up rows
                rows = loaded[e].narrow(0, half * _I + tp_rank * i_shard, i_shard)
                param[j].narrow(0, half * i_shard, i_shard).copy_(rows)

    def w2_loader(param, loaded, shard_id=None):
        for j, e in enumerate(local):
            param[j].copy_(loaded[e].narrow(1, tp_rank * i_shard, i_shard))

    def make():
        model = _Toy()
        model.experts = _Layer()
        model.experts.w13_weight = _param(
            len(local), 2 * i_shard, _H, loader=w13_loader
        )
        model.experts.w2_weight = _param(len(local), _H, i_shard, loader=w2_loader)
        model.stacked = {
            "experts.gate_up_proj": ("experts.w13_weight", None),
            "experts.down_proj": ("experts.w2_weight", None),
        }
        return model

    return make


_EXPERT_SHAPES = {
    "experts.gate_up_proj": ((_E, 2 * _I, _H), BF16),
    "experts.down_proj": ((_E, _H, _I), BF16),
}


@pytest.mark.parametrize("ep_rank", [0, 1])
def test_fused_experts_under_ep(ep_rank):
    _, plan = _check_plan_equals_cold_start(
        _experts_model(ep_rank, 2, 0, 1), _EXPERT_SHAPES
    )
    # the per-expert loop merges into one write per half
    experts = (2 * ep_rank, 2 * ep_rank + 2)
    assert plan.get_boxes("experts.gate_up_proj") == ((experts, (0, 8), (0, 3)),)
    ((_, dest),) = plan.get_dest("experts.gate_up_proj")
    assert dest.is_contiguous() and tuple(dest.shape) == (2, 8, 3)


@pytest.mark.parametrize("tp_rank", [0, 1])
def test_fused_experts_under_expert_tp(tp_rank):
    """Also pins the internal ``Write``/``Region`` form (RFC §11), which CI
    compares across ranks."""
    _, plan = _check_plan_equals_cold_start(
        _experts_model(0, 1, tp_rank, 2), _EXPERT_SHAPES
    )
    gate = ((0, _E), (2 * tp_rank, 2 * tp_rank + 2), (0, _H))
    up = ((0, _E), (_I + 2 * tp_rank, _I + 2 * tp_rank + 2), (0, _H))
    assert plan.get_boxes("experts.gate_up_proj") == (gate, up)
    writes = plan.entries["experts.gate_up_proj"].writes
    assert writes is not None
    assert {(w.module, w.tensor) for w in writes} == {("experts", "w13_weight")}
    assert [w.dst for w in writes] == [
        Region(0, (_E, 2, _H), (2 * 2 * _H, _H, 1)),
        Region(2 * _H, (_E, 2, _H), (2 * 2 * _H, _H, 1)),
    ]


# ------------------------------------------------------------ fallbacks


def _single(loader, shape=(4,), dtype=BF16, param_shape=None):
    def make():
        model = _Toy()
        model.layer = _Layer()
        model.layer.weight = _param(*(param_shape or shape), dtype=dtype, loader=loader)
        return model

    return make, {"layer.weight": (shape, dtype)}


@pytest.mark.parametrize(
    "loader,reason",
    [
        (lambda p, w: p.copy_(w * 2), "applies"),
        (lambda p, w: p.copy_(w.to(torch.float32).to(BF16)), "applies"),
        (lambda p, w: p.copy_(w if bool(w.sum() > 0) else -w), "reads tensor values"),
        (lambda p, w: p.__setitem__(slice(None), w), "without Tensor.copy_"),
        (lambda p, w: p.copy_(w).mul_(2), "reads the destination"),  # composed
        (lambda p, w: p.copy_(w.expand(2, 4)[0]), "applies"),
        (lambda p, w: torch.add(w, 0, out=p), "without Tensor.copy_ (add.out)"),
        (lambda p, w: (w.to("cpu").float().numpy(), p.copy_(w)), "reads tensor values"),
    ],
    ids=[
        "arith",
        "cast-roundtrip",
        "value-read",
        "setitem",
        "composed",
        "expand",
        "out",
        "numpy",
    ],
)
def test_fallbacks(loader, reason):
    make, shapes = _single(loader)
    _, plan = _check_plan_equals_cold_start(make, shapes)
    assert plan.get_dest("layer.weight") is None
    assert reason in (plan.entries["layer.weight"].reason or "")


def test_noop_to_and_constant_fill_stay_direct():
    """A constant write into a checkpoint tensor (padding, as
    ``VocabParallelEmbedding`` fills) runs while planning and stays out of
    the plan: post-load processing owns padding."""

    def loader(param, loaded):
        param[4:].fill_(1)
        param[:4].copy_(loaded.to(param.device).to(BF16))

    make, shapes = _single(loader, param_shape=(6,))
    model = _cold_start(make, _checkpoint(shapes, 0))
    start_reload(model)
    model.layer.weight.zero_()
    plan = _plan(model, shapes)
    assert plan.get_dest("layer.weight") is not None
    assert model.layer.weight[4:].tolist() == [1, 1]  # ran for real
    assert model.layer.weight[:4].tolist() == [0] * 4  # the trace wrote nothing


@pytest.mark.parametrize(
    "loader",
    [
        lambda p, w: p.copy_(w),
        lambda p, w: p.__setitem__(slice(None), w),
        lambda p, w: p.index_put_((torch.arange(4),), w),
        lambda p, w: p.add_(w),
        lambda p, w: torch.ops.aten.copy_(p, w),
    ],
    ids=["copy_", "setitem", "index_put_", "add_", "aten-copy_"],
)
def test_trace_writes_reach_no_memory(loader):
    """The bake runs loaders on the real tensors, but an incoming tensor is a
    data-free trace: writes from it are recorded or run on meta, never on the
    destination."""
    make, shapes = _single(loader)
    model = _cold_start(make, _checkpoint(shapes, 0))
    start_reload(model)
    model.layer.weight.fill_(7)
    plan_loads(model, _metadata(shapes))
    assert model.layer.weight.tolist() == [7] * 4


@pytest.mark.parametrize(
    "loader",
    [
        lambda p, w: (p.copy_(w), p.sum()),
        lambda p, w: (p.copy_(w), p[0].item()),
        lambda p, w: (torch.equal(p, p), p.copy_(w)),
    ],
    ids=["sum", "item", "equal-before-copy"],
)
def test_reading_the_destination_falls_back(loader):
    make, shapes = _single(loader)
    _, plan = _check_plan_equals_cold_start(make, shapes)
    assert plan.get_dest("layer.weight") is None
    assert "reads the destination" in (plan.entries["layer.weight"].reason or "")


def _mirror_conv1d(mirror, param, loaded):
    param.copy_(loaded)
    if not param.is_meta:  # as Kimi-K3's kda.py guards its decode copies
        mirror.copy_(loaded.view(2, 2).t())


def _mirror_norm(mirror, param, loaded):
    default_weight_loader(param, loaded)
    if not param.is_meta:
        mirror.copy_(param.data.view(2, 2))


@pytest.mark.parametrize(
    "loader,reason",
    [
        (_mirror_conv1d, "not a checkpoint tensor"),
        (_mirror_norm, "reads the destination"),
    ],
    ids=["from-incoming", "from-destination"],
)
def test_mirror_buffer_falls_back(loader, reason):
    """Kimi-K3 KDA's decode copies: a loader that also writes a non-persistent
    buffer, behind ``if not param.is_meta``, is a fallback so the buffer is
    reloaded too."""

    def make():
        model = _Toy()
        model.layer = _Layer()
        mirror = torch.zeros(2, 2, dtype=BF16)
        model.layer.register_buffer("mirror", mirror, persistent=False)
        model.layer.weight = _param(4, loader=lambda p, w: loader(mirror, p, w))
        return model

    shapes = {"layer.weight": ((4,), BF16)}
    _, plan = _check_plan_equals_cold_start(make, shapes, mirrors={"layer.mirror"})
    assert plan.get_dest("layer.weight") is None
    assert reason in (plan.entries["layer.weight"].reason or "")


@pytest.mark.parametrize(
    "loader,shape,param_shape",
    [
        (lambda p, w: p.copy_(w.unflatten(0, (2, 3))), (6, 4), (2, 3, 4)),
        (lambda p, w: p.copy_(w.split([2, 4])[1]), (6, 4), (4, 4)),
        (lambda p, w: p.copy_(w.t().contiguous()[1:3]), (4, 6), (2, 4)),
    ],
    ids=["unflatten", "split", "contiguous"],
)
def test_view_ops_stay_direct(loader, shape, param_shape):
    make, shapes = _single(loader, shape=shape, param_shape=param_shape)
    _, plan = _check_plan_equals_cold_start(make, shapes)
    assert plan.get_dest("layer.weight") is not None


def test_tensors_sharing_one_storage():
    """Two checkpoint tensors on one storage: each write lands in its own
    tensor, whichever tensor names the storage."""

    def make():
        model = _Toy()
        base = torch.zeros(24, dtype=BF16)
        for i, name in enumerate(("a", "b")):
            setattr(model, name, _Layer())
            view = base[12 * i : 12 * (i + 1)].view(3, 4)
            getattr(model, name).weight = torch.nn.Parameter(view, requires_grad=False)
        return model

    shapes = {"a.weight": ((3, 4), BF16), "b.weight": ((3, 4), BF16)}
    model, plan = _check_plan_equals_cold_start(make, shapes)
    for name in ("a", "b"):
        ((_, dest),) = plan.get_dest(f"{name}.weight")
        assert dest.data_ptr() == getattr(model, name).weight.data_ptr()


def _two_names(loader_a):
    """``a`` through ``loader_a``, ``b`` a plain copy."""

    def make():
        model = _Toy()
        model.a = _Layer()
        model.a.weight = _param(4, loader=lambda p, w: loader_a(model, p, w))
        model.b = _Layer()
        model.b.weight = _param(4)
        return model

    return make, {"a.weight": ((4,), BF16), "b.weight": ((4,), BF16)}


def test_value_read_falls_back_and_the_bake_continues():
    def loader(model, param, loaded):
        param.copy_(loaded if bool(loaded.sum() > 0) else -loaded)

    make, shapes = _two_names(loader)
    _, plan = _check_plan_equals_cold_start(make, shapes)
    assert "reads tensor values" in (plan.entries["a.weight"].reason or "")
    assert plan.get_dest("b.weight") is not None


def test_meta_failure_is_not_called_a_value_read():
    """An op on a trace that fails on meta stops the bake like a value read,
    but the fallback reason names the failing op, not "reads tensor
    values"."""

    def loader(model, param, loaded):
        # A shape error. On the real path the loader swallows it; under the
        # trace the op stops the loader, and the loader swallows that too: the
        # stop is still recorded and the name falls back.
        with contextlib.suppress(Exception):
            loaded + torch.ones(3)
        param.copy_(loaded)

    make, shapes = _two_names(loader)
    _, plan = _check_plan_equals_cold_start(make, shapes)
    reason = plan.entries["a.weight"].reason or ""
    assert "fails on meta" in reason and "reads tensor values" not in reason
    assert plan.get_dest("b.weight") is not None


class _Paired(_Toy):
    """Stashes ``weight_a`` until ``weight_b`` arrives: a list that ends up
    empty again, or an attribute that stays (``keep_stash``)."""

    keep_stash = False

    def __init__(self):
        super().__init__()
        self.layer = _Layer()
        self.layer.weight = _param(4)
        self.pending = []

    def load_weights(self, weights):
        for name, w in weights:
            if name == "layer.weight_a" and self.keep_stash:
                self.stash = w
            elif name == "layer.weight_a":
                self.pending.append(w)
            elif self.keep_stash:
                self.layer.weight.copy_(self.stash + w)
            else:
                self.layer.weight.copy_(self.pending.pop() + w)


@pytest.mark.parametrize("keep_stash", [False, True])
def test_paired_loader_is_refused(monkeypatch, keep_stash):
    """Whether the loader keeps its stash or not, the combined write is
    labelled ``weight_a``, so ``weight_b`` writes nothing itself and the plan
    is refused."""
    monkeypatch.setattr(_Paired, "keep_stash", keep_stash)
    shapes = {"layer.weight_a": ((4,), BF16), "layer.weight_b": ((4,), BF16)}
    model = _cold_start(_Paired, _checkpoint(shapes, 0))
    model.__dict__.pop("stash", None)
    start_reload(model)
    with pytest.raises(RuntimeError, match=r"write nothing themselves.*weight_b"):
        plan_loads(model, _metadata(shapes))
    assert model.pending == []


class _PairedWithValueRead(_Toy):
    """``weight_a`` is stashed until ``weight_b`` arrives, and ``x``, which
    arrives between them, reads its values."""

    def __init__(self):
        super().__init__()
        self.layer = _Layer()
        self.layer.weight = _param(4)
        self.x = _Layer()
        self.x.weight = _param(4)
        self.pending = []

    def load_weights(self, weights):
        for name, w in weights:
            if name == "layer.weight_a":
                self.pending.append(w)
            elif name == "x.weight":
                self.x.weight.copy_(w if bool(w.sum() > 0) else -w)
            else:
                self.layer.weight.copy_(self.pending.pop(0) + w)


def test_a_stop_resumes_without_replaying_names():
    """A stop resumes ``load_weights`` after the stopped name instead of
    replaying the names before it, so the stash holds only this bake's
    ``weight_a``: the pair is refused, not silently "not needed"."""
    shapes = {
        "layer.weight_a": ((4,), BF16),
        "x.weight": ((4,), BF16),
        "layer.weight_b": ((4,), BF16),
    }
    model = _cold_start(_PairedWithValueRead, _checkpoint(shapes, 0))
    start_reload(model)
    with pytest.raises(RuntimeError, match=r"write nothing themselves.*weight_b"):
        plan_loads(model, _metadata(shapes))
    assert model.pending == []


def test_a_trace_kept_past_planning_fails_loudly(monkeypatch):
    """A loader that keeps an incoming tensor on the module keeps a planning
    trace; using it in a later ``load_weights`` raises instead of writing
    nowhere."""
    monkeypatch.setattr(_Paired, "keep_stash", True)
    shapes = {"layer.weight_a": ((4,), BF16), "layer.weight_b": ((4,), BF16)}
    model = _cold_start(_Paired, _checkpoint(shapes, 0))
    model.__dict__.pop("stash", None)
    start_reload(model)
    with pytest.raises(RuntimeError):
        plan_loads(model, _metadata(shapes))
    weight_b = torch.ones(4, dtype=BF16)
    with pytest.raises(RuntimeError, match="used after planning"):
        model.load_weights([("layer.weight_b", weight_b)])


def test_overlapping_writes_fall_back():
    """Both tied names write the same storage: only they go through
    ``load_weights``, in arrival order as at cold start."""

    def make():
        model = _toy_model()
        model.skipped = frozenset()
        return model

    _, plan = _check_plan_equals_cold_start(make, _TOY_SHAPES)
    for name, other in (
        ("embed.weight", "lm_head.weight"),
        ("lm_head.weight", "embed.weight"),
    ):
        assert plan.get_dest(name) is None
        assert plan.entries[name].reason == f"writes overlap with {other!r}"
    assert plan.get_dest("proj.weight") is not None


def test_layerwise_mode_is_refused(monkeypatch):
    make, shapes = _single(None)
    model = _cold_start(make, _checkpoint(shapes, 0))
    monkeypatch.setenv("VLLM_RELOAD_MODE", "layerwise")
    with pytest.raises(RuntimeError, match="VLLM_RELOAD_MODE=direct"):
        plan_loads(model, _metadata(shapes))


def test_duplicate_names_are_refused():
    make, shapes = _single(None)
    model = _cold_start(make, _checkpoint(shapes, 0))
    start_reload(model)
    with pytest.raises(ValueError, match="declared twice"):
        plan_loads(model, _metadata(shapes) * 2)


def test_plan_outside_a_direct_reload_is_refused():
    """The plan takes its layout from what ``begin_direct_reload`` applied."""
    make, shapes = _single(None)
    model = _cold_start(make, _checkpoint(shapes, 0))
    with pytest.raises(RuntimeError, match="between start_reload and finish_reload"):
        plan_loads(model, _metadata(shapes))


def test_eplb_is_refused():
    make, shapes = _single(None)
    model = _cold_start(make, _checkpoint(shapes, 0))
    parallel = SimpleNamespace(enable_eplb=True)
    model.layer.moe_config = SimpleNamespace(moe_parallel_config=parallel)
    start_reload(model)
    with pytest.raises(RuntimeError, match="EPLB"):
        plan_loads(model, _metadata(shapes))
