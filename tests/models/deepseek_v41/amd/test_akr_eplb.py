# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Static expert placement with replicas (akr_eplb): plan validation, the
replica tables and selection, the MegaMoE experts' loader and id remap, the
shipped plan and the AITER profile registration; the planner that made the
shipped plan (akr_eplb_plan) and the opt-in expert-load counting that feeds it
(akr_expert_counts). CPU, except the remap and counting kernel tests."""

import os
import types

import numpy as np
import pytest
import torch

from vllm.models.deepseek_v41.amd import akr_eplb as P

SHIPPED = os.path.join(P.CONFIG_DIR, "expert_plan_S49.npz")
# the expert load the shipped plan was fitted on, summed over the DP ranks
COUNTS = os.path.join(
    os.path.dirname(__file__), "data", "expert_counts_agentx_c1024.npz"
)


def _cfg(path: str):
    return types.SimpleNamespace(
        kernel_config=types.SimpleNamespace(
            amd_kernels=types.SimpleNamespace(expert_perm=path)
        )
    )


def _write(tmp_path, name="p.npz", **kw) -> str:
    f = str(tmp_path / name)
    np.savez(f, **kw)
    return f


def _rep_row(E=384, S=52, seed=0):
    """8 * S slots: every expert once plus 8 * S - E random replicas."""
    r = np.random.default_rng(seed)
    extra = r.choice(E, 8 * S - E, replace=True)
    return r.permutation(np.concatenate([np.arange(E), extra]))


def _plan(L=3, E=384, seed=0, S=48):
    r = np.random.default_rng(seed)
    rows = [
        r.permutation(E) if 8 * S == E else _rep_row(E, S, seed + i) for i in range(L)
    ]
    return dict(phys2log=np.stack(rows).astype(np.int16), layer_ids=np.arange(L))


def test_shipped_plan_valid(tmp_path, monkeypatch):
    plan = P.Plan(SHIPPED)
    assert sorted(plan.layers[384]) == list(range(40))
    assert sorted(plan.layers[128]) == [40, 41, 42]
    assert plan.slots == {384: 49, 128: 17} and plan.world == 8
    # a bare name resolves to the shipped file, even where the working
    # directory holds a file of that name; a path names the user's file
    monkeypatch.chdir(tmp_path)
    _write(tmp_path, "expert_plan_S49.npz", **_plan())
    assert P.plan_path(_cfg("expert_plan_S49.npz")) == SHIPPED
    assert P.plan_path(_cfg("./expert_plan_S49.npz")) == "./expert_plan_S49.npz"
    assert P.plan_path(_cfg("mine.npz")) == "mine.npz"
    assert P.plan_path(_cfg("")) == "" and P.plan_path(None) == ""


def test_permutation_loader_slots():
    row = np.random.default_rng(1).permutation(384)
    lp = P.LayerPerm(row)
    assert lp.slots_per_rank == 48 and lp.max_rep == 1 and not lp.is_identity
    assert np.array_equal(lp.phys2log[lp.l2p[:, 0]], np.arange(384))
    for r in range(8):
        got = [e for e in range(384) for _ in lp.local_slots(e, r * 48, 48)]
        assert sorted(got) == sorted(row[r * 48 : (r + 1) * 48])
    assert P.LayerPerm(np.arange(384)).is_identity


def test_replica_loader_and_tables():
    row = _rep_row(384, 52, 4)
    lp = P.LayerPerm(row, layer=3, rank=2)
    assert lp.slots_per_rank == 52 and lp.max_rep >= 2
    # slot s of rank r receives exactly phys2log[r * 52 + s], once
    for r in range(8):
        got: dict[int, list[int]] = {}
        for e in range(384):
            for s in lp.local_slots(e, r * 52, 52):
                got.setdefault(s, []).append(e)
        assert all(len(v) == 1 for v in got.values()) and sorted(got) == list(range(52))
        assert [got[s][0] for s in range(52)] == list(row[r * 52 : (r + 1) * 52])
    for e in range(384):
        assert sorted(lp.l2p[e, : lp.count[e]]) == sorted(np.flatnonzero(row == e))


def test_replica_selection_covers_and_balances():
    r = np.random.default_rng(5)
    row = r.permutation(
        np.concatenate([np.arange(384), [7] * 9, r.choice(384, 55, replace=True)])
    )
    lp = P.LayerPerm(row, layer=11, rank=3)
    ids = np.tile(np.arange(384, dtype=np.int32), (3000, 1)).reshape(-1, 6)
    a = lp.remap_reference(ids)
    assert np.array_equal(a, lp.remap_reference(ids))
    assert np.array_equal(lp.phys2log[a], ids)
    for e in np.flatnonzero(lp.count > 1):
        used = np.bincount(a[ids == e], minlength=lp.n_slots)[lp.l2p[e, : lp.count[e]]]
        assert (used > 0).all() and used.max() / used.min() < 1.3, (e, used)
    b = ids.copy()
    b[1, 1] = -1
    b[5] = -1
    out = lp.remap_reference(b)
    assert (out[b < 0] == -1).all() and np.array_equal(out[b >= 0], a[b >= 0])
    # the salt differs per layer: another replica stream, same logical experts
    other = P.LayerPerm(row, layer=12, rank=3).remap_reference(ids)
    assert not np.array_equal(other, a) and np.array_equal(lp.phys2log[other], ids)


def test_validation_fails_loudly(tmp_path):
    ok = _plan()
    P.Plan(_write(tmp_path, **ok))
    bad = _plan()
    bad["phys2log"][1, bad["phys2log"][1] == 5] = 6
    with pytest.raises(ValueError, match="no slot"):
        P.Plan(_write(tmp_path, "b.npz", **bad))
    with pytest.raises(ValueError, match="not world"):
        P.check_row(np.arange(100), 384, 8, "x")
    with pytest.raises(ValueError, match="outside"):
        P.check_row(np.arange(1, 385), 384, 8, "x")
    with pytest.raises(ValueError, match="no 'phys2log'"):
        P.Plan(_write(tmp_path, "c.npz", foo=np.arange(3)))
    dup = _plan()
    dup["layer_ids"] = np.array([0, 0, 1])
    with pytest.raises(ValueError, match="bad"):
        P.Plan(_write(tmp_path, "d.npz", **dup))
    w4 = _write(tmp_path, "w4.npz", world_size=4, **_plan(S=96))
    with pytest.raises(ValueError, match="made for EP size 4"):
        P.layer_perm(_cfg(w4), "model.layers.0.ffn.experts", 384, 0, 8)


def test_layer_perm_lookup(tmp_path):
    kw = _plan(L=2)
    kw["phys2log"][1] = np.arange(384)  # identity row: no remap
    kw.update(
        draft_phys2log=np.stack([np.random.default_rng(3).permutation(128)]),
        draft_layer_ids=np.array([40]),
    )
    cfg = _cfg(_write(tmp_path, **kw))
    assert P.layer_perm(cfg, "model.layers.0.ffn.experts", 384, 0, 8) is not None
    assert P.layer_perm(cfg, "model.layers.1.ffn.experts", 384, 0, 8) is None
    with pytest.raises(ValueError, match="layer 2"):
        P.layer_perm(cfg, "model.layers.2.ffn.experts", 384, 0, 8)
    assert P.layer_perm(cfg, "model.layers.40.ffn.experts", 128, 0, 8) is not None
    assert P.layer_perm(cfg, "model.layers.41.ffn.experts", 128, 0, 8) is None
    with pytest.raises(ValueError, match="no layers with 256"):
        P.layer_perm(cfg, "model.layers.0.ffn.experts", 256, 0, 8)
    assert P.layer_perm(_cfg(""), "model.layers.0.ffn.experts", 384, 0, 8) is None
    assert P.layer_index("draft.model.layers.41.ffn.experts") == 41
    with pytest.raises(ValueError):
        P.layer_index("model.ffn.experts")


def test_config_field():
    from vllm.config.kernel import AmdKernelsConfig

    assert AmdKernelsConfig().expert_perm == ""
    assert AmdKernelsConfig(expert_perm="x.npz").expert_perm == "x.npz"


def _experts(monkeypatch, path: str, rank: int, prefix="model.layers.7.ffn.experts"):
    from vllm.models.deepseek_v4.amd import mega_moe

    monkeypatch.setattr(
        mega_moe,
        "get_ep_group",
        lambda: types.SimpleNamespace(rank_in_group=rank, world_size=8),
    )
    cfg = _cfg(path)
    cfg.scheduler_config = types.SimpleNamespace(max_num_batched_tokens=8192)
    return mega_moe.DeepseekV4AiterMegaMoEExperts(
        cfg,
        num_experts=384,
        top_k=6,
        hidden_size=64,
        intermediate_size=64,
        swiglu_limit=10.0,
        prefix=prefix,
    )


def test_megamoe_loader_places_replicas(monkeypatch):
    """Each rank allocates S slots and gets every logical expert its slots
    hold; without a plan the layout is unchanged."""
    monkeypatch.setattr(P, "register_aiter_profile", lambda *a: False)
    plan = P.Plan(SHIPPED)
    row = plan.layers[384][7]
    for rank in (0, 5):
        ex = _experts(monkeypatch, SHIPPED, rank)
        assert ex.num_routed_local_experts == 49 and ex.num_experts == 392
        assert ex.num_logical_experts == 384 and ex.w13_weight.shape[0] == 49
        for e in range(384):
            w = torch.full((64, 32), e % 251, dtype=torch.uint8)
            got = ex.weight_loader(
                ex.w13_weight, w, "w13", "w1", e, return_success=True
            )
            assert got == (e in row[rank * 49 : (rank + 1) * 49])
        want = torch.tensor(row[rank * 49 : (rank + 1) * 49] % 251, dtype=torch.uint8)
        assert torch.equal(ex.w13_weight.data[:, :64, 0], want[:, None].expand(49, 64))
    ident = _experts(monkeypatch, "", 2)
    assert ident._perm is None and ident.num_routed_local_experts == 48
    assert ident.weight_loader(
        ident.w2_weight, torch.zeros(64, 32, dtype=torch.uint8), "w2", "w2", 96, True
    )
    assert not ident.weight_loader(
        ident.w2_weight, torch.zeros(64, 32, dtype=torch.uint8), "w2", "w2", 95, True
    )


def test_aiter_profile_registration(monkeypatch):
    mc = pytest.importorskip(
        "aiter.ops.flydsl.kernels.mega_moe.mega_moe_config",
        reason="needs AITER's MegaMoE config",
    )
    if not any(p.name == "v41_flash_ep8" for p in mc._NETWORKS):
        pytest.skip("this AITER has no v41_flash_ep8 profile")
    monkeypatch.setattr(mc, "_NETWORKS", mc._NETWORKS)
    mc.build_mega_moe_bundle_plan.cache_clear()
    base = mc._match_network(49, 5120, 2304)
    assert P.register_aiter_profile(17, 5120, 2304) is False
    assert P.register_aiter_profile(49, 7168, 2048) is False
    if base is not None:  # AITER already covers it
        assert P.register_aiter_profile(49, 5120, 2304) is False
        return
    assert P.register_aiter_profile(49, 5120, 2304) is True
    prof = mc._match_network(49, 5120, 2304)
    assert prof is not None and prof.experts_per_rank == 49
    assert P.register_aiter_profile(49, 5120, 2304) is False  # idempotent
    tuned = mc.select_mega_moe_config(
        4096, 32768, experts_per_rank=49, model_dim=5120, inter_dim=2304
    )
    ref = mc.select_mega_moe_config(
        4096, 32768, experts_per_rank=48, model_dim=5120, inter_dim=2304
    )
    assert tuned.stage2 == ref.stage2
    mc.build_mega_moe_bundle_plan.cache_clear()


def test_shipped_plan_balances_measured_load():
    """Routes drawn from the measured per-layer expert popularity through the
    replica selection: the per-rank load max/mean drops well below identity."""
    plan = P.Plan(SHIPPED)
    with np.load(COUNTS) as z:
        tot = (z["counts_decode"] + z["counts_prefill"]).astype(np.float64)
    rng = np.random.default_rng(0)
    lam_id, lam_pl = [], []
    for layer in (0, 11, 22, 33, 39):
        pop = tot[layer] / tot[layer].sum()
        ids = np.stack(
            [rng.choice(384, 6, replace=False, p=pop) for _ in range(2000)]
        ).astype(np.int32)
        ids = np.tile(ids, (60, 1))
        lp = P.LayerPerm(plan.layers[384][layer], layer, 0, 8)
        load = np.bincount(
            lp.remap_reference(ids).ravel() // lp.slots_per_rank, minlength=8
        )
        ident = np.bincount(ids.ravel() // 48, minlength=8)
        lam_id.append(ident.max() / ident.mean())
        lam_pl.append(load.max() / load.mean())
    assert np.mean(lam_pl) < 1.12 and np.mean(lam_pl) < np.mean(lam_id) - 0.2


def test_planner_reproduces_shipped_plan(tmp_path, capsys):
    """The planner CLI on the measured counts writes the shipped plan's
    arrays, bit for bit."""
    from vllm.models.deepseek_v41.amd import akr_eplb_plan

    out = str(tmp_path / "plan.npz")
    akr_eplb_plan.main(["--counts", COUNTS, "--out", out, "--slots", "49"])
    assert "slots per rank {384: 49, 128: 17}" in capsys.readouterr().out
    with np.load(out) as got, np.load(SHIPPED) as want:
        for k in ("phys2log", "layer_ids", "draft_phys2log", "draft_layer_ids"):
            assert got[k].dtype == want[k].dtype and np.array_equal(got[k], want[k])
        assert int(got["world_size"]) == int(want["world_size"]) == 8


def test_planner_rules(tmp_path):
    """Replicas go to the hottest experts, at most one per rank; every expert
    keeps a slot; the plan balances both objectives and loads at runtime."""
    from vllm.models.deepseek_v41.amd import akr_eplb_plan as PL

    rng = np.random.default_rng(7)
    decode = (rng.zipf(1.6, size=(2, 384)) * 100).astype(np.int64)
    prefill = rng.permutation(decode.T).T  # other hot experts in prefill
    counts = dict(
        counts_decode=decode,
        counts_prefill=prefill,
        layer_ids=np.array([0, 1]),
        draft_counts_decode=decode[:1, :128],
        draft_counts_prefill=prefill[:1, :128],
        draft_layer_ids=np.array([40]),
    )
    plan = PL.make_plan(counts, 52)
    assert plan["phys2log"].shape == (2, 8 * 52)
    assert plan["draft_phys2log"].shape == (1, 8 * PL.default_draft_slots(52))
    for row, d, p in zip(plan["phys2log"], decode, prefill):
        nrep = np.bincount(row, minlength=384)
        assert nrep.min() == 1 and nrep.max() <= 8
        for r in range(8):
            assert len(set(row[r * 52 : (r + 1) * 52])) == 52
        hot = np.argmax(np.maximum(d / d.sum(), p / p.sum()))
        assert nrep[hot] == nrep.max() > 1
        ident = np.arange(384)
        for load in (d, p):
            assert PL.rank_imbalance(load, row) < PL.rank_imbalance(load, ident)
            assert PL.rank_imbalance(load, row) < 1.1
    f = str(tmp_path / "p.npz")
    np.savez(f, **plan)
    assert P.Plan(f).slots == {384: 52, 128: PL.default_draft_slots(52)}
    with pytest.raises(ValueError, match="slots for 384 experts"):
        PL.replicate(np.ones((1, 384)), 8 * 384 + 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("rows", [1, 16, 192, 1000, 4096])
def test_remap_kernel_matches_reference(rows):
    plan = P.Plan(SHIPPED)
    for n, layer in ((384, 20), (128, 41)):
        lp = P.LayerPerm(plan.layers[n][layer], layer, 3, 8)
        k = 6 if n == 384 else 3
        g = torch.Generator().manual_seed(rows)
        ids = torch.randint(0, n, (rows, k), generator=g, dtype=torch.int32)
        ids[::7, 1] = -1
        got = lp.remap(ids.cuda())
        assert got.dtype == torch.int32 and got.shape == ids.shape
        assert np.array_equal(got.cpu().numpy(), lp.remap_reference(ids.numpy()))
        # captured once, replayed on new ids
        static = ids.cuda()
        out = lp.remap(static)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = lp.remap(static)
        new = torch.randint(0, n, (rows, k), generator=g, dtype=torch.int32)
        static.copy_(new)
        graph.replay()
        torch.accelerator.synchronize()
        assert np.array_equal(out.cpu().numpy(), lp.remap_reference(new.numpy()))


def test_expert_counts_opt_in(monkeypatch):
    """`VLLM_AKR_EXPERT_COUNTS` picks the counting model state and the MegaMoE
    count hook; unset, neither runs."""
    from vllm.models.deepseek_v41.amd.model import DeepseekV41LLMForCausalLM
    from vllm.models.deepseek_v41.nvidia.model_state import DeepseekV41ModelState

    monkeypatch.delenv("VLLM_AKR_EXPERT_COUNTS", raising=False)
    assert DeepseekV41LLMForCausalLM.get_model_state_cls() is DeepseekV41ModelState
    assert not _experts(monkeypatch, "", 0)._count_experts
    monkeypatch.setenv("VLLM_AKR_EXPERT_COUNTS", "/tmp/counts")
    cls = DeepseekV41LLMForCausalLM.get_model_state_cls()
    assert cls.__name__ == "ExpertCountsModelState"
    assert issubclass(cls, DeepseekV41ModelState)
    assert _experts(monkeypatch, "", 0)._count_experts


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_expert_counts_graph_replay(tmp_path, monkeypatch):
    """Counts under CUDA graph replay and eager steps equal the bincount of
    the routed ids on real rows, split by step kind; the planner reads the
    dump."""
    from vllm.config import VllmConfig
    from vllm.forward_context import set_forward_context
    from vllm.models.deepseek_v41.amd import akr_eplb_plan
    from vllm.models.deepseek_v41.amd import akr_expert_counts as EC

    monkeypatch.setattr(EC, "_S", EC._State())
    EC.configure(str(tmp_path), 3)
    layers = [
        ("model.layers.0.ffn.experts", 384, 6),
        ("model.layers.39.ffn.experts", 384, 6),
        ("model.layers.40.ffn.experts", 128, 3),
    ]
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(0)
    max_rows = 4096
    is_padding = torch.ones(max_rows, dtype=torch.bool, device=dev)
    ref = {name: np.zeros((2, n), np.int64) for name, n, _ in layers}

    def rand_ids(m, n, k):
        ids = torch.rand(m, n, device=dev, generator=g).argsort(1)[:, :k].int()
        ids[torch.rand(m, device=dev, generator=g) < 0.1] = -1
        return ids

    def run(ids_list):
        for (name, n, _), ids in zip(layers, ids_list):
            EC.count(name, n, ids)

    def expect(kind, real, ids_list):
        for (name, n, _), ids in zip(layers, ids_list):
            x = ids[:real].cpu().numpy().ravel()
            ref[name][kind] += np.bincount(x[x >= 0], minlength=n)

    cfg = VllmConfig()
    with torch.inference_mode(), set_forward_context(None, cfg, is_padding=is_padding):
        run([rand_ids(64, n, k) for _, n, k in layers])  # dropped at step 1
        rows = 192
        static = [rand_ids(rows, n, k) for _, n, k in layers]
        EC.step_begin(False)
        steps = [1, 0]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run(static)
        assert EC._S.buf is not None and EC._S.buf.sum().item() == 0
        for step in range(12):
            kind = int(step % 3 == 2)
            EC.step_begin(bool(kind))
            steps[kind] += 1
            if step % 4 == 3:  # eager, prefill-sized
                m = int(torch.randint(500, max_rows, (1,)).item())
                real = int(torch.randint(1, m + 1, (1,)).item())
                ids_list = [rand_ids(m, n, k) for _, n, k in layers]
            else:
                m, real = rows, int(torch.randint(0, rows + 1, (1,)).item())
                for (_, n, k), ids in zip(layers, static):
                    ids.copy_(rand_ids(rows, n, k))
                ids_list = static
            is_padding[:real].fill_(False)
            is_padding[real:].fill_(True)
            if ids_list is static:
                graph.replay()
            else:
                run(ids_list)
            expect(kind, real, ids_list)
        torch.accelerator.synchronize()
    path = EC.dump()
    EC._S.stop.set()
    assert path == str(tmp_path / "expert_counts_dp3.npz")
    with np.load(path) as z:
        assert list(z["layer_ids"]) == [0, 39] and list(z["draft_layer_ids"]) == [40]
        for i, name in enumerate(n for n, _, _ in layers[:2]):
            assert np.array_equal(z["counts_decode"][i], ref[name][0])
            assert np.array_equal(z["counts_prefill"][i], ref[name][1])
        assert np.array_equal(z["draft_counts_decode"][0], ref[layers[2][0]][0])
        assert np.array_equal(z["draft_counts_prefill"][0], ref[layers[2][0]][1])
        assert (int(z["steps_decode"]), int(z["steps_prefill"])) == tuple(steps)
    counts = akr_eplb_plan.load_counts([str(tmp_path)])
    assert counts["counts_decode"].shape == (2, 384)
