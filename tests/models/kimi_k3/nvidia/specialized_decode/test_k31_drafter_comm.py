# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The specialized DSpark drafter's collectives (`drafter/comm.py`) on real ranks
(2 GPUs by default; `KIMI_K31_DRAFTER_COMM_WORLD=4` / 8 runs more).

- `ArNorm`: the all-reduce + residual + RMSNorm against a reference that sums the
  ranks' BF16 rows in FP32 in rank order and rounds once (flashinfer's one-shot
  order), adds the residual with one BF16 rounding and normalizes the new residual
  (FP32 sum of squares in another order: at most one BF16 ulp on the output);
  the new residual is bit-exact. Both mailbox parities, repeated calls (re-arm),
  1-64 rows, with and without a residual; also against vLLM's own
  `fused_allreduce_rms_norm` (flashinfer) when it runs.
- `Gather`: the context and logits gathers are byte copies of NCCL's
  all-gather (a sentinel pair (+0, -0) arrives as (+0, +0)); the in-place query
  gather fills head block `rank` of every rank's DCP q mailbox.
- `DcpValue` as the draft DCP attention calls it (flashinfer natural-log LSE):
  the LSE merge over the DCP sources and W_UV against an FP32 reference.
"""

import os

import pytest
import torch
import torch.multiprocessing as mp

from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port

WORLD = int(os.environ.get("KIMI_K31_DRAFTER_COMM_WORLD", "2"))

pytestmark = [
    pytest.mark.skipif(
        not current_platform.is_device_capability_family(100),
        reason="the specialized decode kernels need SM100/SM103",
    ),
    pytest.mark.skipif(
        torch.accelerator.device_count() < WORLD, reason=f"Need {WORLD} GPUs."
    ),
]


def _bf16_ulp(x):
    """One BF16 ulp of each element of `x` (FP32)."""
    e = torch.floor(torch.log2(x.abs().clamp_min(2.0**-126)))
    return torch.exp2(e - 7)


def _reference_ar_norm(parts, residual, weight, eps):
    """Rank-order FP32 sum, one BF16 rounding; + residual (BF16); RMSNorm."""
    acc = parts[0].float()
    for p in parts[1:]:
        acc = acc + p.float()
    reduced = acc.to(torch.bfloat16)
    new = reduced if residual is None else (reduced.float() + residual.float())
    new = new.to(torch.bfloat16)
    x = new.float()
    rstd = torch.rsqrt(x.pow(2).sum(-1, keepdim=True) / x.shape[-1] + eps)
    out = (x * rstd * weight.float()).to(torch.bfloat16)
    return out, new


def _check_ar_norm(comm, rank, world, device):
    from vllm.models.kimi_k3.nvidia.specialized_decode.drafter.comm import HIDDEN

    class Norm:
        weight: torch.Tensor
        variance_epsilon: float

    norm = Norm()
    gen = torch.Generator(device="cpu").manual_seed(1234)
    norm.weight = (1 + 0.1 * torch.randn(HIDDEN, generator=gen)).to(torch.bfloat16)
    norm.weight = norm.weight.to(device)
    norm.variance_epsilon = 1e-5
    calls = 0
    for rows in (1, 4, 5, 20, 64):
        for with_residual in (True, False):
            for repeat in range(3):
                seed = 100 * rows + 10 * repeat + with_residual
                g = torch.Generator(device="cpu").manual_seed(seed)
                parts = [
                    torch.randn(rows, HIDDEN, generator=g).to(torch.bfloat16)
                    for _ in range(world)
                ]
                # Sentinel pairs (+0, -0) and negative zeros in the payload.
                parts[0][0, 0], parts[0][0, 1] = 0.0, -0.0
                parts[-1][-1, 6] = -0.0
                residual = torch.randn(rows, HIDDEN, generator=g).to(torch.bfloat16)
                ref_out, ref_res = _reference_ar_norm(
                    parts, residual if with_residual else None, norm.weight.cpu(), 1e-5
                )
                x = parts[rank].to(device)
                res = residual.to(device) if with_residual else None
                out = comm.ar_norm(x, res, norm, calls % 2)
                calls += 1
                assert out is not None
                got_out, got_res = (t.cpu() for t in out)
                assert torch.equal(got_res, ref_res), (rows, with_residual)
                diff = (got_out.float() - ref_out.float()).abs()
                bound = _bf16_ulp(ref_out.float())
                assert bool((diff <= bound).all()), (rows, float(diff.max()))
                assert (diff > 0).float().mean() < 1e-3
    return calls


def _check_stock_ar_norm(comm, rank, world, device):
    """Against vLLM's fused_allreduce_rms_norm (flashinfer one-shot) when it runs."""
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.models.common.ops.fused_allreduce_rms_norm import (
        fused_allreduce_rms_norm,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.drafter.comm import HIDDEN

    norm = RMSNorm(HIDDEN, eps=1e-5).to(device=device, dtype=torch.bfloat16)
    g = torch.Generator(device="cpu").manual_seed(7)
    with torch.no_grad():
        norm.weight.copy_(1 + 0.1 * torch.randn(HIDDEN, generator=g))
    for rows in (4, 20):
        parts = [torch.randn(rows, HIDDEN, generator=g) for _ in range(world)]
        parts = [p.to(torch.bfloat16) for p in parts]
        residual = torch.randn(rows, HIDDEN, generator=g).to(torch.bfloat16)
        x, res = parts[rank].to(device), residual.to(device)
        stock_out, stock_res = fused_allreduce_rms_norm(x.clone(), res.clone(), norm)
        ours_out, ours_res = comm.ar_norm(x, res, norm, rows % 2)
        for stock, ours in ((stock_res, ours_res), (stock_out, ours_out)):
            diff = (stock.float() - ours.float()).abs()
            assert bool((diff <= 2 * _bf16_ulp(stock.float())).all()), rows


def _check_gathers(comm, rank, world, device):
    from vllm.models.kimi_k3.nvidia.specialized_decode.drafter.comm import HIDDEN

    for rows in (1, 5, 20, 80):
        g = torch.Generator(device="cpu").manual_seed(rows)
        full = torch.randn(rows, HIDDEN, generator=g).to(torch.bfloat16)
        full[0, 0], full[0, 1] = 0.0, -0.0  # a sentinel pair: arrives as (+0, +0)
        shard = full.view(rows, world, -1)[:, rank].contiguous().to(device)
        for _ in range(2):
            got = comm.gather_context(shard)
            assert got is not None
            expect = full.clone()
            expect[0, 1] = 0.0
            assert torch.equal(got.cpu().view(torch.int16), expect.view(torch.int16))
    vs = comm.vocab_shard
    for rows in (4, 16, 64):
        g = torch.Generator(device="cpu").manual_seed(1000 + rows)
        full = (10 * torch.randn(rows, vs * world, generator=g)).to(torch.bfloat16)
        shard = full.view(rows, world, vs)[:, rank].contiguous().to(device)
        got = comm.gather_logits(shard)
        assert got is not None
        assert torch.equal(got.cpu().view(torch.int16), full.view(torch.int16))


def _check_query_gather(dcp, rank, world, device):
    """The in-place FP8 query gather into the DCP q mailbox (then re-armed as
    DcpValue would)."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.drafter.comm import (
        Q_SENT,
        compile_gather,
        gather_ctas,
    )

    heads = dcp.heads
    for rows in (4, 16, 64):
        for parity in (0, 1):
            g = torch.Generator(device="cpu").manual_seed(rows + parity)
            full = torch.randint(0, 0x7F, (rows, world * heads, 576), generator=g)
            full = full.to(torch.uint8)
            mine = full[:, rank * heads : (rank + 1) * heads].contiguous().to(device)
            words = heads * 576 // 4
            kernel = compile_gather(world, words, dcp.cap, Q_SENT, False, False, device)
            mailbox = dcp.box.q[parity]
            mc = dcp.box.q_mc + parity * mailbox.numel()
            flat = mailbox.view(torch.int32).view(dcp.cap, -1)
            shard = mine.view(rows, -1).view(torch.int32)
            ctas = gather_ctas(rows, words * world // 4)
            kernel(shard, flat[:rows], flat, mc, rank, ctas)
            torch.accelerator.synchronize(device)
            assert torch.equal(mailbox[:rows].cpu(), full)
            torch.distributed.barrier()
            mailbox[:rows].fill_(0xFF)  # re-arm (DcpValue's job in the layer)
            torch.accelerator.synchronize(device)
            torch.distributed.barrier()


def _check_value(dcp, rank, world, device):
    """DcpValue with a natural-log LSE and the +inf gate: merge + W_UV."""
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.host import (
        stream,
        to_cute,
    )
    from vllm.models.kimi_k3.nvidia.specialized_decode.drafter.attention import (
        cutlass_int,
    )

    heads, size = dcp.heads, world
    for rows, q in ((4, 4), (16, 4), (20, 5)):
        reqs = rows // q
        g = torch.Generator(device="cpu").manual_seed(rows)
        # Every rank's local rows for all size * heads heads + natural-log LSE.
        all_rows = torch.randn(size, rows, size * heads, 512, generator=g)
        all_rows = all_rows.to(torch.bfloat16)
        all_lse = 3 * torch.randn(size, rows, size * heads, generator=g)
        w_uv = (torch.randn(heads, 512, 128, generator=g) / 16).to(torch.bfloat16)
        lens = torch.full((reqs,), 100, dtype=torch.int32)
        weights = torch.softmax(all_lse, dim=0)  # [size, rows, size*heads]
        merged = (weights[..., None] * all_rows.float()).sum(0).to(torch.bfloat16)
        mine = merged[:, rank * heads : (rank + 1) * heads].float()  # [rows, H, 512]
        ref = torch.einsum("rhl,hlv->rhv", mine, w_uv.float()).reshape(rows, -1)
        rows_d = all_rows[rank].to(device).view(rows, 1, size * heads, 512)
        lse_d = all_lse[rank].to(device).view(rows, 1, size * heads).contiguous()
        out = torch.empty(rows, heads * 128, dtype=torch.bfloat16, device=device)
        w_uv_d = dcp.latent_major(w_uv.to(device))
        value = dcp.value(rows, q, w_uv_d)
        mail, parity = dcp.box.mailbox(rows), rows % 2
        totals = lens.to(device)
        value(
            to_cute(rows_d),
            to_cute(lse_d, 4),
            to_cute(totals, 4),
            to_cute(mail.partial[parity]),
            to_cute(mail.lse[parity], 4),
            to_cute(mail.peers, 8),
            to_cute(mail.lse_peers, 8),
            to_cute(dcp.box.q[parity]),
            to_cute(w_uv_d),
            to_cute(dcp.gate[:rows]),
            to_cute(totals, 4),
            to_cute(out),
            cutlass_int(parity),
            cutlass_int(rank),
            stream(device),
        )
        torch.accelerator.synchronize(device)
        got = out.cpu().float()
        err = (got - ref).abs().max() / ref.abs().max()
        assert err < 1e-2, (rows, float(err))
        torch.distributed.barrier()


def _worker(rank, world, port, result):
    os.environ.update(
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
    )
    device = torch.device("cuda", rank)
    torch.accelerator.set_device_index(rank)
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )

    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(world, rank, f"tcp://127.0.0.1:{port}", rank)
        initialize_model_parallel(tensor_model_parallel_size=world)
        from vllm.models.kimi_k3.nvidia.specialized_decode.drafter.attention import (
            DraftDcp,
        )
        from vllm.models.kimi_k3.nvidia.specialized_decode.drafter.comm import (
            ROWS_CAP,
            DrafterComm,
        )

        tp = get_tp_group()
        comm = DrafterComm(tp.device_group, device, world, rank, 163840 // world)
        _check_ar_norm(comm, rank, world, device)
        _check_stock_ar_norm(comm, rank, world, device)
        _check_gathers(comm, rank, world, device)
        dcp = DraftDcp(tp.device_group, device, world, rank, 64 // world, ROWS_CAP)
        _check_query_gather(dcp, rank, world, device)
        _check_value(dcp, rank, world, device)
        torch.distributed.barrier()
    result[rank] = 1


def test_drafter_comm():
    ctx = mp.get_context("spawn")
    result = ctx.Array("i", [0] * WORLD)
    port = get_open_port()
    procs = [
        ctx.Process(target=_worker, args=(r, WORLD, port, result)) for r in range(WORLD)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=600)
    assert all(p.exitcode == 0 for p in procs), [p.exitcode for p in procs]
    assert list(result) == [1] * WORLD


if __name__ == "__main__":
    test_drafter_comm()
    print("ok")


# ---------------------------------------------------------------------------------
# The whole draft step at TP2 / DCP2 against the stock collectives.
# ---------------------------------------------------------------------------------

MODEL = os.environ.get("KIMI_K31_DEP_MODEL")
DRAFT = os.environ.get("KIMI_K31_DEP_DRAFT")
STEP_CASES = ("d1=1*5@1900:4", "d4=4*5@1500:4", "d12=12*5@900:4")


def _step_worker(rank, port, out_dir, dcp):
    """`vllm bench forward` in-process on the layer-truncated K3.1 with its DSpark
    drafter; every eager draft step runs twice from the same state: with the stock
    collectives (the specialized drafter's hook, context and lm_head mailboxes
    detached: aa72383eeb's drafter) and with the mailboxes; the draft tokens and
    the draft logits cache are compared."""
    import json

    os.environ.update(
        RANK=str(rank),
        WORLD_SIZE="2",
        LOCAL_RANK=str(rank),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        VLLM_ALLOW_LONG_MAX_MODEL_LEN="1",
    )
    import vllm.models.kimi_k3.nvidia.model as target
    import vllm.v1.worker.gpu.spec_decode.dflash.speculator as dflash

    # The target decodes stock at TP2 (its specialized decode serves TP 1/8/16);
    # the draft model still runs the specialized drafter (registered below).
    target.specialized_decode_config = lambda vllm_config: None
    from vllm.models.kimi_k3.nvidia.specialized_decode.drafter.model import register

    register()  # what the bypassed specialized_decode_config does for the drafter
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

    dispatch = dflash.dispatch_cg_and_sync_dp

    def eager_dispatch(*args, **kwargs):  # every draft step eager (then captured)
        return dispatch(*args, **{**kwargs, "need_eager": True})

    dflash.dispatch_cg_and_sync_dp = eager_dispatch
    generate = DSparkSpeculator._generate_draft
    records: list = []

    def compared(self, num_reqs, *args, **kwargs):
        model = self.model
        backbone = model.model
        comm = getattr(model, "_drafter_comm", None)
        if torch.cuda.is_current_stream_capturing() or comm is None:
            return generate(self, num_reqs, *args, **kwargs)
        saved = (backbone.specialized_forward, backbone.context_proj.comm)
        backbone.specialized_forward = None
        backbone.context_proj.comm = None
        model._drafter_comm = None
        try:
            generate(self, num_reqs, *args, **kwargs)
        finally:
            backbone.specialized_forward, backbone.context_proj.comm = saved
            model._drafter_comm = comm
        stock = self.draft_tokens[:num_reqs].clone()
        stock_logits = self.draft_logits.values.float().clone()
        generate(self, num_reqs, *args, **kwargs)
        ours = self.draft_tokens[:num_reqs]
        logits = self.draft_logits.values.float()
        touched = stock_logits != 0  # the cache rows this step wrote
        num = (logits - stock_logits)[touched].pow(2).sum().sqrt()
        den = stock_logits[touched].pow(2).sum().sqrt().clamp_min(1e-30)
        records.append(
            dict(
                reqs=num_reqs,
                equal=int((ours == stock).sum()),
                total=int(stock.numel()),
                logits_rel_rms=float(num / den),
                hooked=backbone.specialized_forward is not None,
                dcp_mailbox=getattr(model, "_drafter_dcp", None) is not None,
            )
        )
        return None

    DSparkSpeculator._generate_draft = compared  # type: ignore[method-assign]
    import vllm.benchmarks.forward.main as fm
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    spec = {
        "method": "dspark",
        "num_speculative_tokens": 4,
        "model": DRAFT,
        "draft_sample_method": "probabilistic",
    }
    parallel = ["-tp", "2"]
    if dcp:
        parallel += ["--decode-context-parallel-size", "2"]
        parallel += ["--dcp-comm-backend", "a2a", "--cp-kv-cache-interleave-size", "1"]
    argv = [
        "--model", MODEL, "--load-format", "dummy", "--trust-remote-code",
        *parallel, "--routing", "balanced",
        "--kv-cache-dtype", "fp8", "--mamba-cache-dtype", "float32",
        "--mamba-ssm-cache-dtype", "float32",
        "--max-model-len", "8192", "--language-model-only",
        "--gpu-memory-utilization", "0.8", "--max-num-seqs", "16",
        "--speculative-config", json.dumps(spec),
        "--additional-config", json.dumps({"decode_backend": "latency_specialized"}),
        "--kernel-config", json.dumps({"enable_flashinfer_autotune": False}),
        "--kv-len-jitter", "0", "--iters", "2",
        "--samples", "1",
    ]  # fmt: skip
    for case in STEP_CASES:
        argv += ["--case", case]
    parser = FlexibleArgumentParser()
    fm.add_cli_args(parser)
    fm.main(parser.parse_args(argv))
    with open(os.path.join(out_dir, f"drafts{rank}.json"), "w") as f:
        json.dump(records, f)


@pytest.mark.skipif(
    not all(os.path.isdir(p or "") for p in (MODEL, DRAFT)),
    reason="needs a layer-truncated K3.1 config (KIMI_K31_DEP_MODEL) and its "
    "DSpark draft config (KIMI_K31_DEP_DRAFT)",
)
@pytest.mark.parametrize("dcp", [False, True], ids=["tp2", "dcp2"])
def test_draft_step_matches_stock(tmp_path, dcp):
    import json
    import time

    port = get_open_port()
    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(target=_step_worker, args=(r, port, str(tmp_path), dcp))
        for r in range(2)
    ]
    for p in procs:
        p.start()
    deadline = time.monotonic() + 1200
    for p in procs:
        p.join(timeout=max(0.0, deadline - time.monotonic()))
    for p in procs:
        if p.is_alive():
            p.kill()
            p.join()
    assert [p.exitcode for p in procs] == [0, 0]
    for rank in range(2):
        records = json.loads((tmp_path / f"drafts{rank}.json").read_text())
        print(f"rank {rank}: {records}")
        assert {r["reqs"] for r in records} >= {1, 4, 12}, records
        assert all(r["hooked"] for r in records), records
        # DCP2 must run the draft attention on the DCP mailboxes (DraftDcp).
        assert all(r["dcp_mailbox"] == dcp for r in records), records
        equal = sum(r["equal"] for r in records)
        total = sum(r["total"] for r in records)
        # The collectives change only the order of FP32 sums before one BF16
        # rounding: logits agree to BF16 noise and draft ids flip only at
        # near-ties of the Gumbel race.
        assert equal >= 0.97 * total, (equal, total)
        for r in records:
            assert r["logits_rel_rms"] < 2e-2, r
