# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU numerics for the AITER ASM cprr DCP-verify route.

Runs the REAL metadata builder (``AiterMLAMetadataBuilder.build``, so
``get_mla_metadata_v1`` is exercised with the cprr kwargs and the persistent
schedule buffers are produced) and the REAL asm kernel, on ONE GPU with the DCP
ranks simulated one after another.

That is legitimate because the aiter MLA backend contains no collective on this
path: the query all-gather happens in the *layer*, strictly before
``forward_mqa``, so the head fold, the round-robin kernel call and the un-fold
are pure per-rank local arithmetic in which ``cp_world_size``/``cp_rank`` are
just numbers feeding the causal-mask maths. Restricting a shard to a residue
class of global positions is exactly what DCP rank r sees.

Three properties are checked:
  * each shard matches an exact torch reference over the positions it holds,
    with causality applied on the GLOBAL position (the thing cprr reconstructs
    in-kernel and the segmented route instead carries in per-row lengths);
  * the LSE merge of all shards reproduces full-context attention;
  * a shard told the wrong ``cp_rank`` fails both -- without this positive
    control a passing run proves nothing, since a kernel that ignored global
    positions still produces plausible-looking output.

Destined for tests/v1/attention/, alongside test_rocm_aiter_mla_dcp_cprr.py.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm AITER MLA tests", allow_module_level=True)

from vllm._aiter_ops import is_aiter_found  # noqa: E402

KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_NOPE_HEAD_DIM = 128
HEAD_SIZE = KV_LORA_RANK + QK_ROPE_HEAD_DIM
SCALE = (QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM) ** -0.5

DCP = 8
NUM_REQS = 8
CTX = 256
QLEN = 5  # 1 + num_speculative_tokens; must clear _MIN_CPRR_QLEN

# fp8-e4m3 keeps 3 mantissa bits, so the relative error floor is ~6.2e-2. Gates
# are relative for the same reason: an absolute bound would just be a bound on
# the output magnitude.
SHARD_RTOL = 1.0e-1
MERGE_RTOL = 5.0e-2

MODEL = "deepseek-ai/DeepSeek-R1"


def _gpu_available() -> bool:
    return is_aiter_found() and torch.accelerator.is_available()


def _on_gfx950() -> bool:
    if not is_aiter_found():
        return False
    from vllm.platforms.rocm import on_gfx950

    return on_gfx950()


requires_cprr = pytest.mark.skipif(
    not (_gpu_available() and _on_gfx950()),
    reason="cprr DCP verify needs ROCm + AITER on gfx950",
)


# --------------------------------------------------------------------------
# reference maths
# --------------------------------------------------------------------------


def _torch_reference(q_f32, kv_f32, ctx, qlen, scale, positions):
    """Exact absorbed-MLA decode over an explicit set of GLOBAL KV positions.

    A shard sees only ``positions``, but causality still uses the global
    position: query token i (global position ctx + i) may attend to any p in
    positions with p <= ctx + i. Restricting ``positions`` to a round-robin
    residue class is precisely DCP rank r's view.
    """
    nr, _, heads, _ = q_f32.shape
    dev = q_f32.device
    o = torch.zeros(nr, qlen, heads, KV_LORA_RANK, dtype=torch.float32, device=dev)
    lse = torch.full((nr, qlen, heads), float("-inf"), dtype=torch.float32, device=dev)
    for r in range(nr):
        kv_sel = kv_f32[r].index_select(0, positions)
        for i in range(qlen):
            valid = positions <= (ctx + i)
            if not bool(valid.any()):
                continue
            k = kv_sel[valid]
            s = (q_f32[r, i] @ k.T) * scale
            m = s.max(dim=-1, keepdim=True).values
            p = torch.exp(s - m)
            denom = p.sum(dim=-1, keepdim=True)
            o[r, i] = (p / denom) @ k[:, :KV_LORA_RANK]
            lse[r, i] = m.squeeze(-1) + torch.log(denom.squeeze(-1))
    return o, lse


def _merge_partials(os_, lses):
    """Cross-shard softmax merge (natural-log LSE)."""
    stack_lse = torch.stack(lses)
    m = stack_lse.max(dim=0).values
    w = torch.exp(stack_lse - m)
    num = (w.unsqueeze(-1) * torch.stack(os_)).sum(dim=0)
    den = w.sum(dim=0).unsqueeze(-1)
    return num / den, m + torch.log(w.sum(dim=0))


# --------------------------------------------------------------------------
# single-process rank simulation
# --------------------------------------------------------------------------


class _FakeGroup:
    def __init__(self, world_size: int, rank: int):
        self.world_size = world_size
        self.rank_in_group = rank
        self.rank = rank
        self.local_rank = rank

    def all_gather(self, tensor, dim=0):
        raise RuntimeError(
            "collective called in a single-process test -- the path under test "
            "is supposed to be collective-free; this indicates a real change"
        )


def _install_fake_groups(dcp_size: int, cp_rank: int, tp_size: int = 1) -> None:
    """Point vLLM's parallel_state at rank-simulating stubs (no NCCL)."""
    import vllm.distributed.parallel_state as ps
    import vllm.model_executor.layers.attention.mla_attention as mla_attention
    import vllm.v1.attention.backends.mla.rocm_aiter_mla as backend

    dcp = _FakeGroup(dcp_size, cp_rank)
    tp = _FakeGroup(tp_size, cp_rank)
    ps.get_dcp_group = lambda: dcp
    ps.get_tp_group = lambda: tp
    # Both of these imported the symbol directly at module import time, so
    # patching parallel_state alone leaves them resolving the real group (and
    # the builder then reports dcp_world_size 1, silently skipping the route).
    backend.get_dcp_group = lambda: dcp
    mla_attention.get_dcp_group = lambda: dcp


@pytest.fixture(autouse=True)
def _restore_parallel_groups():
    """Undo _install_fake_groups after every test.

    It patches module globals in three modules, so without this the fake DCP
    group leaks into the rest of the pytest session and silently breaks
    unrelated tests that expect a real (or absent) group.
    """
    import vllm.distributed.parallel_state as ps
    import vllm.model_executor.layers.attention.mla_attention as mla_attention
    import vllm.v1.attention.backends.mla.rocm_aiter_mla as backend

    saved = [
        (mod, name, getattr(mod, name))
        for mod, name in (
            (ps, "get_dcp_group"),
            (ps, "get_tp_group"),
            (backend, "get_dcp_group"),
            (mla_attention, "get_dcp_group"),
        )
    ]
    yield
    for mod, name, value in saved:
        setattr(mod, name, value)


class _PrefillBackendStub:
    """The base builder does ``attention_layer.prefill_backend.clone()``.

    prefill_backend is consulted only from the chunked-context prefill path,
    which a decode-only test never enters.
    """

    def clone(self):
        return self


def _make_ctx_layer(vllm_config):
    """Stand-in for the registered attention layer in static_forward_context.

    ``MLACommonMetadataBuilder.__init__`` reads the MLA dims, ``dcp_manager``
    and ``prefill_backend`` off this object. ``dcp_manager`` is the Direct-DCP
    symmetric-memory workspace, used ONLY by the chunked-prefill KV all-gather;
    it is constructed without running __init__ so no peer pointers are exchanged
    and no collective is issued. That is what keeps this test single-process.
    """
    from vllm.v1.attention.ops.dcp import MLADCPManager

    mgr = object.__new__(MLADCPManager)
    mgr.init_kv_gather = lambda *a, **k: None

    return SimpleNamespace(
        non_causal_multi_token_decode=False,
        q_lora_rank=None,
        kv_lora_rank=KV_LORA_RANK,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        v_head_dim=QK_NOPE_HEAD_DIM,
        dcp_manager=mgr,
        prefill_backend=_PrefillBackendStub(),
    )


def _ensure_workspace_manager(device: torch.device) -> None:
    """Init the global workspace manager, which GPUModelRunner does in production.

    ``AiterMLAMetadataBuilder.__init__`` reserves the fp8 prefill
    persistent-scheduling scratch through ``current_workspace_manager()``, so
    merely constructing a builder outside a serve trips its assert. Idempotent:
    one builder is constructed per DCP shard. Deliberately does NOT lock the
    workspace -- production locks only after warmup, and this keeps allocating
    across shards.
    """
    from vllm.v1.worker.workspace import (
        init_workspace_manager,
        is_workspace_manager_initialized,
    )

    if not is_workspace_manager_initialized():
        init_workspace_manager(device)


class _Layer:
    def __init__(self, device):
        self._q_scale = torch.tensor(1.0, device=device)
        self._k_scale = torch.tensor(1.0, device=device)
        self._v_scale = torch.tensor(1.0, device=device)


def _build_impl(num_heads: int, dcp: int, cp_rank: int):
    """Real AiterMLAImpl.forward_mqa bound to a minimally-populated instance.

    Constructing the full impl would drag in kv_b_proj and the rest of the
    weight-dependent MLA setup -- i.e. model loading, which this test exists to
    avoid. forward_mqa's fold path reads only the attributes set below.
    """
    from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLAImpl

    impl = object.__new__(AiterMLAImpl)
    impl.kv_lora_rank = KV_LORA_RANK
    impl.qk_rope_head_dim = QK_ROPE_HEAD_DIM
    impl.num_heads = num_heads
    impl.dcp_world_size = dcp
    impl.dcp_rank = cp_rank
    impl.pcp_world_size = 1
    impl.kv_cache_dtype = "fp8"
    impl.scale = SCALE
    # `_decode_num_heads` is a read-only property deriving num_heads *
    # dcp_world_size, both set above; assert rather than assign.
    assert impl._decode_num_heads == num_heads * dcp
    return impl


def _make_vllm_config(heads_per_rank: int):
    """A DCP8 MLA config with no weights and no distributed init.

    DeepSeek-R1's config supplies the MLA dims (kv_lora_rank 512, rope 64,
    nope 128) that the aiter asm kernel is built for; only the head count is
    overridden, to place the GATHERED count (heads_per_rank * DCP) on the
    kernel variant under test. ``get_num_attention_heads`` reads
    model_arch_config, a snapshot taken at config-creation time, so patching
    hf_config alone would be inert.
    """
    from tests.v1.attention.utils import create_vllm_config

    vllm_config = create_vllm_config(
        model_name=MODEL,
        tensor_parallel_size=1,
        max_model_len=2048,
        max_num_seqs=NUM_REQS * 2,
        max_num_batched_tokens=4096,
    )
    vllm_config.model_config.model_arch_config.total_num_attention_heads = (
        heads_per_rank
    )
    # Without a speculative config the batch reorder threshold stays at 1 and a
    # qlen-QLEN row is classified as a PREFILL, so the decode path under test is
    # never reached. _init_reorder_batch_threshold reads exactly these two
    # fields, so a stand-in is enough and avoids pulling in a draft model.
    vllm_config.speculative_config = SimpleNamespace(
        num_speculative_tokens=QLEN - 1, parallel_drafting=False
    )
    vllm_config.cache_config.cache_dtype = "fp8"
    vllm_config.parallel_config.decode_context_parallel_size = DCP
    vllm_config.parallel_config.cp_kv_cache_interleave_size = 1
    return vllm_config


def _run_shard(
    vllm_config,
    spec,
    layer_name,
    device,
    cp_rank,
    kern_rank,
    q_flat,
    kv_flat,
    nr,
    qlen,
    ctx,
    dcp,
    pos,
):
    """Build real metadata for one DCP rank and run the real asm kernel.

    ``kern_rank`` is separate from ``cp_rank`` so the positive control can hand
    rank r's pages to the kernel while telling it that it is rank r+1.
    """
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLAMetadataBuilder

    L = ctx + qlen
    local_len = int(pos.numel())

    # One GLOBAL kv cache: page (req*L + p) holds request req position p. Shards
    # select pages by residue class, so every shard reads the same underlying
    # bytes and no re-quantisation noise is introduced between arms.
    block_table = (
        torch.arange(nr, device=device).view(nr, 1) * L + pos.view(1, -1)
    ).to(torch.int32)

    # seq_lens is the GLOBAL context; dcp_local_seq_lens is what this rank
    # actually holds. Passing the global length as the local one walks the
    # kernel off the end of the page table.
    seq_lens_cpu = torch.full((nr,), L, dtype=torch.int32)
    local_cpu = torch.full((nr,), local_len, dtype=torch.int32)

    common = CommonAttentionMetadata(
        query_start_loc=torch.arange(
            0, nr * qlen + 1, qlen, dtype=torch.int32, device=device
        ),
        query_start_loc_cpu=torch.arange(0, nr * qlen + 1, qlen, dtype=torch.int32),
        seq_lens=seq_lens_cpu.to(device),
        num_reqs=nr,
        num_actual_tokens=nr * qlen,
        max_query_len=qlen,
        max_seq_len=L,
        block_table_tensor=block_table,
        slot_mapping=torch.arange(nr * qlen, dtype=torch.int64, device=device),
        causal=True,
        dcp_local_seq_lens=local_cpu.to(device),
        dcp_local_seq_lens_cpu_upper_bound=local_cpu,
    )

    _install_fake_groups(dcp, kern_rank)
    builder = AiterMLAMetadataBuilder(spec, [layer_name], vllm_config, device)
    md = builder.build(0, common)

    impl = _build_impl(builder.num_heads, dcp, kern_rank)
    o, lse = impl.forward_mqa(q_flat, kv_flat, md, _Layer(device))
    torch.accelerator.synchronize()
    return md, o, lse, local_len


def _numerics(heads_per_rank: int, sabotage: bool = False):
    """Run every DCP shard and return (per-shard max rel err, merged rel err)."""
    import aiter

    from vllm.config import set_current_vllm_config

    device = torch.device("cuda", 0)
    torch.accelerator.set_device_index(device.index)
    torch.manual_seed(0)
    _ensure_workspace_manager(device)

    _install_fake_groups(DCP, 0)
    vllm_config = _make_vllm_config(heads_per_rank)

    with set_current_vllm_config(vllm_config):
        from vllm.v1.kv_cache_interface import MLAAttentionSpec

        layer_name = "model.layers.0.self_attn.attn"
        vllm_config.compilation_config.static_forward_context[layer_name] = (
            _make_ctx_layer(vllm_config)
        )
        # block_size 1: one page per token, so page ids are positions and the
        # residue-class page table is expressible directly.
        spec = MLAAttentionSpec(
            block_size=1,
            num_kv_heads=1,
            head_size=HEAD_SIZE,
            dtype=vllm_config.model_config.dtype,
            non_causal_multi_token_decode=False,
        )

        nr, ctx, qlen = NUM_REQS, CTX, QLEN
        L = ctx + qlen
        gathered_heads = heads_per_rank * DCP
        kv_dtype = aiter.dtypes.fp8

        kv_flat = torch.randn(
            nr * L, 1, HEAD_SIZE, dtype=torch.float32, device=device
        ).to(kv_dtype)
        q_flat = (
            torch.randn(
                nr * qlen,
                gathered_heads,
                HEAD_SIZE,
                dtype=torch.float32,
                device=device,
            )
            .mul_(0.5)
            .to(kv_dtype)
        )
        # Dequantise for the reference so the comparison measures the kernel,
        # not the fp8 cast that both sides share.
        kv_f32 = kv_flat.float().view(nr, L, HEAD_SIZE)
        q_f32 = q_flat.float().view(nr, qlen, gathered_heads, HEAD_SIZE)

        ref_o, _ = _torch_reference(
            q_f32, kv_f32, ctx, qlen, SCALE, torch.arange(L, device=device)
        )

        shard_o, shard_lse = [], []
        max_shard_rel = 0.0
        for r in range(DCP):
            pos = torch.arange(r, L, DCP, device=device)  # residue class r
            kern_rank = (r + 1) % DCP if sabotage else r
            md, o, lse, _ = _run_shard(
                vllm_config,
                spec,
                layer_name,
                device,
                r,
                kern_rank,
                q_flat,
                kv_flat,
                nr,
                qlen,
                ctx,
                DCP,
                pos,
            )
            # The route actually under test must have been taken; a silent
            # fall-out to Triton/segmented would otherwise pass quietly.
            assert md.decode.asm_decode_num_heads, "cprr asm route was not taken"

            o = o.float().view(nr, qlen, gathered_heads, KV_LORA_RANK)
            shard_lse.append(lse.float().view(nr, qlen, gathered_heads))
            shard_o.append(o)

            po, _ = _torch_reference(q_f32, kv_f32, ctx, qlen, SCALE, pos)
            rel = (o - po).abs().max().item() / max(po.abs().max().item(), 1e-6)
            max_shard_rel = max(max_shard_rel, rel)

        merged, _ = _merge_partials(shard_o, shard_lse)
        merge_rel = (merged - ref_o).abs().max().item() / max(
            ref_o.abs().max().item(), 1e-6
        )
        return max_shard_rel, merge_rel


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------


@requires_cprr
@pytest.mark.parametrize(
    "heads_per_rank",
    [
        16,  # 128 gathered heads -- a NATIVE cprr kernel, no padding
        12,  # 96 gathered heads  -- K3's DSpark target, must pad up to 128
    ],
    ids=["native-128", "padded-96-to-128"],
)
@torch.inference_mode()
def test_cprr_shards_match_reference_and_merge_to_full_context(heads_per_rank):
    """Every DCP shard must be right, and their LSE merge must reproduce
    full-context attention.

    A shard that ignored global positions (applying a local causal tail
    instead) still produces plausible-looking output; it only diverges from the
    reference on the tokens whose causal window is truncated. Checking the merge
    alone would hide that, so both are asserted.
    """
    shard_rel, merge_rel = _numerics(heads_per_rank)
    assert shard_rel < SHARD_RTOL, f"per-shard rel err {shard_rel:.3e}"
    assert merge_rel < MERGE_RTOL, f"merged rel err {merge_rel:.3e}"


@requires_cprr
@torch.inference_mode()
def test_wrong_cp_rank_breaks_the_result():
    """Positive control: without it, a passing run above proves nothing.

    Feeding shard r's pages to the kernel while telling it that it is rank r+1
    shifts every reconstructed global position by one, so both the causal mask
    and the round-robin reconstruction are wrong. If the error does NOT blow up,
    the kernel is not using cp_rank and the test above has no power.
    """
    shard_rel, merge_rel = _numerics(16, sabotage=True)
    assert shard_rel > SHARD_RTOL or merge_rel > MERGE_RTOL, (
        f"sabotaged run still passed (shard {shard_rel:.3e}, merge "
        f"{merge_rel:.3e}) -- cp_rank is not reaching the kernel"
    )


@requires_cprr
def test_cprr_kernel_is_selected_for_the_k3_shape():
    """The K3 DSpark target shape must actually reach the cprr route.

    Guards against a silent fall-out to segmented/Triton, which is what the
    stock configuration does and what this route exists to avoid.
    """
    from vllm.v1.attention.backends.mla import rocm_aiter_mla as m

    assert (
        m._asm_dcp_verify_configured(
            dcp_world_size=DCP, cp_interleave=1, multi_token_decode=True
        )
        is True
    )
    assert m._asm_dcp_verify_selected(96) is True
    assert QLEN >= m._MIN_CPRR_QLEN
    assert m._asm_dcp_verify_heads(96) in m._NATIVE_CPRR_HEADS
