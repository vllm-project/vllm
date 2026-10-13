# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash KDA recurrent (decode) kernel.

The decode path hands the kernel column slices of the merged ``q|k|v`` conv
output and of the fused ``qkvbfg_a`` projection (beta), so q/k/v/beta are
token-strided rather than contiguous. Both backends must match a pure-PyTorch
recurrence, read these slices in place and reject unsupported layouts.
"""

import pytest
import torch

from vllm.platforms import current_platform

if current_platform.is_rocm():
    from vllm.models.glm5next.amd.ops.third_party.kda import fused_recurrent_kda
else:
    from vllm.models.glm5next.nvidia.ops.third_party.kda import fused_recurrent_kda

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="Requires CUDA or ROCm"
)

H, D = 16, 128
LOWER_BOUND = -5.0


def naive_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    a_log: torch.Tensor,
    g_bias: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 reference for one sequence: ``[T, H, D]`` inputs, ``[H, D, D]``
    (v-major) state; mirrors the kernel's in-kernel gate, beta sigmoid and
    q/k l2norm.
    """
    q, k, v, raw_g, raw_beta = (x.float() for x in (q, k, v, raw_g, raw_beta))
    q = q / torch.sqrt(q.square().sum(-1, keepdim=True) + 1e-6) * D**-0.5
    k = k / torch.sqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    gate = LOWER_BOUND * torch.sigmoid(a_log.exp()[:, None] * (raw_g + g_bias))
    beta = torch.sigmoid(raw_beta)
    s = state.clone()
    out = torch.empty_like(v)
    for t in range(q.shape[0]):
        s = s * gate[t].exp()[:, None, :]
        u = beta[t][:, None] * (v[t] - torch.einsum("hvk,hk->hv", s, k[t]))
        s = s + u[:, :, None] * k[t][:, None, :]
        out[t] = torch.einsum("hvk,hk->hv", s, q[t])
    return out, s


def make_inputs(num_seqs: int, query_len: int, device: torch.device):
    """Token-strided q/k/v/beta as the decode path produces them: column
    slices of a merged ``[T, q|k|v]`` conv output and of the fused
    ``[T, qkv|beta|f_a|g_a]`` projection.
    """
    T, proj = num_seqs * query_len, H * D
    qkv = torch.randn(T, 3 * proj, dtype=torch.bfloat16, device=device)
    projected = torch.randn(
        T, 3 * proj + H + 2 * D, dtype=torch.bfloat16, device=device
    )
    q, k, v = (x.view(1, T, H, D) for x in qkv.split(proj, dim=-1))
    beta = projected[:, 3 * proj : 3 * proj + H].unsqueeze(0)
    # (A size-1 token dim gets an arbitrary stride from `view`.)
    assert T == 1 or (q.stride(1) == 3 * proj and beta.stride(1) == projected.stride(0))
    inputs = dict(
        q=q,
        k=k,
        v=v,
        g=torch.randn(1, T, H, D, dtype=torch.bfloat16, device=device),
        beta=beta,
        a_log=0.5 * torch.randn(H, dtype=torch.float32, device=device),
        g_bias=0.1 * torch.randn(H * D, dtype=torch.float32, device=device),
        cu_seqlens=torch.arange(0, T + 1, query_len, dtype=torch.int32, device=device),
    )
    # Slot 0 is NULL_BLOCK_ID; sequences own random distinct slots (one per
    # token in the spec-decode layout).
    slots = torch.randperm(T, device=device).to(torch.int32) + 1
    if query_len == 1:
        inputs["ssm_state_indices"] = slots
    else:
        inputs["ssm_state_indices"] = slots.view(num_seqs, query_len)
        inputs["num_accepted_tokens"] = torch.randint(
            1, query_len + 1, (num_seqs,), dtype=torch.int32, device=device
        )
    state = torch.randn(T + 1, H, D, D, dtype=torch.float32, device=device)
    return inputs, state


def run_kernel(inputs: dict, state: torch.Tensor) -> torch.Tensor:
    out, _ = fused_recurrent_kda(
        **inputs,
        initial_state=state,
        use_qk_l2norm_in_kernel=True,
        sigmoid_beta=True,
        compute_gate=True,
        lower_bound=LOWER_BOUND,
    )
    return out


@pytest.mark.parametrize(
    ("num_seqs", "query_len"), [(1, 1), (7, 1), (3, 3)], ids=["1x1", "7x1", "3x3"]
)
@torch.inference_mode()
def test_fused_recurrent_kda_matches_reference(num_seqs: int, query_len: int):
    torch.manual_seed(0)
    device = torch.device("cuda")
    inputs, state = make_inputs(num_seqs, query_len, device)
    expected_state = state.clone()
    out = run_kernel(inputs, state)

    indices = inputs["ssm_state_indices"].view(num_seqs, query_len)
    accepted = inputs.get("num_accepted_tokens")
    expected = torch.empty_like(out[0], dtype=torch.float32)
    for n in range(num_seqs):
        first = indices[n, 0 if accepted is None else accepted[n] - 1]
        s = expected_state[first]
        for t in range(query_len):
            tok = slice(n * query_len + t, n * query_len + t + 1)
            expected[tok], s = naive_recurrent_kda(
                inputs["q"][0, tok],
                inputs["k"][0, tok],
                inputs["v"][0, tok],
                inputs["g"][0, tok],
                inputs["beta"][0, tok],
                inputs["a_log"],
                inputs["g_bias"].view(H, D),
                s,
            )
            expected_state[indices[n, t]] = s

    torch.testing.assert_close(out[0].float(), expected, rtol=1e-2, atol=1e-3)
    used = indices.flatten().long()
    torch.testing.assert_close(state[used], expected_state[used], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    ("num_seqs", "query_len"), [(7, 1), (3, 3)], ids=["7x1", "3x3"]
)
@torch.inference_mode()
def test_fused_recurrent_kda_strided_inputs_bit_identical_to_contiguous(
    num_seqs: int, query_len: int
):
    torch.manual_seed(0)
    device = torch.device("cuda")
    inputs, state = make_inputs(num_seqs, query_len, device)
    for name in ("q", "k", "v", "beta"):
        assert not inputs[name].is_contiguous()
    contiguous = {
        name: x.contiguous() if name in ("q", "k", "v", "beta") else x
        for name, x in inputs.items()
    }
    state_ref = state.clone()
    out_ref = run_kernel(contiguous, state_ref)
    out = run_kernel(inputs, state)
    torch.testing.assert_close(out, out_ref, rtol=0, atol=0)
    torch.testing.assert_close(state, state_ref, rtol=0, atol=0)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="Both implementations require zero-copy token-strided layouts",
)
@torch.inference_mode()
def test_fused_recurrent_kda_rejects_unaddressable_layouts():
    """Layouts the token-stride addressing cannot express must fail loudly
    rather than read the wrong tokens: a batch slice of a wider buffer
    (``stride(0) != T * stride(1)``), overlapping tokens, and a head-strided
    (transposed) block.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    inputs, state = make_inputs(1, 1, device)
    T = 4
    base = torch.randn(4, T, 2 * H * D, dtype=torch.bfloat16, device=device)
    bad_q = {
        "batch slice": base[::2, :, : H * D].view(2, T, H, D),
        "overlapping tokens": base[:1, :, : H * D].as_strided(
            (1, T, H, D), (0, D, D, 1)
        ),
        "head-strided": base[:1, :, : H * D].view(1, T, D, H).transpose(2, 3),
    }
    for q in bad_q.values():
        broken = dict(inputs, q=q, k=q, v=q)
        broken["cu_seqlens"] = None if q.shape[0] > 1 else inputs["cu_seqlens"]
        with pytest.raises(AssertionError, match=r"torch.Size"):
            run_kernel(broken, state)


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="The large-grid fix is NVIDIA-only"
)
@pytest.mark.parametrize("num_seqs,num_heads", [(1024, 64), (2048, 48)])
def test_fused_recurrent_kda_more_than_65535_programs(num_seqs: int, num_heads: int):
    """Regression test for the launch grid.

    GLM-5.3-Flash has 64 KDA value heads; at TP=1 (every DP rank) a decode
    batch of 1024 sequences needs 1024 * 64 = 65536 (sequence, head)
    programs. The kernel used to put that product into gridDim.z, whose CUDA
    limit is 65535, so the launch failed with ``Triton Error [CUDA]: invalid
    argument`` at CUDA-graph capture and DP deployments could not start. The
    grid now carries the product in gridDim.x. Small head dims keep the test
    cheap; the shape that matters is the number of programs. Every (sequence,
    head) pair is independent, so the full batch must be bitwise identical to
    the same batch run in two halves.
    """
    torch.manual_seed(0)
    dev = torch.device("cuda")
    head_dim = 16
    pool = num_seqs + 8
    q = torch.randn(1, num_seqs, num_heads, head_dim, device=dev, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    g = torch.randn_like(q)
    beta = torch.randn(1, num_seqs, num_heads, device=dev, dtype=torch.bfloat16)
    state = torch.randn(
        pool, num_heads, head_dim, head_dim, device=dev, dtype=torch.float32
    )
    cu = torch.arange(0, num_seqs + 1, device=dev, dtype=torch.int32)
    # slot 0 is NULL_BLOCK_ID and is skipped by the kernel: use slots 1..
    idx = (torch.randperm(pool - 1, device=dev)[:num_seqs] + 1).to(torch.int32)
    a_log = torch.randn(num_heads, device=dev, dtype=torch.float32)
    g_bias = torch.randn(num_heads, head_dim, device=dev, dtype=torch.float32)

    def run(sl: slice, cu_: torch.Tensor, st: torch.Tensor, out: torch.Tensor):
        fused_recurrent_kda(
            q=q[:, sl],
            k=k[:, sl],
            v=v[:, sl],
            g=g[:, sl],
            beta=beta[:, sl],
            initial_state=st,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_,
            ssm_state_indices=idx[sl],
            sigmoid_beta=True,
            a_log=a_log,
            g_bias=g_bias,
            compute_gate=True,
            lower_bound=-5.0,
            out=out,
        )

    full_state = state.clone()
    full_out = torch.zeros_like(k)
    run(slice(0, num_seqs), cu, full_state, full_out)

    half = num_seqs // 2
    ref_state = state.clone()
    ref_out = torch.zeros_like(k)
    for s, e in ((0, half), (half, num_seqs)):
        cu_h = torch.arange(0, e - s + 1, device=dev, dtype=torch.int32)
        part = torch.zeros_like(k[:, s:e])
        run(slice(s, e), cu_h, ref_state, part)
        ref_out[:, s:e] = part
    torch.accelerator.synchronize()
    assert torch.equal(full_out, ref_out)
    assert torch.equal(full_state, ref_state)


@pytest.mark.parametrize("offset", [16, 144, 256])
@pytest.mark.parametrize("dim_first", [False, True])
@pytest.mark.parametrize("num_spec", [0, 3])
@torch.inference_mode()
def test_prefill_checkpoint_resumes_suffix(monkeypatch, dim_first, num_spec, offset):
    """Restoring both cached states must reproduce an uninterrupted prefill."""
    from types import SimpleNamespace

    from vllm.model_executor.layers.mamba.checkpoint import (
        MambaPrefillCheckpointMetadata,
    )
    from vllm.model_executor.layers.mamba.kda_checkpoint import (
        FlashKDAPrefillCheckpointExporter,
    )
    from vllm.models.glm5next.common import kda
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
    from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata
    from vllm.v1.worker.workspace import WorkspaceManager

    if kda._resolve_kda_prefill_backend("auto", D, torch.bfloat16, -5) != "flashkda":
        pytest.skip("requires FlashKDA")
    import vllm._flashkda_C  # noqa: F401

    torch.manual_seed(42)
    device = torch.device("cuda")
    length, heads = 273, 2
    width, state_len = 3 * heads * D, 3 + num_spec
    layer = object.__new__(kda.Glm5NextLinearAttention)
    torch.nn.Module.__init__(layer)
    layer.prefix = "layer.0"
    layer.local_num_heads = heads
    layer.head_dim = D
    layer.conv_size = 4
    layer.local_projection_size = heads * D
    layer.kda_safe_gate = True
    layer.kda_lower_bound = -5.0
    layer.use_recoverssm = False
    layer.spec_query_len = num_spec + 1
    layer.kda_prefill_backend = "flashkda"
    layer._conv_state_dim_first = dim_first
    layer._merged_conv_weight = torch.randn(width, 4, device=device) * 0.1
    layer.q_conv1d = SimpleNamespace(bias=None)
    layer.A_log = torch.randn(1, 1, heads, 1, device=device) * 0.1
    layer.dt_bias = torch.randn(heads * D, device=device) * 0.1
    state_shape = (2, heads, D, D)
    layer._flashkda_buffer_specs = (
        (state_shape, torch.float32),
        (state_shape, torch.float32),
        (
            (torch.ops._flashkda_C.get_workspace_size(2 * length, heads, 2),),
            torch.uint8,
        ),
        ((1, 2 * length, heads, D), torch.bfloat16),
    )
    layer._checkpoint_exporter = FlashKDAPrefillCheckpointExporter(state_len=3)
    workspace = WorkspaceManager(device)
    monkeypatch.setattr(kda, "current_workspace_manager", lambda: workspace)
    conv_shape = (width, state_len) if dim_first else (state_len, width)
    num_slots = 5 + (num_spec + 1 if num_spec else 0)
    conv_storage = torch.zeros(
        num_slots, width * state_len + 8, device=device, dtype=torch.bfloat16
    )
    conv = conv_storage[:, : width * state_len].view(num_slots, *conv_shape)
    recurrent_storage = torch.zeros(
        num_slots, heads * D * D + 8, device=device, dtype=torch.float32
    )
    recurrent = recurrent_storage[:, : heads * D * D].view(num_slots, heads, D, D)
    layer.kv_cache = (conv, recurrent)
    projected = torch.randn(
        2 * length, width + heads, device=device, dtype=torch.bfloat16
    )
    raw_qkv, beta = projected[:, :width], projected[:, width:].unsqueeze(0)
    raw_before = raw_qkv.clone()
    gate = torch.randn(1, 2 * length, heads, D, device=device, dtype=torch.bfloat16)

    def run(qkv, g, b, slots, has_initial, checkpoint=None):
        n = qkv.shape[0] // 2
        cu_cpu = torch.tensor([0, n, 2 * n], dtype=torch.int32)
        nums, batch_ptr, chunk_offsets = compute_causal_conv1d_metadata(
            cu_cpu, device=device
        )
        metadata = GDNAttentionMetadata(
            num_prefills=2,
            num_prefill_tokens=2 * n,
            num_decodes=0,
            num_decode_tokens=0,
            num_spec_decodes=0,
            num_spec_decode_tokens=0,
            num_actual_tokens=2 * n,
            has_initial_state=torch.tensor(has_initial, device=device),
            non_spec_query_start_loc=cu_cpu.to(device),
            non_spec_state_indices_tensor=torch.tensor(
                slots, dtype=torch.int32, device=device
            ),
            checkpoint=checkpoint,
            nums_dict=nums,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=chunk_offsets,
        )
        monkeypatch.setattr(
            kda,
            "get_forward_context",
            lambda: SimpleNamespace(attn_metadata={layer.prefix: metadata}),
        )
        spec_tokens = num_spec + 1 if num_spec else 0
        if spec_tokens:
            qkv = torch.cat([qkv[:spec_tokens], qkv])
            g = torch.cat([g[:, :spec_tokens], g], dim=1)
            b = torch.cat([b[:, :spec_tokens], b], dim=1)
            metadata.num_spec_decodes = 1
            metadata.num_spec_decode_tokens = spec_tokens
            metadata.num_actual_tokens += spec_tokens
            metadata.spec_sequence_masks = torch.tensor(
                [True, False, False], device=device
            )
            metadata.spec_query_start_loc = torch.tensor(
                [0, spec_tokens], dtype=torch.int32, device=device
            )
            metadata.spec_state_indices_tensor = torch.arange(
                5, num_slots, dtype=torch.int32, device=device
            ).unsqueeze(0)
            metadata.spec_token_indx = torch.arange(spec_tokens, device=device)
            metadata.non_spec_token_indx = torch.arange(
                spec_tokens, qkv.shape[0], device=device
            )
            metadata.num_accepted_tokens = torch.ones(
                1, dtype=torch.int32, device=device
            )
        out = torch.empty(
            1, qkv.shape[0], heads, D, device=device, dtype=torch.bfloat16
        )
        layer._forward(qkv, g, b, out)
        return out[:, spec_tokens:]

    checkpoint = MambaPrefillCheckpointMetadata(
        torch.tensor([offset, 0], dtype=torch.int32, device=device),
        torch.tensor([3, 0], dtype=torch.int32, device=device),
    )
    expected = run(raw_qkv, gate, beta, [1, 2], [False, False], checkpoint)
    expected_state = recurrent[1].clone()
    saved_conv, saved_recurrent = conv[3].clone(), recurrent[3].clone()
    torch.testing.assert_close(raw_qkv, raw_before)
    conv_view = conv if dim_first else conv.transpose(-1, -2)
    torch.testing.assert_close(conv_view[3, :, :3], raw_qkv[offset - 3 : offset].T)
    assert torch.count_nonzero(conv_view[3, :, 3:]) == 0
    assert torch.count_nonzero(conv[0]) == torch.count_nonzero(recurrent[0]) == 0

    run(
        torch.cat([raw_qkv[:offset], raw_qkv[length : length + offset]]),
        torch.cat([gate[:, :offset], gate[:, length : length + offset]], dim=1),
        torch.cat([beta[:, :offset], beta[:, length : length + offset]], dim=1),
        [1, 2],
        [False, False],
    )
    torch.testing.assert_close(saved_recurrent, recurrent[1], atol=0.003, rtol=0.03)

    suffix = slice(offset, length)
    qkv = torch.cat([raw_qkv[suffix], raw_qkv[length + offset :]])
    g = torch.cat([gate[:, suffix], gate[:, length + offset :]], dim=1)
    b = torch.cat([beta[:, suffix], beta[:, length + offset :]], dim=1)
    conv[4].copy_(saved_conv)
    recurrent[4].copy_(saved_recurrent)
    actual = run(qkv, g, b, [3, 4], [True, True])
    torch.testing.assert_close(
        actual[:, : length - offset], expected[:, suffix], atol=0.003, rtol=0.03
    )
    torch.testing.assert_close(recurrent[3], expected_state, atol=0.003, rtol=0.03)


@pytest.mark.parametrize("ssm_cache_dtype", ["float32", "bfloat16"])
def test_recoverssm_state_layout_from_config(ssm_cache_dtype):
    """RecoverSSM appends its correction and key/gate records to the state."""
    from types import SimpleNamespace

    from vllm.models.glm5next.common.model import Glm5NextForCausalLM

    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            hf_text_config=SimpleNamespace(
                linear_num_heads=4, linear_head_dim=32, linear_conv_kernel_dim=4
            ),
        ),
        cache_config=SimpleNamespace(
            mamba_cache_dtype="auto",
            mamba_ssm_cache_dtype=ssm_cache_dtype,
            use_kda_recoverssm=True,
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=2),
        speculative_config=SimpleNamespace(num_speculative_tokens=3),
    )
    assert Glm5NextForCausalLM.get_mamba_state_dtype_from_config(vllm_config) == (
        torch.bfloat16,
        {"float32": torch.float32, "bfloat16": torch.bfloat16}[ssm_cache_dtype],
        torch.float32,
        torch.bfloat16,
    )
    assert Glm5NextForCausalLM.get_mamba_state_shape_from_config(vllm_config)[2:] == (
        (2, 4, 32),
        (2, 4, 64),
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA RecoverSSM")
@pytest.mark.parametrize("dim_first", [False, True])
@pytest.mark.parametrize("query_lens", [(4, 4, 4), (1, 3, 4)])
@torch.inference_mode()
def test_recoverssm_spec_decode_matches_per_draft_slots(
    monkeypatch, dim_first, query_lens
):
    """RecoverSSM keeps one checkpoint slot and commits after sampling. It must
    reproduce the per-draft-slot spec decode: the same verify outputs, and the
    recurrent state and convolution window the baseline reads next step."""
    from types import SimpleNamespace

    from vllm.models.glm5next.common import kda, kda_metadata
    from vllm.models.kimi_k3.nvidia.kda_metadata import KDARecoverSSMCommitMetadata
    from vllm.models.kimi_k3.nvidia.ops import recoverssm
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

    monkeypatch.setattr(recoverssm, "is_conv_state_dim_first", lambda: dim_first)
    torch.manual_seed(7)
    device = torch.device("cuda")
    num_seqs, query_len, heads = 3, 4, 2
    num_tokens = sum(query_lens)
    width, state_len = 3 * heads * D, 3 + query_len - 1

    layer = object.__new__(kda.Glm5NextLinearAttention)
    torch.nn.Module.__init__(layer)
    layer.prefix = "layer.0"
    layer.local_num_heads = heads
    layer.head_dim = D
    layer.local_projection_size = heads * D
    layer.kda_safe_gate = True
    layer.kda_lower_bound = LOWER_BOUND
    layer._conv_state_dim_first = dim_first
    layer._merged_conv_weight = torch.randn(width, 4, device=device) * 0.1
    layer.q_conv1d = SimpleNamespace(bias=None)
    layer.A_log = torch.randn(1, 1, heads, 1, device=device) * 0.1
    layer.dt_bias = torch.randn(heads * D, device=device) * 0.1
    layer.spec_query_len = query_len

    num_slots = 1 + num_seqs * query_len
    conv_shape = (width, state_len) if dim_first else (state_len, width)
    conv = torch.randn(num_slots, *conv_shape, device=device, dtype=torch.bfloat16)
    recurrent = 0.1 * torch.randn(num_slots, heads, D, D, device=device)
    qkv = torch.randn(num_tokens, width, device=device, dtype=torch.bfloat16)
    g1 = torch.randn(1, num_tokens, heads, D, device=device, dtype=torch.bfloat16)
    beta = torch.randn(1, num_tokens, heads, device=device, dtype=torch.bfloat16)
    slots = 1 + torch.arange(
        num_seqs * query_len, dtype=torch.int32, device=device
    ).view(num_seqs, query_len)
    query_start_loc = torch.tensor(
        [0, *query_lens], dtype=torch.int32, device=device
    ).cumsum(0, dtype=torch.int32)

    def run(metadata, kv_cache, use_recoverssm):
        layer.use_recoverssm = use_recoverssm
        layer.kv_cache = kv_cache
        monkeypatch.setattr(
            kda,
            "get_forward_context",
            lambda: SimpleNamespace(attn_metadata={layer.prefix: metadata}),
        )
        out = torch.empty(1, num_tokens, heads, D, device=device, dtype=torch.bfloat16)
        # The spec-decode convolution writes its output over qkv.
        layer._forward(qkv.clone(), g1, beta, out)
        return out

    def spec_metadata(cls, state_indices, num_accepted_tokens, **extra):
        return cls(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decodes=0,
            num_decode_tokens=0,
            num_spec_decodes=num_seqs,
            num_spec_decode_tokens=num_tokens,
            num_actual_tokens=num_tokens,
            spec_query_start_loc=query_start_loc,
            spec_state_indices_tensor=state_indices,
            spec_sequence_masks=torch.ones(num_seqs, dtype=torch.bool, device=device),
            num_accepted_tokens=num_accepted_tokens,
            **extra,
        )

    base_cache = (conv.clone(), recurrent.clone())
    rss_cache = (
        conv.clone(),
        recurrent.clone(),
        torch.empty(num_slots, heads, query_len, D, device=device),
        torch.empty(
            num_slots, heads, query_len, 2 * D, device=device, dtype=torch.bfloat16
        ),
    )
    layer.kv_cache = rss_cache
    context = recoverssm.KDARecoverSSMCommitContext.create(
        [kda_metadata.KDARecoverSSMLayerView.of(layer)],
        spec_query_len=query_len,
        max_num_reqs=num_seqs,
    )
    ones = torch.ones(num_seqs, dtype=torch.int32, device=device)
    metadata = spec_metadata(
        kda_metadata.Glm5NextKDAMetadata,
        slots[:, :1],
        ones,
        recoverssm_commit=KDARecoverSSMCommitMetadata(
            state_indices=slots[:, :1],
            query_start_loc=query_start_loc,
            request_indices=None,
            align=None,
        ),
        recoverssm_context=context,
    )
    previous_accepted = ones
    for step in range(query_len):
        qkv.normal_()
        g1.normal_()
        beta.normal_()
        expected = run(
            spec_metadata(GDNAttentionMetadata, slots, previous_accepted),
            base_cache,
            False,
        )
        checkpoint = rss_cache[1].clone()
        actual = run(metadata, rss_cache, True)
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-3)
        torch.testing.assert_close(rss_cache[1], checkpoint, rtol=0, atol=0)

        accepted = torch.tensor(query_lens, dtype=torch.int32, device=device)
        accepted.clamp_max_(step + 1)
        assert metadata.commit_recoverssm_state(accepted) is None
        base_conv, rss_conv = base_cache[0], rss_cache[0]
        if not dim_first:
            base_conv, rss_conv = (
                base_conv.transpose(-1, -2),
                rss_conv.transpose(-1, -2),
            )
        for n, length in enumerate(accepted.tolist()):
            slot = slots[n, 0]
            torch.testing.assert_close(
                rss_cache[1][slot],
                base_cache[1][slots[n, length - 1]],
                rtol=1e-5,
                atol=1e-5,
            )
            torch.testing.assert_close(
                rss_conv[slot, :, :3], base_conv[slot, :, length - 1 : length + 2]
            )
        previous_accepted = accepted
