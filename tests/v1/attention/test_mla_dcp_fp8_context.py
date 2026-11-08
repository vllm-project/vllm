# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fp8 MLA DCP chunked-context path against the dequantizing path.

The row gather must dequantize fp8 rows exactly as the default path's
dequantizing cache gather does. Both paths then run the real chunked-context
loop: the local gather from a paged fp8 cache, the DCP all-gather and the row
gather. The fp8 path must hand ``kv_b_proj`` and the prefill backend the same
bytes as the default path.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from tests.utils import multi_gpu_only
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonMetadataBuilder,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.tokenspeed_mla import TokenspeedMLAMetadataBuilder
from vllm.v1.attention.ops.mla_dcp_context import gather_dcp_context_rows

_KV_LORA_RANK = 512
_QK_ROPE = 64
_QK_NOPE = 128
_V_HEAD_DIM = 128
_ENTRY = _KV_LORA_RANK + _QK_ROPE
_NUM_HEADS = 2
_BLOCK_SIZE = 64
_DCP = 2
_K_SCALES = (1.0, 0.37, 300.0)
# (interleave, context lengths, workspace rows): single- and multi-chunk
# batches, with empty contexts and a rank that holds no rows of a request.
_LAYOUTS = (
    (1, [145, 37], 1024),
    (64, [145, 37], 1024),
    (1, [545, 0, 3, 0], 256),
    (64, [545, 0, 3, 129], 256),
)


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "k_scale",
    [1.0, 0.37, 300.0, pytest.param(2.0**-120, id="subnormal-products")],
)
# The dequantizing cache gather takes rows of 320, 512 or 576 values; 448 + 64
# has a latent width that is not a power of two and 512 + 0 is a NoPE layer.
@pytest.mark.parametrize("dims", [(512, 64), (448, 64), (512, 0)])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("num_rows", [0, 1, 33, 1000])
def test_gather_dcp_context_rows_dequantizes_like_the_cache_gather(
    num_rows, index_dtype, dims, k_scale, out_dtype
):
    """The rows come back exactly as the default path's dequantizing cache
    gather writes them, for any k_scale, either workspace dtype and a latent
    width that is not a power of two. The rows cover every fp8 code, NaN
    included."""
    from vllm import _custom_ops as ops

    kv_lora_rank, rope_dim = dims
    width = kv_lora_rank + rope_dim
    generator = torch.Generator(device="cuda").manual_seed(num_rows)
    raw = torch.randint(
        0, 256, (1200, width), dtype=torch.uint8, device="cuda", generator=generator
    )
    gathered = raw.view(current_platform.fp8_dtype())
    row_indices = torch.randperm(1200, device="cuda", generator=generator)
    row_indices = row_indices[:num_rows].to(index_dtype)
    # 0-dim, as layers hold their k_scale.
    scale = torch.tensor(k_scale, dtype=torch.float32, device="cuda")

    latent, k_pe = gather_dcp_context_rows(
        gathered, row_indices, kv_lora_rank, k_scale=scale, out_dtype=out_dtype
    )

    # Dequantize every row with the default path's gather, reading the rows
    # as a paged cache of 75 blocks of 16.
    dequantized = torch.empty((1200, width), dtype=out_dtype, device="cuda")
    ops.gather_and_maybe_dequant_cache(
        src_cache=gathered.view(75, 16, width),
        dst=dequantized,
        block_table=torch.arange(75, dtype=torch.int32, device="cuda")[None],
        cu_seq_lens=torch.tensor([0, 1200], dtype=torch.int32, device="cuda"),
        token_to_seq=torch.zeros(1200, dtype=torch.int32, device="cuda"),
        num_tokens=1200,
        kv_cache_dtype="fp8",
        scale=scale,
    )
    expected = dequantized[row_indices.long()].unsqueeze(1)
    assert latent.shape == (num_rows, 1, kv_lora_rank)
    assert k_pe.shape == (num_rows, 1, rope_dim)
    assert latent.dtype == k_pe.dtype == out_dtype
    assert latent.is_contiguous() and k_pe.is_contiguous()
    assert torch.equal(_bits(latent), _bits(expected[..., :kv_lora_rank]))
    assert torch.equal(_bits(k_pe), _bits(expected[..., kv_lora_rank:]))


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.parametrize(
    "case",
    [
        "no_out_dtype",
        "no_k_scale",
        "fp32_out",
        "fp16_scale",
        "two_scales",
        "host_scale",
        "bf16_rows",
        "strided_rows",
    ],
)
def test_gather_dcp_context_rows_rejects_bad_dequant_inputs(case):
    """Dequantizing takes contiguous fp8 rows, a one-element fp32 k_scale on
    their device and a bf16 or fp16 output dtype, the last two together."""
    fp8 = current_platform.fp8_dtype()
    rows = torch.zeros((8, 2 * _ENTRY), dtype=torch.uint8, device="cuda").view(fp8)
    gathered = rows[:, :_ENTRY].contiguous()
    row_indices = torch.zeros(4, dtype=torch.int64, device="cuda")
    scale = torch.ones((), dtype=torch.float32, device="cuda")
    kwargs = {"k_scale": scale, "out_dtype": torch.bfloat16}
    if case == "no_out_dtype":
        del kwargs["out_dtype"]
    elif case == "no_k_scale":
        del kwargs["k_scale"]
    elif case == "fp32_out":
        kwargs["out_dtype"] = torch.float32
    elif case == "fp16_scale":
        kwargs["k_scale"] = scale.half()
    elif case == "two_scales":
        kwargs["k_scale"] = scale.expand(2).contiguous()
    elif case == "host_scale":
        kwargs["k_scale"] = scale.cpu()
    elif case == "bf16_rows":
        gathered = gathered.to(torch.bfloat16)
    elif case == "strided_rows":
        gathered = rows[:, :_ENTRY]
    with pytest.raises(AssertionError):
        gather_dcp_context_rows(gathered, row_indices, _KV_LORA_RANK, **kwargs)


@pytest.mark.parametrize(
    "opt_in,dcp_world_size,q_fp8,model_dtype,last_layer,enabled",
    [
        (True, 2, True, torch.bfloat16, "fp8", True),
        (True, 2, True, torch.bfloat16, "fp8_e4m3", True),
        (True, 2, True, torch.float16, "fp8", True),
        (False, 2, True, torch.bfloat16, "fp8", False),
        (True, 1, True, torch.bfloat16, "fp8", False),
        (True, 2, False, torch.bfloat16, "fp8", False),
        (True, 2, True, torch.float32, "fp8", False),
        (True, 2, True, torch.bfloat16, "fp8_e5m2", False),
        (True, 2, True, torch.bfloat16, "fp8_ds_mla", False),
    ],
)
def test_fp8_dcp_context_only_when_exact(
    opt_in, dcp_world_size, q_fp8, model_dtype, last_layer, enabled
):
    """A group gathers its DCP context in fp8 only when its backend opts in, DCP
    and the fp8 prefill query are on, the default workspace (the model dtype) is
    bf16 or fp16, and every layer has a plain fp8 cache.
    """
    builder_cls = TokenspeedMLAMetadataBuilder if opt_in else MLACommonMetadataBuilder
    q_data_type = current_platform.fp8_dtype() if q_fp8 else torch.bfloat16
    assert (
        builder_cls.use_fp8_dcp_context(
            dcp_world_size, q_data_type, model_dtype, ["fp8", last_layer]
        )
        is enabled
    )


@pytest.mark.parametrize(
    "kv_cache_dtypes",
    [
        ["auto", "fp8"],
        ["fp8_e5m2", "fp8"],
        ["fp8_inc", "fp8"],
        ["fp8", "nvfp4_ds_mla"],
        ["auto", "auto"],
    ],
)
def test_fp8_dcp_context_needs_every_layer_fp8(kv_cache_dtypes):
    """Any layer without a plain fp8 cache, the first one included, keeps the
    group on the default path."""
    assert not TokenspeedMLAMetadataBuilder.use_fp8_dcp_context(
        _DCP, current_platform.fp8_dtype(), torch.bfloat16, kv_cache_dtypes
    )


class _RecordingKVBProj:
    """Records the latent ``kv_b_proj`` receives and projects it."""

    def __init__(self, weight: torch.Tensor) -> None:
        self.weight = weight
        self.inputs: list[torch.Tensor] = []

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        self.inputs.append(x.clone())
        out = x.float().reshape(x.shape[0], -1) @ self.weight.float().T
        return out.to(self.weight.dtype), None


class _RecordingPrefillBackend:
    """Records the K/V of every context chunk."""

    def __init__(self) -> None:
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    @staticmethod
    def get_name() -> str:
        return "recording"

    def supports_out(self) -> bool:
        return False

    def run_prefill_context_chunk(self, *, chunk, q, k, v, out=None):
        self.calls.append((k.clone(), v.clone()))
        num_q = q.shape[0]
        # Fold K/V into the partial so the merged output depends on them too.
        digest = k.float().mean() + v.float().mean()
        out = digest.expand(num_q, _NUM_HEADS, _V_HEAD_DIM).to(q.dtype)
        lse = torch.full(
            (_NUM_HEADS, num_q), 1.0 + chunk.index, dtype=torch.float32, device=q.device
        )
        return out, lse


def _fp8_cache(context_lens, generator, device) -> torch.Tensor:
    """A paged fp8 cache with one row of blocks per request."""
    blocks_per_request = -(-max(context_lens) // _BLOCK_SIZE) + 1
    raw = torch.randint(
        0,
        256,
        (len(context_lens) * blocks_per_request, _BLOCK_SIZE, _ENTRY),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )
    # Kept finite so that the merged output and lse stay informative.
    raw[(raw & 0x7F) == 0x7F] = 0
    return raw.view(current_platform.fp8_dtype())


def _kv_b_proj_weight(model_dtype, generator, device) -> torch.Tensor:
    weight = torch.randn(
        _NUM_HEADS * (_QK_NOPE + _V_HEAD_DIM),
        _KV_LORA_RANK,
        device=device,
        generator=generator,
    )
    return (weight * 0.02).to(model_dtype)


def _run_context(
    manager,
    workspace,
    workspace_size,
    context_lens,
    interleave,
    cache,
    k_scale,
    model_dtype,
    weight,
    fp8_rows,
):
    """One pass of ``_context_parallel_compute_prefill_context``."""
    from vllm.model_executor.layers.attention.mla_attention import (
        MLACommonImpl,
        build_mla_chunked_context_metadata,
    )

    device = cache.device
    query_lens = [i + 2 for i in range(len(context_lens))]
    query_start_loc = torch.zeros(len(query_lens) + 1, dtype=torch.int32)
    query_start_loc[1:] = torch.tensor(query_lens).cumsum(0)
    manager.init_kv_gather(workspace, workspace_size)
    chunked = build_mla_chunked_context_metadata(
        context_lens_cpu=torch.tensor(context_lens, dtype=torch.int32),
        prefill_query_start_loc_cpu=query_start_loc,
        chunked_prefill_workspace=workspace,
        chunked_prefill_workspace_size=workspace_size,
        block_size=_BLOCK_SIZE,
        align_chunk_to_block=True,
        device=device,
        dcp_world_size=_DCP,
        dcp_local_block_size=interleave,
        dcp_virtual_block_size=interleave * _DCP,
        dcp_manager=manager,
    )
    assert chunked is not None
    assert chunked.fp8_context_dtype is None
    if fp8_rows:
        # What the metadata builder does for a group on the fp8 path.
        chunked.fp8_context_dtype = model_dtype
    blocks_per_request = cache.shape[0] // len(context_lens)
    block_table = torch.arange(cache.shape[0], dtype=torch.int32, device=device).view(
        len(context_lens), blocks_per_request
    )

    kv_b_proj = _RecordingKVBProj(weight)
    impl = SimpleNamespace(
        kv_cache_dtype="fp8",
        kv_lora_rank=_KV_LORA_RANK,
        qk_rope_head_dim=_QK_ROPE,
        qk_nope_head_dim=_QK_NOPE,
        v_head_dim=_V_HEAD_DIM,
        num_heads=_NUM_HEADS,
        kv_b_proj=kv_b_proj,
        _use_flashinfer_concat_mla_k=False,
    )
    impl._concat_k_nope_k_pe = MLACommonImpl._concat_k_nope_k_pe.__get__(impl)
    backend = _RecordingPrefillBackend()
    attn_metadata = SimpleNamespace(
        prefill=SimpleNamespace(
            prefill_backend=backend,
            chunked_context=chunked,
            q_data_type=current_platform.fp8_dtype(),
            block_table=block_table,
        )
    )
    q = torch.zeros(
        (sum(query_lens), _NUM_HEADS, _QK_NOPE + _QK_ROPE),
        dtype=model_dtype,
        device=device,
    )
    out, lse = MLACommonImpl._context_parallel_compute_prefill_context(
        impl, q, cache, attn_metadata, k_scale=k_scale, dcp_world_size=_DCP
    )
    torch.accelerator.synchronize()
    return kv_b_proj.inputs, backend.calls, out, lse


def _run_both_paths(workspaces, proj_input_dtype, **kwargs):
    """Runs the dequantizing path, then the fp8 path, on the same inputs.

    ``proj_input_dtype`` None keeps the latent dtype; fp8 is the input an
    fp8-weight ``kv_b_proj`` takes on the fp8 prefill path.
    """
    from vllm.model_executor.layers.attention import mla_attention

    with patch.object(
        mla_attention, "_get_kv_b_proj_input_dtype", lambda *_: proj_input_dtype
    ):
        return [
            _run_context(workspace=workspaces[fp8_rows], fp8_rows=fp8_rows, **kwargs)
            for fp8_rows in (False, True)
        ]


def _assert_same_bytes(runs, case: str) -> None:
    (ref_in, ref_kv, ref_out, ref_lse), (fp8_in, fp8_kv, fp8_out, fp8_lse) = runs
    assert len(ref_in) == len(fp8_in) > 0, case
    for ref_t, fp8_t in zip(ref_in, fp8_in):
        assert ref_t.dtype == fp8_t.dtype, case
        assert torch.equal(_bits(ref_t), _bits(fp8_t)), f"kv_b_proj input, {case}"
    assert len(ref_kv) == len(fp8_kv), case
    for (ref_k, ref_v), (fp8_k, fp8_v) in zip(ref_kv, fp8_kv):
        assert torch.equal(_bits(ref_k), _bits(fp8_k)), f"K, {case}"
        assert torch.equal(_bits(ref_v), _bits(fp8_v)), f"V, {case}"
    assert torch.equal(_bits(ref_out), _bits(fp8_out)), f"output, {case}"
    assert torch.equal(_bits(ref_lse), _bits(fp8_lse)), f"lse, {case}"


def _workspaces(workspace_size, model_dtype, device):
    rows = workspace_size + workspace_size // _DCP
    return {
        fp8_rows: torch.empty(
            (rows, _ENTRY),
            dtype=current_platform.fp8_dtype() if fp8_rows else model_dtype,
            device=device,
        )
        for fp8_rows in (False, True)
    }


@torch.inference_mode()
def _fp8_context_worker(rank, init_method, direct_kv_gather):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.v1.attention.ops import dcp

    torch.accelerator.set_device_index(rank)
    device = torch.device(f"cuda:{rank}")
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=_DCP,
            rank=rank,
            local_rank=rank,
            distributed_init_method=init_method,
        )
        initialize_model_parallel(
            tensor_model_parallel_size=_DCP,
            decode_context_model_parallel_size=_DCP,
        )
        try:
            config = MagicMock()
            config.parallel_config.num_ubatches = 1
            config.parallel_config.dcp_comm_backend = "ag_rs"
            config.scheduler_config.max_num_batched_tokens = 64
            config.scheduler_config.max_num_seqs = 8
            config.num_speculative_tokens = 0
            config.speculative_config = None
            config.compilation_config.max_cudagraph_capture_size = 0
            manager = dcp.MLADCPManager(
                config,
                device,
                num_heads=_NUM_HEADS,
                query_head_dim=_ENTRY,
                output_head_dim=_V_HEAD_DIM,
                query_dtype=torch.bfloat16,
                output_dtype=torch.bfloat16,
                padded_num_heads=None,
                is_lse_base_on_e=True,
                use_pcp=False,
            )
            for interleave, context_lens, workspace_size in _LAYOUTS:
                generator = torch.Generator(device=device)
                generator.manual_seed(1000 * rank + interleave + len(context_lens))
                cache = _fp8_cache(context_lens, generator, device)
                for model_dtype in (torch.bfloat16, torch.float16):
                    weight = _kv_b_proj_weight(model_dtype, generator, device)
                    workspaces = _workspaces(workspace_size, model_dtype, device)
                    # A 0-dim scale, as layers hold it, updated in place as a
                    # weight reload updates it.
                    k_scale = torch.ones((), dtype=torch.float32, device=device)
                    previous = None
                    for scale in _K_SCALES:
                        k_scale.fill_(scale)
                        for proj_input_dtype in (None, current_platform.fp8_dtype()):
                            runs = _run_both_paths(
                                workspaces,
                                proj_input_dtype,
                                manager=manager,
                                workspace_size=workspace_size,
                                context_lens=context_lens,
                                interleave=interleave,
                                cache=cache,
                                k_scale=k_scale,
                                model_dtype=model_dtype,
                                weight=weight,
                            )
                            case = (
                                f"{rank=} {interleave=} {context_lens=} "
                                f"{model_dtype=} {scale=} {proj_input_dtype=}"
                            )
                            _assert_same_bytes(runs, case)
                        # The fp8 path follows the updated scale.
                        fp8_k = runs[1][1][0][0]
                        if previous is not None:
                            assert not torch.equal(_bits(previous), _bits(fp8_k)), case
                        previous = fp8_k
            gather = manager._kv_gather
            uses_direct = isinstance(
                getattr(gather, "__self__", None), dcp.DirectDCPKVGatherWorkspace
            )
            assert uses_direct == direct_kv_gather
        finally:
            cleanup_dist_env_and_memory()


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@multi_gpu_only(num_gpus=2)
@pytest.mark.parametrize("direct_kv_gather", [False, True])
def test_fp8_dcp_context_matches_dequantizing_path(
    direct_kv_gather, monkeypatch, tmp_path
):
    """With either DCP KV gather, the fp8 path feeds ``kv_b_proj`` and the
    prefill backend the same bytes as the dequantizing path, for several
    k_scale values updated in place, both model dtypes and both kv_b_proj
    input dtypes."""
    import torch.multiprocessing as mp

    from vllm.distributed.device_communicators.custom_all_reduce import (
        _has_local_multicast_support,
    )

    if direct_kv_gather and not _has_local_multicast_support(torch.device("cuda", 0)):
        pytest.skip("The direct DCP KV gather needs symmetric-memory multicast.")
    # The spawned ranks inherit the environment.
    monkeypatch.setenv(
        "VLLM_USE_DIRECT_DCP_KV_GATHER", "1" if direct_kv_gather else "0"
    )
    mp.spawn(
        _fp8_context_worker,
        args=(f"file://{tmp_path / 'dist_init'}", direct_kv_gather),
        nprocs=_DCP,
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.parametrize("model_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("scale", [1.0, 0.37])
@torch.inference_mode()
def test_fp8_dcp_context_loop_applies_k_scale(scale, model_dtype):
    """On one GPU, the chunked-context loop hands the fp8 row gather the
    layer's k_scale and the default workspace dtype.

    The DCP all-gather is replaced by a copy of the local rows to every rank
    slot, so both paths see the same gathered rows.
    """
    device = torch.device("cuda:0")
    manager = SimpleNamespace(
        init_kv_gather=lambda *_: None,
        kv_gather=lambda dst, src: dst.view(_DCP, *src.shape).copy_(
            src.expand(_DCP, *src.shape)
        ),
    )
    k_scale = torch.tensor(scale, dtype=torch.float32, device=device)
    for interleave, context_lens, workspace_size in _LAYOUTS:
        generator = torch.Generator(device=device)
        generator.manual_seed(interleave + len(context_lens))
        runs = _run_both_paths(
            _workspaces(workspace_size, model_dtype, device),
            None,
            manager=manager,
            workspace_size=workspace_size,
            context_lens=context_lens,
            interleave=interleave,
            cache=_fp8_cache(context_lens, generator, device),
            k_scale=k_scale,
            model_dtype=model_dtype,
            weight=_kv_b_proj_weight(model_dtype, generator, device),
        )
        assert runs[1][0][0].dtype == model_dtype
        _assert_same_bytes(runs, f"{interleave=} {context_lens=}")
