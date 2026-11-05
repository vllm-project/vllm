# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import multiprocessing
import random
import sys
import types
from types import SimpleNamespace

import pytest
import ray
import torch
import torch.distributed as dist

import vllm._aiter_ops as aiter_ops
import vllm.distributed as vllm_distributed
import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.config import ParallelConfig
from vllm.distributed.communication_op import tensor_model_parallel_all_reduce  # noqa
from vllm.distributed.device_communicators.quick_all_reduce import (
    KB,
    MB,
    QuickAllReduce,
    QuickReduceRegime,
)
from vllm.distributed.parallel_state import get_tp_group, graph_capture
from vllm.envs import disable_envs_cache
from vllm.logger import _print_warning_once
from vllm.models.common.ops import sequence_parallel as common_sp
from vllm.platforms import current_platform

from ..utils import (
    ensure_model_parallel_initialized,
    init_test_distributed_environment,
    multi_process_parallel,
    set_random_seed,
)


def on_gfx942() -> bool:
    if current_platform.is_rocm():
        from vllm.platforms.rocm import on_gfx942 as rocm_on_gfx942

        return rocm_on_gfx942()
    return False


def _quickreduce_tolerance(quant_level: QuickReduceRegime) -> dict:
    # With INT3 inputs of absmax 23, per-rank error peaks at 2.75,
    # doubled by the TP2 sum = 5.5.
    if quant_level == QuickReduceRegime.INT3:
        return {"atol": 5.5, "rtol": 0.1}
    return {"atol": 2.5, "rtol": 0.1}


set_random_seed(42)
_test_size_rng = random.Random(44)
# Size over 8MB is sufficient for custom quick allreduce.
test_sizes = [
    _test_size_rng.randint(8 * 1024 * 1024, 10 * 1024 * 1024) for _ in range(8)
]
for i, v in enumerate(test_sizes):
    test_sizes[i] -= v % 8


def _assert_quickreduce(fa, inp):
    assert fa is not None
    assert not fa.disabled
    assert fa.should_quick_allreduce(inp)


@pytest.fixture
def envs_cache_disabled():
    disable_envs_cache()
    yield
    disable_envs_cache()


def _make_quick_allreduce_for_test(
    min_size_mb: int | None = None,
    quantization_min_size: int | None = None,
    max_size_mb: int = 16,
) -> QuickAllReduce:
    quick_reduce = QuickAllReduce.__new__(QuickAllReduce)
    quick_reduce._flydsl_int4 = None
    quick_reduce._flydsl_sp = False
    quick_reduce.disabled = False
    quick_reduce.qr_max_size = max_size_mb * MB
    quick_reduce.qr_min_size = min_size_mb * MB if min_size_mb is not None else None
    quick_reduce.qr_quant_level = QuickReduceRegime.INT4
    quick_reduce.qr_quantization_min_size = quantization_min_size
    quick_reduce.use_fp16_kernels = False
    quick_reduce.world_size = 2
    return quick_reduce


class _FakeFlyDSLInt4:
    """The sequence-parallel methods play rank 0 of 2: the peer's all-gather
    block is `block + 1`."""

    def __init__(self):
        self.allreduce_calls = 0
        self.closed = False
        self.sp_calls = []

    def allreduce(self, inp, out):
        self.allreduce_calls += 1

    def reduce_scatter(self, inp, out):
        self._check_tensors(inp, out, full=inp)
        self.sp_calls.append(("reduce_scatter", inp, out))
        out.copy_(inp[: out.shape[0]])

    def all_gather(self, inp, out):
        self._check_tensors(inp, out, full=out)
        self.sp_calls.append(("all_gather", inp, out))
        out[: inp.shape[0]] = inp
        out[inp.shape[0] :] = inp + 1

    @staticmethod
    def _check_tensors(inp, out, *, full):
        # AITER's QuickAllReduceInt4._check_tensors raises on misaligned buffers
        # and on a full tensor that is not a whole number of 32 KiB tiles.
        if inp.data_ptr() % 16 or out.data_ptr() % 16:
            raise ValueError("requires 16-byte-aligned input/output")
        if full.numel() * full.element_size() % (32 * KB):
            raise ValueError("the full tensor must be a whole number of tiles")

    def compile_sp(self, full, shard):
        self.sp_calls.append(("compile_sp", full, shard))

    def close(self):
        self.closed = True


def _bf16(num_elems: int) -> torch.Tensor:
    return torch.empty(num_elems, dtype=torch.bfloat16)


@pytest.mark.parametrize(
    "inp, quantization_min_size, expect_flydsl",
    [
        pytest.param(_bf16(MB), None, True, id="bf16"),
        pytest.param(torch.empty(MB, dtype=torch.float16), None, False, id="fp16"),
        pytest.param(_bf16(2 * MB)[::2], None, False, id="non_contiguous"),
        # Right size, but the data pointer is 8 bytes past a 16-byte boundary.
        pytest.param(_bf16(MB + 16)[4 : MB + 4], None, False, id="misaligned"),
        # Below the quantization min size the FP codec runs, which FlyDSL lacks.
        pytest.param(_bf16(MB), 4 * MB, False, id="fp_codec_below_quant_min"),
    ],
)
def test_flydsl_int4_dispatch_falls_back_to_hip(
    monkeypatch: pytest.MonkeyPatch,
    inp: torch.Tensor,
    quantization_min_size: int | None,
    expect_flydsl: bool,
):
    quick_reduce = _make_quick_allreduce_for_test(
        quantization_min_size=quantization_min_size
    )
    quick_reduce._ptr = 0
    fake = _FakeFlyDSLInt4()
    quick_reduce._flydsl_int4 = fake
    hip_calls = []
    monkeypatch.setattr(ops, "qr_all_reduce", lambda *args: hip_calls.append(args))

    quick_reduce.quick_all_reduce(inp)

    assert fake.allreduce_calls == int(expect_flydsl)
    assert len(hip_calls) == int(not expect_flydsl)


def test_close_closes_flydsl_int4():
    quick_reduce = _make_quick_allreduce_for_test()
    quick_reduce.disabled = True  # skip the HIP teardown
    fake = _FakeFlyDSLInt4()
    quick_reduce._flydsl_int4 = fake

    quick_reduce.close()

    assert fake.closed
    assert quick_reduce._flydsl_int4 is None


def test_should_quick_allreduce_uses_builtin_min_size_when_unset():
    quick_reduce = _make_quick_allreduce_for_test(min_size_mb=None)

    below_builtin_min = torch.empty(MB // 4, dtype=torch.float16)
    at_builtin_min = torch.empty(MB // 2, dtype=torch.float16)

    assert not quick_reduce.should_quick_allreduce(below_builtin_min)
    assert quick_reduce.should_quick_allreduce(at_builtin_min)


def test_should_quick_allreduce_uses_min_size_override():
    quick_reduce = _make_quick_allreduce_for_test(min_size_mb=0)

    below_builtin_min = torch.empty(8, dtype=torch.float16)

    assert quick_reduce.should_quick_allreduce(below_builtin_min)


def test_quick_allreduce_min_size_env_unset(
    monkeypatch: pytest.MonkeyPatch,
    envs_cache_disabled,
):
    monkeypatch.delenv("VLLM_ROCM_QUICK_REDUCE_MIN_SIZE_BYTES_MB", raising=False)

    assert QuickAllReduce._get_qr_min_size(qr_max_size=16 * MB) is None


def test_quick_allreduce_min_size_env_converts_mb_to_bytes(
    monkeypatch: pytest.MonkeyPatch,
    envs_cache_disabled,
):
    monkeypatch.setenv("VLLM_ROCM_QUICK_REDUCE_MIN_SIZE_BYTES_MB", "4")

    assert QuickAllReduce._get_qr_min_size(qr_max_size=16 * MB) == 4 * MB


def test_quick_allreduce_min_size_env_rejects_negative(
    monkeypatch: pytest.MonkeyPatch,
    envs_cache_disabled,
):
    monkeypatch.setenv("VLLM_ROCM_QUICK_REDUCE_MIN_SIZE_BYTES_MB", "-1")

    with pytest.raises(ValueError, match="must be non-negative"):
        QuickAllReduce._get_qr_min_size(qr_max_size=16 * MB)


def test_quick_allreduce_min_size_env_allows_equal_to_max(
    monkeypatch: pytest.MonkeyPatch,
    envs_cache_disabled,
):
    monkeypatch.setenv("VLLM_ROCM_QUICK_REDUCE_MIN_SIZE_BYTES_MB", "16")

    assert QuickAllReduce._get_qr_min_size(qr_max_size=16 * MB) == 16 * MB


def test_quick_allreduce_min_size_env_rejects_larger_than_max(
    monkeypatch: pytest.MonkeyPatch,
    envs_cache_disabled,
):
    monkeypatch.setenv("VLLM_ROCM_QUICK_REDUCE_MIN_SIZE_BYTES_MB", "17")

    with pytest.raises(ValueError, match="effective QuickReduce max size"):
        QuickAllReduce._get_qr_min_size(qr_max_size=16 * MB)


def test_quick_allreduce_quantization_min_size_env_unset(
    monkeypatch: pytest.MonkeyPatch,
    envs_cache_disabled,
):
    monkeypatch.delenv("VLLM_ROCM_QUICK_REDUCE_QUANTIZATION_MIN_SIZE_KB", raising=False)

    assert QuickAllReduce._get_qr_quantization_min_size() is None


def test_quick_allreduce_quantization_min_size_env_converts_kb_to_bytes(
    monkeypatch: pytest.MonkeyPatch,
    envs_cache_disabled,
):
    monkeypatch.setenv("VLLM_ROCM_QUICK_REDUCE_QUANTIZATION_MIN_SIZE_KB", "2048")

    assert QuickAllReduce._get_qr_quantization_min_size() == 2048 * KB


def test_quick_allreduce_quantization_min_size_env_rejects_negative(
    monkeypatch: pytest.MonkeyPatch,
    envs_cache_disabled,
):
    monkeypatch.setenv("VLLM_ROCM_QUICK_REDUCE_QUANTIZATION_MIN_SIZE_KB", "-1")

    with pytest.raises(ValueError, match="must be non-negative"):
        QuickAllReduce._get_qr_quantization_min_size()


def test_quick_allreduce_quantization_min_size_unset_uses_configured_codec():
    quick_reduce = _make_quick_allreduce_for_test(quantization_min_size=None)
    inp = torch.empty(8, dtype=torch.float16)

    assert quick_reduce._get_qr_quant_level(inp) == QuickReduceRegime.INT4.value


def test_quick_allreduce_quantization_min_size_uses_fp_below_threshold():
    quick_reduce = _make_quick_allreduce_for_test(quantization_min_size=2048)
    inp = torch.empty(1024 // 2, dtype=torch.float16)

    assert quick_reduce._get_qr_quant_level(inp) == QuickReduceRegime.FP.value


def test_quick_allreduce_quantization_min_size_uses_configured_codec_at_threshold():
    quick_reduce = _make_quick_allreduce_for_test(quantization_min_size=2048)
    inp = torch.empty(2048 // 2, dtype=torch.float16)

    assert quick_reduce._get_qr_quant_level(inp) == QuickReduceRegime.INT4.value


def test_quick_allreduce_quantization_min_size_does_not_change_eligibility():
    quick_reduce = _make_quick_allreduce_for_test(quantization_min_size=2 * MB)

    below_builtin_min = torch.empty(MB // 4, dtype=torch.float16)
    at_builtin_min = torch.empty(MB // 2, dtype=torch.float16)

    assert not quick_reduce.should_quick_allreduce(below_builtin_min)
    assert quick_reduce.should_quick_allreduce(at_builtin_min)


def test_quick_allreduce_passes_dynamic_quant_level(
    monkeypatch: pytest.MonkeyPatch,
):
    quick_reduce = _make_quick_allreduce_for_test(quantization_min_size=2 * KB)
    quick_reduce._ptr = object()
    inp = torch.empty(KB // 2, dtype=torch.float16)
    called_quant_level = None

    def fake_qr_all_reduce(
        fa,
        inp,
        out,
        quant_level,
        cast_bf2half,
    ):
        nonlocal called_quant_level
        called_quant_level = quant_level

    monkeypatch.setattr(ops, "qr_all_reduce", fake_qr_all_reduce)

    quick_reduce.quick_all_reduce(inp)

    assert called_quant_level == QuickReduceRegime.FP.value


def test_rocm_aiter_fused_rmsnorm_uses_aiter_qr_rmsnorm_for_prefill(
    monkeypatch: pytest.MonkeyPatch,
):
    class FakeAiterAllReduce:
        aiter_ca = SimpleNamespace(world_size=2, fully_connected=True)

        def use_1stage_fused_ar_rms(self, inp):
            return False

        def custom_fused_ar_rms(self, *args, **kwargs):
            raise AssertionError("expected fused QR+RMSNorm dispatch")

    class FakeQuickReduce:
        disabled = False

        def __init__(self):
            self.checked_input = None

        def uses_flydsl_int4(self, inp):
            return False

        def should_quick_allreduce(self, inp):
            self.checked_input = inp
            return True

        def _get_qr_quant_level(self, inp):
            return QuickReduceRegime.INT4.value

    qr_comm = FakeQuickReduce()
    device_comm = SimpleNamespace(cpu_group=object(), qr_comm=qr_comm)
    monkeypatch.setattr(
        vllm_distributed,
        "get_tp_group",
        lambda: SimpleNamespace(device_communicator=device_comm),
    )
    monkeypatch.setattr(
        aiter_ops.rocm_aiter_ops,
        "get_aiter_allreduce",
        lambda: FakeAiterAllReduce(),
    )
    monkeypatch.setattr(aiter_ops.dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(aiter_ops.dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(
        aiter_ops.dist,
        "all_gather_object",
        lambda handles, handle, group: handles.__setitem__(0, handle),
    )

    calls = []
    fake_aiter = types.SimpleNamespace(
        init_custom_qr=lambda rank, world_size, qr_max_size: 123,
        qr_get_handle=lambda ptr: b"handle",
        qr_open_handles=lambda ptr, handles: None,
    )

    def fake_qr_all_reduce_rmsnorm(
        ptr,
        inp,
        residual,
        residual_out,
        out,
        weight,
        eps,
        hidden_dim,
        quant_level,
        cast_bf2half,
    ):
        calls.append(
            {
                "ptr": ptr,
                "inp": inp,
                "residual": residual,
                "residual_out": residual_out,
                "out": out,
                "weight": weight,
                "eps": eps,
                "hidden_dim": hidden_dim,
                "quant_level": quant_level,
                "cast_bf2half": cast_bf2half,
            }
        )
        out.copy_(inp)
        residual_out.copy_(residual)

    fake_aiter.qr_all_reduce_rmsnorm = fake_qr_all_reduce_rmsnorm
    monkeypatch.setitem(sys.modules, "aiter", fake_aiter)
    monkeypatch.setattr(envs, "VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB", None)
    monkeypatch.setattr(envs, "VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16", True)

    # token_num > 80 forces vLLM's AITER fused AR+RMS wrapper onto its
    # non-1stage prefill path. hidden_dim=4096 bf16 gives an 8 KiB row, so four
    # independent RMSNorm rows fit in one 32 KiB QR tile.
    inp = torch.randn(81, 4096, dtype=torch.bfloat16)
    residual = torch.randn_like(inp)
    weight = torch.randn(4096, dtype=torch.bfloat16)

    out, residual_out = aiter_ops._rocm_aiter_fused_allreduce_rmsnorm_impl(
        inp, residual, weight, 1e-6
    )

    assert qr_comm.checked_input is inp
    assert len(calls) == 1
    assert calls[0]["ptr"] == 123
    assert calls[0]["inp"] is inp
    assert calls[0]["residual"] is residual
    assert calls[0]["out"] is out
    assert calls[0]["residual_out"] is residual_out
    assert calls[0]["weight"] is weight
    assert calls[0]["hidden_dim"] == 4096
    assert calls[0]["quant_level"] == QuickReduceRegime.INT4.value
    assert calls[0]["cast_bf2half"] is True
    torch.testing.assert_close(out, inp)
    torch.testing.assert_close(residual_out, residual)


def test_rocm_aiter_fused_rmsnorm_keeps_1stage_decode_path(
    monkeypatch: pytest.MonkeyPatch,
):
    calls = []

    class FakeAiterCA:
        world_size = 2
        fully_connected = True

        def custom_fused_ar_rms(
            self, inp, residual, weight, epsilon, *, use_1stage, gemma_norm
        ):
            calls.append(
                {
                    "inp": inp,
                    "residual": residual,
                    "weight": weight,
                    "eps": epsilon,
                    "use_1stage": use_1stage,
                    "gemma_norm": gemma_norm,
                }
            )
            return inp + 1, residual + 1

    class FakeAiterAllReduce:
        aiter_ca = FakeAiterCA()

        def use_1stage_fused_ar_rms(self, inp):
            return True

    monkeypatch.setattr(
        aiter_ops.rocm_aiter_ops,
        "get_aiter_allreduce",
        lambda: FakeAiterAllReduce(),
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    inp = torch.randn(4, 4096, dtype=torch.bfloat16)
    residual = torch.randn_like(inp)
    weight = torch.randn(4096, dtype=torch.bfloat16)

    out, residual_out = aiter_ops._rocm_aiter_fused_allreduce_rmsnorm_impl(
        inp, residual, weight, 1e-6
    )

    assert len(calls) == 1
    assert calls[0]["inp"] is inp
    assert calls[0]["residual"] is residual
    assert calls[0]["weight"] is weight
    assert calls[0]["use_1stage"] is True
    assert calls[0]["gemma_norm"] is False
    torch.testing.assert_close(out, inp + 1)
    torch.testing.assert_close(residual_out, residual + 1)


# Sequence-parallel entry points (`--enable-hc-sp-quick-reduce`). The fake plays
# AITER's FlyDSL kernels, so these cover the gating, the padding to whole tiles
# and the unpadding, not the kernels.
SP_HIDDEN = 2560  # a row is 5,120 B: the kernels' 32 KiB tile holds 6.4 rows


def _make_sp_quick_allreduce(
    sp_enabled: bool = True, **kwargs
) -> tuple[QuickAllReduce, _FakeFlyDSLInt4]:
    quick_reduce = _make_quick_allreduce_for_test(**kwargs)
    fake = _FakeFlyDSLInt4()
    quick_reduce._flydsl_int4 = fake
    quick_reduce._flydsl_sp = sp_enabled
    return quick_reduce, fake


def _sp_rows(num_rows: int) -> torch.Tensor:
    """bf16 rows that differ between the two halves of a 2 * 1,030-row tensor.

    Integers up to 255 are exact in bf16, so the expected values are exact.
    """
    values = (
        torch.arange(num_rows)[:, None] % 251 + torch.arange(SP_HIDDEN)[None, :] % 5
    )
    return values.to(torch.bfloat16)


def test_sp_reduce_scatter_pads_each_block_to_half_tiles():
    quick_reduce, fake = _make_sp_quick_allreduce()
    # 65 * 16 rows: 16 rows are 80 KiB, half a tile times 5, so the full tensor
    # of 2 blocks is a whole number of tiles (325)
    chunk, padded = 1030, 1040
    inp = _sp_rows(2 * chunk)

    out = quick_reduce.quick_reduce_scatter(inp)

    ((name, src, kernel_out),) = fake.sp_calls
    assert name == "reduce_scatter"
    assert src.shape == (2 * padded, SP_HIDDEN)
    assert kernel_out.shape == (padded, SP_HIDDEN)
    for rank in range(2):  # every block keeps its rows, then zeros
        block = src[rank * padded : (rank + 1) * padded]
        assert torch.equal(block[:chunk], inp[rank * chunk : (rank + 1) * chunk])
        assert not block[chunk:].any()
    assert out.shape == (chunk, SP_HIDDEN)
    assert torch.equal(out, inp[:chunk])  # the fake's rank-0 block, unpadded


def test_sp_all_gather_pads_each_block_and_keeps_rank_order():
    quick_reduce, fake = _make_sp_quick_allreduce()
    chunk, padded = 1030, 1040
    inp = _sp_rows(chunk)

    out = quick_reduce.quick_all_gather(inp)

    ((name, src, kernel_out),) = fake.sp_calls
    assert name == "all_gather"
    assert src.shape == (padded, SP_HIDDEN)
    assert kernel_out.shape == (2 * padded, SP_HIDDEN)
    assert torch.equal(src[:chunk], inp) and not src[chunk:].any()
    assert out.shape == (2 * chunk, SP_HIDDEN)
    assert torch.equal(out[:chunk], inp)  # rank 0, then the fake's rank 1
    assert torch.equal(out[chunk:], inp + 1)


def test_sp_aligned_blocks_are_not_copied():
    quick_reduce, fake = _make_sp_quick_allreduce()
    chunk = 1040  # a multiple of 16 rows, not of 32: half-tile blocks are enough
    inp = _sp_rows(2 * chunk)
    shard = _sp_rows(chunk)

    out_rs = quick_reduce.quick_reduce_scatter(inp)
    out_ag = quick_reduce.quick_all_gather(shard)

    (_, rs_src, rs_out), (_, ag_src, ag_out) = fake.sp_calls
    assert rs_src is inp and ag_src is shard
    assert out_rs is rs_out and out_ag is ag_out


@pytest.mark.parametrize(
    "case",
    ["flag_off", "fp16", "non_contiguous", "below_min_size", "above_max_size"]
    + ["below_quantization_min_size", "one_dimensional"],
)
def test_sp_ineligible_inputs_return_none(case):
    kwargs = {}
    inp = _sp_rows(2 * 1024)
    if case == "flag_off":
        kwargs["sp_enabled"] = False
    elif case == "fp16":
        inp = inp.to(torch.float16)
    elif case == "non_contiguous":
        inp = _sp_rows(4 * 1024)[::2]
    elif case == "below_min_size":
        inp = _sp_rows(2 * 64)  # 655 KB against the 8 MB built-in TP2 INT4 minimum
    elif case == "above_max_size":
        kwargs["max_size_mb"] = 8  # the 10.5 MB message does not fit
    elif case == "below_quantization_min_size":
        kwargs["quantization_min_size"] = 32 * MB
    elif case == "one_dimensional":
        inp = inp.reshape(-1)
    quick_reduce, fake = _make_sp_quick_allreduce(**kwargs)

    assert quick_reduce.quick_reduce_scatter(inp) is None
    assert quick_reduce.quick_all_gather(inp) is None
    assert not fake.sp_calls


@pytest.mark.parametrize(
    "hidden, chunk, padded",
    [
        (2560, 1030, 1040),  # 5,120 B rows: 16 rows are 5 * 16 KiB
        (1024, 1030, 1032),  # 2,048 B rows: 8 rows are 16 KiB
        (4096, 1031, 1032),  # 8,192 B rows: 2 rows are 16 KiB
        (3000, 1030, 2048),  # 6,000 B rows: 1,024 rows are 375 * 16 KiB
        (2560, 1040, 1040),  # already a multiple: nothing to pad
    ],
)
def test_sp_pad_unit_is_the_fewest_rows_of_whole_half_tiles(hidden, chunk, padded):
    quick_reduce, fake = _make_sp_quick_allreduce(min_size_mb=0, max_size_mb=64)
    inp = torch.zeros(chunk, hidden, dtype=torch.bfloat16)

    out = quick_reduce.quick_all_gather(inp)

    # the fake refuses a full tensor that is not a whole number of tiles
    ((_, src, kernel_out),) = fake.sp_calls
    assert src.shape == (padded, hidden) and kernel_out.shape == (2 * padded, hidden)
    assert out.shape == (2 * chunk, hidden)


def _misaligned_rows(rows: int) -> torch.Tensor:
    """A contiguous bf16 [rows, SP_HIDDEN] tensor 2 bytes past a 16-byte boundary."""
    storage = torch.empty(rows * SP_HIDDEN + 8, dtype=torch.bfloat16)
    view = storage[1 : 1 + rows * SP_HIDDEN].view(rows, SP_HIDDEN)
    view.copy_(_sp_rows(rows))
    assert view.data_ptr() % 16 == 2
    return view


def test_sp_misaligned_input_is_copied_not_declined():
    """A misaligned view reaches the kernel as an aligned copy: every rank must
    take the same path, so it is not declined."""
    quick_reduce, fake = _make_sp_quick_allreduce()
    inp = _misaligned_rows(2 * 1024)  # whole tiles: no padding copy hides it
    shard = _misaligned_rows(1024)

    out_rs = quick_reduce.quick_reduce_scatter(inp)
    out_ag = quick_reduce.quick_all_gather(shard)

    (_, rs_src, _), (_, ag_src, _) = fake.sp_calls
    assert rs_src is not inp and torch.equal(rs_src, inp)
    assert ag_src is not shard and torch.equal(ag_src, shard)
    assert torch.equal(out_rs, inp[:1024]) and torch.equal(out_ag[:1024], shard)


def test_sp_reduce_scatter_needs_rows_divisible_by_world_size():
    quick_reduce, fake = _make_sp_quick_allreduce()

    assert quick_reduce.quick_reduce_scatter(_sp_rows(2 * 1024 + 1)) is None
    assert not fake.sp_calls


def test_sp_min_size_override_admits_small_messages():
    quick_reduce, fake = _make_sp_quick_allreduce(min_size_mb=0)

    assert quick_reduce.quick_all_gather(_sp_rows(64)) is not None
    assert len(fake.sp_calls) == 1


def _init_flydsl_sp(
    monkeypatch, world_size: int, flydsl_int4, flag: bool | None, cdna: int = 4
):
    """`flag` None stands for no VllmConfig being set; `cdna` 4 is gfx950."""
    monkeypatch.setitem(
        sys.modules,
        "vllm.platforms.rocm",
        SimpleNamespace(get_cdna_version=lambda: cdna),
    )
    quick_reduce = _make_quick_allreduce_for_test()
    quick_reduce.world_size = world_size
    vllm_config = (
        None
        if flag is None
        else SimpleNamespace(
            parallel_config=ParallelConfig(enable_hc_sp_quick_reduce=flag)
        )
    )
    warmup = torch.empty(MB, dtype=torch.bfloat16)
    return quick_reduce._init_flydsl_sp(flydsl_int4, warmup, vllm_config), warmup


def test_warm_up_prefers_preload_and_does_not_launch():
    calls = []
    fake = SimpleNamespace(
        preload=lambda: calls.append("preload"),
        compile=lambda inp, out: calls.append("compile"),
    )

    QuickAllReduce._warm_up_flydsl(fake, torch.empty(MB, dtype=torch.bfloat16))

    assert calls == ["preload"]


def test_warm_up_compiles_on_an_aiter_without_preload():
    calls = []
    fake = SimpleNamespace(compile=lambda inp, out: calls.append((inp, out)))
    warmup = torch.empty(MB, dtype=torch.bfloat16)

    QuickAllReduce._warm_up_flydsl(fake, warmup)

    ((inp, out),) = calls
    assert inp is warmup and out.shape == warmup.shape


def test_sp_init_runs_compile_sp_once_when_the_flag_is_on(monkeypatch):
    fake = _FakeFlyDSLInt4()

    enabled, warmup = _init_flydsl_sp(monkeypatch, 2, fake, True)

    assert enabled
    ((name, full, shard),) = fake.sp_calls
    assert name == "compile_sp"
    assert full is warmup and shard.numel() == warmup.numel() // 2
    assert full.numel() * full.element_size() % (32 * 1024) == 0  # whole tiles


@pytest.mark.parametrize("flag", [None, False])
def test_sp_init_is_inert_without_the_flag(monkeypatch, flag):
    fake = _FakeFlyDSLInt4()

    enabled, _ = _init_flydsl_sp(monkeypatch, 2, fake, flag)

    assert not enabled and not fake.sp_calls


@pytest.mark.parametrize("world_size, cdna", [(4, 4), (2, 3)], ids=["tp4", "gfx942"])
def test_sp_init_stays_exact_where_it_was_not_validated(monkeypatch, world_size, cdna):
    fake = _FakeFlyDSLInt4()

    enabled, _ = _init_flydsl_sp(monkeypatch, world_size, fake, True, cdna)

    assert not enabled and not fake.sp_calls


def test_sp_init_stays_exact_when_aiter_lacks_the_methods(monkeypatch):
    enabled, _ = _init_flydsl_sp(
        monkeypatch, 2, SimpleNamespace(allreduce=lambda *args: None), True
    )

    assert not enabled


def test_sp_entry_points_are_inert_after_an_early_init_return():
    """`__init__` returns before `world_size` is set when the group spans nodes."""
    quick_reduce = QuickAllReduce.__new__(QuickAllReduce)
    quick_reduce.disabled = True
    quick_reduce._flydsl_int4 = None
    quick_reduce._flydsl_sp = False
    inp = _sp_rows(2 * 1024)

    assert quick_reduce.quick_reduce_scatter(inp) is None
    assert quick_reduce.quick_all_gather(inp) is None


class _FakeCommunicator:
    def __init__(self, result=None, qr_comm=None):
        self.result = result
        self.qr_comm = qr_comm
        self.seen = []

    def quick_reduce_scatter(self, x):
        self.seen.append(("reduce_scatter", x.shape))
        return self.result

    def quick_all_gather(self, x):
        self.seen.append(("all_gather", x.shape))
        return self.result


def _patch_tp(monkeypatch, communicator):
    monkeypatch.setattr(
        common_sp,
        "get_tp_group",
        lambda: SimpleNamespace(device_communicator=communicator),
    )
    monkeypatch.setattr(common_sp, "get_tensor_model_parallel_world_size", lambda: 2)
    exact = SimpleNamespace(rs=[], ag=[])
    monkeypatch.setattr(
        common_sp, "sp_reduce_scatter", lambda x: exact.rs.append(x.shape) or "exact_rs"
    )
    monkeypatch.setattr(
        common_sp, "sp_all_gather", lambda x: exact.ag.append(x.shape) or "exact_ag"
    )
    return exact


@pytest.mark.parametrize(
    "make_communicator, expect_quick",
    [
        pytest.param(
            lambda: _FakeCommunicator(qr_comm=_make_sp_quick_allreduce()[0]),
            True,
            id="serves_sp",
        ),
        pytest.param(
            lambda: _FakeCommunicator(
                qr_comm=_make_sp_quick_allreduce(sp_enabled=False)[0]
            ),
            False,
            id="qr_comm_without_sp",
        ),
        pytest.param(lambda: _FakeCommunicator(), False, id="no_qr_comm"),
        pytest.param(lambda: SimpleNamespace(), False, id="no_qr_comm_attribute"),
        pytest.param(lambda: None, False, id="no_communicator"),
    ],
)
def test_use_quick_sp_follows_the_flag_and_the_communicator(
    monkeypatch, caplog_vllm, make_communicator, expect_quick
):
    _patch_tp(monkeypatch, make_communicator())
    _print_warning_once.cache_clear()
    flag_on = SimpleNamespace(
        parallel_config=ParallelConfig(enable_hc_sp_quick_reduce=True)
    )
    flag_off = SimpleNamespace(
        parallel_config=ParallelConfig(enable_hc_sp_quick_reduce=False)
    )

    assert common_sp.use_quick_sp(flag_off) is False
    assert "--enable-hc-sp-quick-reduce" not in caplog_vllm.text  # no flag, no warning
    assert common_sp.use_quick_sp(flag_on) is expect_quick
    # A flag that cannot be served warns, once, and says why it stays exact.
    assert caplog_vllm.text.count("--enable-hc-sp-quick-reduce") == int(
        not expect_quick
    )


# Qwen4Exp's AITER IPC sequence-parallel collectives (`--enable-hc-sp-aiter-ipc`).
# The fakes play AITER's custom all-reduce and the TP communicator, so these cover
# the selection and the fallbacks, not the kernels.
@pytest.fixture
def amd_sp():
    from vllm.models.qwen4_exp.amd.ops import sequence_parallel

    return sequence_parallel


class _FakeAiterComm:
    def __init__(self, eligible=True, disabled=False):
        self.eligible = eligible
        self.disabled = disabled
        self.calls = []

    def should_custom_rs(self, x, dim):
        return self.eligible

    def should_custom_ag(self, x):
        return self.eligible

    def custom_reduce_scatter(self, x, out, dim):
        self.calls.append(("reduce_scatter", x.shape, out.shape))

    def custom_all_gather(self, x, dim):
        self.calls.append(("all_gather", x.shape))
        return "aiter_ag"


def _patch_amd_sp(monkeypatch, amd_sp, communicator, aiter_comm):
    """The TP group's communicator and AITER's, as the Qwen helpers look them up;
    returns the record of the calls that reach the exact RCCL pair."""
    exact = _patch_tp(monkeypatch, communicator)
    monkeypatch.setattr(
        amd_sp,
        "get_tp_group",
        lambda: SimpleNamespace(device_communicator=communicator),
    )
    monkeypatch.setattr(amd_sp, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(
        aiter_ops.rocm_aiter_ops, "get_aiter_allreduce", lambda: aiter_comm
    )
    return exact


def test_sp_ipc_helpers_use_the_aiter_communicator(monkeypatch, amd_sp):
    aiter_comm = _FakeAiterComm()
    exact = _patch_amd_sp(monkeypatch, amd_sp, None, aiter_comm)

    reduced = amd_sp._sp_reduce_scatter_ipc(torch.zeros(7, 4))
    gathered = amd_sp._sp_all_gather_ipc(torch.zeros(3, 4))

    # reduce-scatter pads the rows to a multiple of the world size first
    assert aiter_comm.calls == [
        ("reduce_scatter", (8, 4), (4, 4)),
        ("all_gather", (3, 4)),
    ]
    assert reduced.shape == (4, 4) and gathered == "aiter_ag"
    assert not exact.rs and not exact.ag


@pytest.mark.parametrize(
    "aiter_comm",
    [
        pytest.param(None, id="no_communicator"),
        pytest.param(_FakeAiterComm(disabled=True), id="disabled"),
        pytest.param(_FakeAiterComm(eligible=False), id="declines_the_shape"),
    ],
)
def test_sp_ipc_helpers_fall_back_to_the_exact_collectives(
    monkeypatch, amd_sp, aiter_comm
):
    exact = _patch_amd_sp(monkeypatch, amd_sp, None, aiter_comm)

    assert amd_sp._sp_reduce_scatter_ipc(torch.zeros(8, 4)) == "exact_rs"
    assert amd_sp._sp_all_gather_ipc(torch.zeros(3, 4)) == "exact_ag"
    assert exact.rs == [(8, 4)] and exact.ag == [(3, 4)]
    assert aiter_comm is None or not aiter_comm.calls


def test_sp_quantized_helpers_use_the_quick_result_above_the_gate(monkeypatch, amd_sp):
    sentinel = torch.zeros(1)
    quick, aiter_comm = _FakeCommunicator(sentinel), _FakeAiterComm()
    exact = _patch_amd_sp(monkeypatch, amd_sp, quick, aiter_comm)

    assert amd_sp._sp_reduce_scatter_quantized(torch.zeros(7, 4)) is sentinel
    assert amd_sp._sp_all_gather_quantized(torch.zeros(3, 4)) is sentinel

    assert quick.seen == [("reduce_scatter", (8, 4)), ("all_gather", (3, 4))]
    assert not aiter_comm.calls and not exact.rs and not exact.ag


@pytest.mark.parametrize(
    "make_communicator",
    [
        pytest.param(lambda: _FakeCommunicator(None), id="quick_declines"),
        pytest.param(lambda: SimpleNamespace(), id="no_quick_methods"),
        pytest.param(lambda: None, id="no_communicator"),
    ],
)
def test_sp_quantized_helpers_fall_back_to_the_ipc_pair(
    monkeypatch, amd_sp, make_communicator
):
    aiter_comm = _FakeAiterComm()
    exact = _patch_amd_sp(monkeypatch, amd_sp, make_communicator(), aiter_comm)

    reduced = amd_sp._sp_reduce_scatter_quantized(torch.zeros(7, 4))
    gathered = amd_sp._sp_all_gather_quantized(torch.zeros(3, 4))

    assert aiter_comm.calls == [
        ("reduce_scatter", (8, 4), (4, 4)),
        ("all_gather", (3, 4)),
    ]
    assert reduced.shape == (4, 4) and gathered == "aiter_ag"
    assert not exact.rs and not exact.ag


def test_sp_quantized_helpers_fall_back_to_rccl_without_an_aiter_communicator(
    monkeypatch, amd_sp
):
    exact = _patch_amd_sp(monkeypatch, amd_sp, _FakeCommunicator(None), None)

    assert amd_sp._sp_reduce_scatter_quantized(torch.zeros(7, 4)) == "exact_rs"
    assert amd_sp._sp_all_gather_quantized(torch.zeros(3, 4)) == "exact_ag"
    assert exact.rs == [(8, 4)] and exact.ag == [(3, 4)]


@pytest.mark.parametrize(
    "op, rows, expected_rows",
    [
        ("sp_reduce_scatter", 7, 4),
        ("sp_shard", 7, 4),
        ("sp_reduce_scatter_quantized", 7, 4),
        ("sp_reduce_scatter_ipc", 7, 4),
        ("sp_all_gather_quantized", 3, 6),
        ("sp_all_gather_ipc", 3, 6),
    ],
)
def test_opaque_ops_are_registered_with_the_right_fakes(
    monkeypatch, amd_sp, op, rows, expected_rows
):
    """The compiler sees the ops through their fake implementations: a
    reduce-scatter or shard shrinks the rows to this rank's block, an all-gather
    grows them."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    monkeypatch.setattr(amd_sp, "get_tensor_model_parallel_world_size", lambda: 2)
    with FakeTensorMode():
        out = getattr(torch.ops.vllm, op)(torch.empty(rows, 4))
    assert out.shape == (expected_rows, 4)


@ray.remote(num_gpus=1, max_calls=1)
def graph_quickreduce(
    monkeypatch: pytest.MonkeyPatch,
    tp_size,
    pp_size,
    rank,
    distributed_init_port,
):
    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        m.delenv("HIP_VISIBLE_DEVICES", raising=False)
        m.delenv("ROCR_VISIBLE_DEVICES", raising=False)
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)
        ensure_model_parallel_initialized(tp_size, pp_size)
        group = get_tp_group().device_group
        fa = get_tp_group().device_communicator.qr_comm

        # A small all_reduce for warmup.
        # this is needed because device communicators might be created lazily
        # (e.g. NCCL). This will ensure that the communicator is initialized
        # before any communication happens, so that this group can be used for
        # graph capture immediately.
        data = torch.zeros(1)
        data = data.to(device=device)
        torch.distributed.all_reduce(data, group=group)
        torch.accelerator.synchronize()
        del data

        # we use the first group to communicate once
        # and the second group to communicate twice
        # and so on
        # this is used to demonstrate that each group can
        # communicate independently
        num_communication = rank // tp_size + 1

        for sz in test_sizes:
            for dtype in [torch.float16, torch.bfloat16]:
                with graph_capture(device=device) as graph_capture_context:
                    device_idx = torch.accelerator.current_device_index()
                    inp1 = torch.randint(1, 23, (sz,), dtype=dtype, device=device_idx)
                    inp2 = torch.randint(-23, 1, (sz,), dtype=dtype, device=device_idx)
                    _assert_quickreduce(fa, inp1)
                    _assert_quickreduce(fa, inp2)

                    torch.accelerator.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=graph_capture_context.stream):
                        for _ in range(num_communication):
                            out1 = tensor_model_parallel_all_reduce(inp1)
                            dist.all_reduce(inp1, group=group)
                            out2 = tensor_model_parallel_all_reduce(inp2)
                            dist.all_reduce(inp2, group=group)
                graph.replay()
                tol = _quickreduce_tolerance(fa.qr_quant_level)
                torch.testing.assert_close(out1, inp1, **tol)
                torch.testing.assert_close(out2, inp2, **tol)


@ray.remote(num_gpus=1, max_calls=1)
def eager_quickreduce(
    monkeypatch: pytest.MonkeyPatch,
    tp_size,
    pp_size,
    rank,
    distributed_init_port,
):
    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        m.delenv("HIP_VISIBLE_DEVICES", raising=False)
        m.delenv("ROCR_VISIBLE_DEVICES", raising=False)
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)

        init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)

        # Size over 8MB is sufficient for custom quick allreduce.
        sz = 16 * 1024 * 1024
        fa = get_tp_group().device_communicator.qr_comm
        inp = torch.tensor(
            [1.0 * ((i) % 23) for i in range(sz)], dtype=torch.float16, device=device
        )
        _assert_quickreduce(fa, inp)
        out = fa.quick_all_reduce(inp)
        tol = _quickreduce_tolerance(fa.qr_quant_level)
        torch.testing.assert_close(out, inp * tp_size, **tol)

        inp = torch.tensor(
            [1.0 * ((i) % 23) for i in range(sz)], dtype=torch.bfloat16, device=device
        )
        _assert_quickreduce(fa, inp)
        out = fa.quick_all_reduce(inp)
        torch.testing.assert_close(out, inp * tp_size, **tol)


@ray.remote(num_gpus=1, max_calls=1)
def bf16_cast_quickreduce(
    monkeypatch: pytest.MonkeyPatch,
    tp_size,
    pp_size,
    rank,
    distributed_init_port,
):
    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        m.delenv("HIP_VISIBLE_DEVICES", raising=False)
        m.delenv("ROCR_VISIBLE_DEVICES", raising=False)
        m.setenv("VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16", "1")
        device = torch.device(f"cuda:{rank}")
        torch.accelerator.set_device_index(device)
        init_test_distributed_environment(tp_size, pp_size, rank, distributed_init_port)

        sz = 16 * 1024 * 1024
        fa = get_tp_group().device_communicator.qr_comm
        inp = torch.tensor(
            [1.0 * (i % 23) for i in range(sz)], dtype=torch.bfloat16, device=device
        )
        _assert_quickreduce(fa, inp)
        assert fa.use_fp16_kernels
        out = fa.quick_all_reduce(inp)
        torch.testing.assert_close(out, inp * tp_size, atol=2.5, rtol=0.1)


@pytest.mark.skipif(
    not current_platform.is_rocm(), reason="only test quick allreduce for rocm"
)
@pytest.mark.parametrize("quant_mode", ["FP", "INT8", "INT6", "INT4", "INT3"])
@pytest.mark.parametrize("tp_size", [2])
@pytest.mark.parametrize("pipeline_parallel_size", [1, 2])
@pytest.mark.parametrize("test_target", [graph_quickreduce, eager_quickreduce])
def test_custom_quick_allreduce(
    monkeypatch: pytest.MonkeyPatch,
    tp_size,
    pipeline_parallel_size,
    test_target,
    quant_mode,
):
    world_size = tp_size * pipeline_parallel_size
    if world_size > torch.accelerator.device_count():
        pytest.skip("Not enough GPUs to run the test.")
    if test_target is graph_quickreduce and on_gfx942():
        pytest.xfail(
            "CUDA graph capture with quick reduce hits "
            "hipErrorStreamCaptureInvalidated on gfx942"
        )

    monkeypatch.setenv("VLLM_ROCM_QUICK_REDUCE_QUANTIZATION", quant_mode)

    multi_process_parallel(monkeypatch, tp_size, pipeline_parallel_size, test_target)


@pytest.mark.skipif(
    not current_platform.is_rocm(), reason="only test quick allreduce for rocm"
)
def test_custom_quick_allreduce_bf16_cast(monkeypatch: pytest.MonkeyPatch):
    if torch.accelerator.device_count() < 2:
        pytest.skip("Not enough GPUs to run the test.")
    monkeypatch.setenv("VLLM_ROCM_QUICK_REDUCE_QUANTIZATION", "FP")
    multi_process_parallel(monkeypatch, 2, 1, bf16_cast_quickreduce)


def qr_variable_input(rank, world_size):
    """When the tensor parallelism is set to 4 or 8, frequent changes
    in the input shape can cause QuickReduce to hang (this issue
    has been observed with the gpt_oss model).
    """
    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)
    qr_max_size = None  # MB
    _ptr = ops.init_custom_qr(rank, world_size, qr_max_size)
    ranks = []
    for i in range(world_size):
        ranks.append(i)
    if envs.VLLM_DISTRIBUTED_USE_SPLIT_GROUP:
        dist.init_process_group(
            backend="cpu:gloo,cuda:nccl",
            init_method="tcp://127.0.0.1:29500",
            rank=rank,
            world_size=world_size,
            device_id=device,
        )
    else:
        dist.init_process_group(
            backend="nccl",
            init_method="tcp://127.0.0.1:29500",
            rank=rank,
            world_size=world_size,
        )
    if envs.VLLM_DISTRIBUTED_USE_SPLIT_GROUP:
        cpu_group = torch.distributed.split_group(
            split_ranks=[ranks], backend="cpu:gloo,cuda:nccl"
        )
    else:
        cpu_group = torch.distributed.new_group(ranks, backend="nccl")

    handle = ops.qr_get_handle(_ptr)
    world_size = dist.get_world_size(group=cpu_group)
    handles = [None] * world_size
    dist.all_gather_object(handles, handle, group=cpu_group)
    ops.qr_open_handles(_ptr, handles)

    num = 1
    s1 = 1024
    while num < 50000:  # 50000 is sufficient to identify issues.
        dtype = torch.float16
        device_idx = torch.accelerator.current_device_index()
        if num % 2 == 0:
            s2 = 1024
            inp1 = torch.zeros((s1, s2), dtype=dtype, device=device_idx)
        else:
            s2 = 2048
            inp1 = torch.ones((s1, s2), dtype=dtype, device=device_idx)
        result = torch.empty_like(inp1)
        # FP = 0 INT8 = 1 INT6 = 2 INT4 = 3 INT3 = 4
        ops.qr_all_reduce(_ptr, inp1, result, 3, cast_bf2half=True)
        try:
            if inp1[0, 0] == 0:
                assert torch.all(result == 0)
            else:
                assert torch.all(result == world_size)
        except AssertionError:
            print("Assertion failed! Allreduce results are incorrect.")
            raise
        num += 1


@pytest.mark.skipif(
    not current_platform.is_rocm(), reason="only test quick allreduce for rocm"
)
@pytest.mark.parametrize("tp_size", [4, 8])
@pytest.mark.parametrize("pipeline_parallel_size", [1])
def test_custom_quick_allreduce_variable_input(tp_size, pipeline_parallel_size):
    world_size = tp_size * pipeline_parallel_size
    if world_size > torch.accelerator.device_count():
        pytest.skip("Not enough GPUs to run the test.")

    multiprocessing.set_start_method("spawn", force=True)
    # 60s is enough
    timeout = 60
    processes = []
    for rank in range(tp_size):
        p = multiprocessing.Process(target=qr_variable_input, args=(rank, tp_size))
        p.start()
        processes.append((rank, p))
    for rank, p in processes:
        p.join(timeout=timeout)
        if p.is_alive():
            for r, proc in processes:
                if proc.is_alive():
                    proc.terminate()
                    proc.join()
            raise RuntimeError(f"QuickReduce hang detected after {timeout} seconds!")


if __name__ == "__main__":
    test_custom_quick_allreduce_variable_input(tp_size=4, pipeline_parallel_size=1)
