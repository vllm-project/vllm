# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of the sequence-parallel entry points of QuickAllReduce.

AITER's FlyDSL INT4 reduce-scatter and all-gather are replaced by a fake that
plays rank 0 of 2, so these tests cover the gating, the padding to whole tiles
and the unpadding, not the kernels.
"""

from types import SimpleNamespace

import pytest
import torch

import vllm.envs as envs
from vllm.distributed.device_communicators.quick_all_reduce import (
    MB,
    QuickAllReduce,
    QuickReduceRegime,
)
from vllm.envs import disable_envs_cache
from vllm.models.common.ops import sequence_parallel as common_sp

HIDDEN = 2560  # a row is 5,120 B: the kernels' 32 KiB tile holds 6.4 rows
UNIT = 32  # rows per whole number of tiles at this width: 32 * 5,120 B = 5 tiles


class _FakeFlyDSLSp:
    """Plays rank 0 of 2. The peer's all-gather block is `block + 1`."""

    def __init__(self):
        self.calls = []

    def reduce_scatter(self, inp, out):
        self._check_aligned(inp, out)
        self.calls.append(("reduce_scatter", inp, out))
        out.copy_(inp[: out.shape[0]])

    def all_gather(self, inp, out):
        self._check_aligned(inp, out)
        self.calls.append(("all_gather", inp, out))
        out[: inp.shape[0]] = inp
        out[inp.shape[0] :] = inp + 1

    @staticmethod
    def _check_aligned(inp, out):
        # AITER's QuickAllReduceInt4._check_tensors raises on misaligned buffers.
        if inp.data_ptr() % 16 or out.data_ptr() % 16:
            raise ValueError("requires 16-byte-aligned input/output")

    def compile_sp(self, full, shard):
        self.calls.append(("compile_sp", full, shard))

    def close(self):
        pass


def _make_quick_allreduce(
    *,
    min_size: int | None = None,
    max_size: int = 64 * MB,
    quantization_min_size: int | None = None,
    enabled: bool = True,
) -> tuple[QuickAllReduce, _FakeFlyDSLSp]:
    quick_reduce = QuickAllReduce.__new__(QuickAllReduce)
    fake = _FakeFlyDSLSp()
    quick_reduce._flydsl_int4 = fake
    quick_reduce._flydsl_sp = enabled
    quick_reduce.disabled = False
    quick_reduce.qr_max_size = max_size
    quick_reduce.qr_min_size = min_size
    quick_reduce.qr_quant_level = QuickReduceRegime.INT4
    quick_reduce.qr_quantization_min_size = quantization_min_size
    quick_reduce.world_size = 2
    return quick_reduce, fake


def _rows(num_rows: int) -> torch.Tensor:
    """bf16 rows that differ between the two halves of a 2 * 1,030-row tensor.

    Integers up to 255 are exact in bf16, so the expected values are exact.
    """
    values = torch.arange(num_rows)[:, None] % 251 + torch.arange(HIDDEN)[None, :] % 5
    return values.to(torch.bfloat16)


def test_reduce_scatter_pads_each_block_to_whole_tiles():
    quick_reduce, fake = _make_quick_allreduce()
    chunk, padded = 1030, 1056  # 33 * UNIT
    inp = _rows(2 * chunk)

    out = quick_reduce.quick_reduce_scatter(inp)

    ((name, src, kernel_out),) = fake.calls
    assert name == "reduce_scatter"
    assert src.shape == (2 * padded, HIDDEN) and kernel_out.shape == (padded, HIDDEN)
    for rank in range(2):  # every block keeps its rows, then zeros
        block = src[rank * padded : (rank + 1) * padded]
        assert torch.equal(block[:chunk], inp[rank * chunk : (rank + 1) * chunk])
        assert not block[chunk:].any()
    assert out.shape == (chunk, HIDDEN)
    assert torch.equal(out, inp[:chunk])  # the fake's rank-0 block, unpadded


def test_all_gather_pads_each_block_and_keeps_rank_order():
    quick_reduce, fake = _make_quick_allreduce()
    chunk, padded = 1030, 1056
    inp = _rows(chunk)

    out = quick_reduce.quick_all_gather(inp)

    ((name, src, kernel_out),) = fake.calls
    assert name == "all_gather"
    assert src.shape == (padded, HIDDEN) and kernel_out.shape == (2 * padded, HIDDEN)
    assert torch.equal(src[:chunk], inp) and not src[chunk:].any()
    assert out.shape == (2 * chunk, HIDDEN)
    assert torch.equal(out[:chunk], inp)  # rank 0, then the fake's rank 1
    assert torch.equal(out[chunk:], inp + 1)


def test_whole_tile_blocks_are_not_copied():
    quick_reduce, fake = _make_quick_allreduce()
    chunk = 1024  # a multiple of UNIT
    inp = _rows(2 * chunk)
    shard = _rows(chunk)

    out_rs = quick_reduce.quick_reduce_scatter(inp)
    out_ag = quick_reduce.quick_all_gather(shard)

    (_, rs_src, rs_out), (_, ag_src, ag_out) = fake.calls
    assert rs_src is inp and ag_src is shard
    assert out_rs is rs_out and out_ag is ag_out


@pytest.mark.parametrize(
    "case",
    ["disabled_switch", "fp16", "non_contiguous", "below_min_size", "above_max_size"]
    + ["below_quantization_min_size", "one_dimensional"],
)
def test_ineligible_inputs_return_none(case):
    kwargs = {}
    inp = _rows(2 * 1024)
    if case == "disabled_switch":
        kwargs["enabled"] = False
    elif case == "fp16":
        inp = inp.to(torch.float16)
    elif case == "non_contiguous":
        inp = _rows(4 * 1024)[::2]
    elif case == "below_min_size":
        inp = _rows(2 * 64)  # 655 KB against the 8 MB built-in TP2 INT4 minimum
    elif case == "above_max_size":
        kwargs["max_size"] = 8 * MB  # the 10.5 MB message does not fit
    elif case == "below_quantization_min_size":
        kwargs["quantization_min_size"] = 32 * MB
    elif case == "one_dimensional":
        inp = inp.reshape(-1)
    quick_reduce, fake = _make_quick_allreduce(**kwargs)

    assert quick_reduce.quick_reduce_scatter(inp) is None
    assert quick_reduce.quick_all_gather(inp) is None
    assert not fake.calls


def _misaligned(rows: int) -> torch.Tensor:
    """A contiguous bf16 [rows, HIDDEN] tensor 2 bytes past a 16-byte boundary."""
    storage = torch.empty(rows * HIDDEN + 8, dtype=torch.bfloat16)
    view = storage[1 : 1 + rows * HIDDEN].view(rows, HIDDEN)
    view.copy_(_rows(rows))
    assert view.data_ptr() % 16 == 2
    return view


def test_misaligned_input_is_copied_not_declined():
    """A misaligned view reaches the kernel as an aligned copy: every rank must
    take the same path, so it is not declined."""
    quick_reduce, fake = _make_quick_allreduce()
    inp = _misaligned(2 * 1024)  # whole tiles: no padding copy hides the alignment
    shard = _misaligned(1024)

    out_rs = quick_reduce.quick_reduce_scatter(inp)
    out_ag = quick_reduce.quick_all_gather(shard)

    (_, rs_src, _), (_, ag_src, _) = fake.calls
    assert rs_src is not inp and torch.equal(rs_src, inp)
    assert ag_src is not shard and torch.equal(ag_src, shard)
    assert torch.equal(out_rs, inp[:1024]) and torch.equal(out_ag[:1024], shard)


def test_reduce_scatter_needs_rows_divisible_by_world_size():
    quick_reduce, fake = _make_quick_allreduce()

    assert quick_reduce.quick_reduce_scatter(_rows(2 * 1024 + 1)) is None
    assert not fake.calls


def test_min_size_override_admits_small_messages():
    quick_reduce, fake = _make_quick_allreduce(min_size=0)

    assert quick_reduce.quick_all_gather(_rows(64)) is not None
    assert len(fake.calls) == 1


def _init_flydsl_sp(world_size: int, flydsl_int4, switch: str | None, monkeypatch):
    if switch is None:
        monkeypatch.delenv("VLLM_ROCM_SP_QUICK_REDUCE", raising=False)
    else:
        monkeypatch.setenv("VLLM_ROCM_SP_QUICK_REDUCE", switch)
    disable_envs_cache()
    quick_reduce, _ = _make_quick_allreduce()
    quick_reduce.world_size = world_size
    warmup = torch.empty(MB, dtype=torch.bfloat16)
    return quick_reduce._init_flydsl_sp(flydsl_int4, warmup), warmup


@pytest.fixture(autouse=True)
def _fresh_envs_cache():
    disable_envs_cache()
    yield
    disable_envs_cache()


def test_init_runs_compile_sp_once_when_switched_on(monkeypatch):
    fake = _FakeFlyDSLSp()

    enabled, warmup = _init_flydsl_sp(2, fake, "1", monkeypatch)

    assert enabled
    ((name, full, shard),) = fake.calls
    assert name == "compile_sp"
    assert full is warmup and shard.numel() == warmup.numel() // 2
    assert full.numel() * full.element_size() % (32 * 1024) == 0  # whole tiles


@pytest.mark.parametrize("switch", [None, "0"])
def test_init_is_inert_when_switched_off(monkeypatch, switch):
    fake = _FakeFlyDSLSp()

    enabled, _ = _init_flydsl_sp(2, fake, switch, monkeypatch)

    assert not enabled and not fake.calls


def test_init_stays_exact_for_other_world_sizes(monkeypatch):
    fake = _FakeFlyDSLSp()

    enabled, _ = _init_flydsl_sp(4, fake, "1", monkeypatch)

    assert not enabled and not fake.calls


def test_init_stays_exact_when_aiter_lacks_the_methods(monkeypatch):
    enabled, _ = _init_flydsl_sp(
        2, SimpleNamespace(allreduce=lambda *args: None), "1", monkeypatch
    )

    assert not enabled


def test_env_switch_defaults_to_off_and_parses(monkeypatch):
    monkeypatch.delenv("VLLM_ROCM_SP_QUICK_REDUCE", raising=False)
    assert envs.VLLM_ROCM_SP_QUICK_REDUCE is False
    monkeypatch.setenv("VLLM_ROCM_SP_QUICK_REDUCE", "1")
    disable_envs_cache()
    assert envs.VLLM_ROCM_SP_QUICK_REDUCE is True


class _FakeCommunicator:
    def __init__(self, result=None):
        self.result = result
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


def test_quantized_helpers_use_the_quick_result(monkeypatch):
    sentinel = torch.zeros(1)
    communicator = _FakeCommunicator(sentinel)
    exact = _patch_tp(monkeypatch, communicator)

    assert common_sp.sp_reduce_scatter_quantized(torch.zeros(7, 4)) is sentinel
    assert common_sp.sp_all_gather_quantized(torch.zeros(3, 4)) is sentinel

    # reduce-scatter pads the rows to a multiple of the world size first
    assert communicator.seen == [("reduce_scatter", (8, 4)), ("all_gather", (3, 4))]
    assert not exact.rs and not exact.ag


def test_quantized_helpers_fall_back_to_the_exact_collectives(monkeypatch):
    exact = _patch_tp(monkeypatch, _FakeCommunicator(None))

    assert common_sp.sp_reduce_scatter_quantized(torch.zeros(8, 4)) == "exact_rs"
    assert common_sp.sp_all_gather_quantized(torch.zeros(3, 4)) == "exact_ag"
    assert exact.rs == [(8, 4)] and exact.ag == [(3, 4)]


def test_quantized_helpers_fall_back_without_a_communicator(monkeypatch):
    exact = _patch_tp(monkeypatch, None)

    assert common_sp.sp_all_gather_quantized(torch.zeros(3, 4)) == "exact_ag"
    assert exact.ag == [(3, 4)]


def test_opaque_op_fakes_match_the_real_shapes(monkeypatch):
    from vllm.models.qwen4_exp.amd.ops import sequence_parallel as amd_sp

    monkeypatch.setattr(amd_sp, "get_tensor_model_parallel_world_size", lambda: 2)

    assert amd_sp._sp_local_fake(torch.empty(7, 4)).shape == (4, 4)
    assert amd_sp._sp_all_gather_fake(torch.empty(3, 4)).shape == (6, 4)
