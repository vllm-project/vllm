# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch-invariant LoRA shrink: fixed split-K partials, fixed-order reduction."""

import json

import pytest
import torch

import vllm.lora.ops.triton_ops.lora_shrink_op as shrink_op
import vllm.lora.ops.triton_ops.utils as lora_utils
from vllm import SamplingParams, envs
from vllm.config.lora import LoRAConfig
from vllm.lora.ops.triton_ops import LoRAKernelMeta, lora_shrink
from vllm.lora.ops.triton_ops.utils import _LORA_A_PTR_DICT
from vllm.lora.request import LoRARequest
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON
from vllm.utils.torch_utils import set_random_seed

DEVICE = "cuda"
pytestmark = pytest.mark.skipif(
    not (
        (current_platform.is_cuda() and current_platform.has_device_capability(80))
        or (current_platform.is_rocm() and HAS_TRITON)
    ),
    reason="Requires CUDA >= Ampere (SM80) or ROCm",
)


@pytest.fixture(
    params=[(True, "atomic"), (True, "kernel"), (False, "kernel")],
    ids=[
        "batch-invariant-overrides-atomic",
        "batch-invariant-kernel",
        "lora-only-kernel",
    ],
)
def batch_invariant(monkeypatch, request):
    enabled, reduction = request.param
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1" if enabled else "0")
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", enabled)
    monkeypatch.delenv("VLLM_TUNED_CONFIG_FOLDER", raising=False)
    monkeypatch.setattr(lora_utils, "is_batch_invariant", enabled)
    lora_utils.load_lora_op_config.cache_clear()
    lora_utils.get_lora_op_configs.cache_clear()
    yield LoRAConfig(split_k_reduction=reduction)
    lora_utils.load_lora_op_config.cache_clear()
    lora_utils.get_lora_op_configs.cache_clear()


def _shrink(inputs, weights, mapping, scaling, lora_config):
    num_tokens = inputs.size(0)
    meta = LoRAKernelMeta.make(weights[0].size(0), num_tokens, device=DEVICE)
    meta.prepare_tensors(mapping)
    out = torch.zeros(
        (len(weights), num_tokens, weights[0].size(1)),
        dtype=torch.float32,
        device=DEVICE,
    )
    _LORA_A_PTR_DICT.clear()
    args = meta.meta_args(token_nums=num_tokens, specialize_active_lora=False)
    lora_shrink(
        inputs,
        weights,
        out,
        *args,
        scaling,
        split_k_reduction=lora_config.split_k_reduction,
    )
    return out


@pytest.mark.parametrize(
    "hidden_size,rank,nslices,dtype",
    [
        (2560, 8, 3, torch.bfloat16),
        (9728, 16, 1, torch.float16),
        (4097, 64, 1, torch.bfloat16),  # K tail block
        (1024, 8, 1, torch.float16),  # K < 8 * BLOCK_K: some splits are empty
    ],
)
def test_lora_shrink_matches_reference(
    batch_invariant, hidden_size, rank, nslices, dtype
):
    set_random_seed(0)
    inputs = torch.randn((33, hidden_size), dtype=dtype, device=DEVICE)
    weights = [
        torch.randn((2, rank, hidden_size), dtype=dtype, device=DEVICE)
        for _ in range(nslices)
    ]
    mapping = torch.randint(-1, 2, (33,), dtype=torch.int32, device=DEVICE)
    out = _shrink(inputs, weights, mapping, 0.5, batch_invariant).double().cpu()

    ids, x = mapping.cpu(), inputs.double().cpu()
    ref = torch.zeros_like(out)
    for s, weight in enumerate(weights):
        for lora_id in (0, 1):
            rows = ids == lora_id
            ref[s, rows] = 0.5 * x[rows] @ weight[lora_id].double().cpu().T
    torch.testing.assert_close(out, ref, rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize("nslices", [1, 3])
@pytest.mark.parametrize(
    "batch_invariant,tuned",
    [
        pytest.param((True, "atomic"), False, id="batch-invariant-overrides-atomic"),
        pytest.param((True, "kernel"), False, id="batch-invariant-kernel"),
        pytest.param((False, "kernel"), False, id="lora-only-kernel"),
        pytest.param((True, "atomic"), True, id="batch-invariant-tuned-atomic"),
        pytest.param((True, "kernel"), True, id="batch-invariant-tuned-kernel"),
    ],
    indirect=["batch_invariant"],
)
def test_lora_shrink_invariant_across_batch_sizes(
    batch_invariant, monkeypatch, nslices, tuned, tmp_path
):
    if tuned:
        gpu = torch.cuda.get_device_name().replace(" ", "_").replace("-", "_")
        config = {
            "block_m": 32,
            "block_n": 16,
            "block_k": 32,
            "split_k": 64,
            "num_warps": 4,
            "num_ctas": 1,
            "num_stages": 2,
        }
        configs = {"3": {str(nslices): {"1": {"2560": {"16": config}}}}}
        (tmp_path / f"{gpu}_SHRINK.json").write_text(json.dumps(configs))
        monkeypatch.setenv("VLLM_TUNED_CONFIG_FOLDER", str(tmp_path))
        lora_utils.load_lora_op_config.cache_clear()
        lora_utils.get_lora_op_configs.cache_clear()

    reduce_kernel, reduce_grids = shrink_op._lora_shrink_reduce_kernel, []

    class RecordReduce:
        def __getitem__(self, grid):
            reduce_grids.append(grid)
            return reduce_kernel[grid]

    monkeypatch.setattr(shrink_op, "_lora_shrink_reduce_kernel", RecordReduce())
    set_random_seed(0)
    # K = 2560 is ten BLOCK_K=256 blocks, so all 8 splits contribute.
    dtype, hidden_size = torch.bfloat16, 2560
    weights = [
        torch.randn((3, 16, hidden_size), dtype=dtype, device=DEVICE)
        for _ in range(nslices)
    ]
    target = torch.randn((1, hidden_size), dtype=dtype, device=DEVICE)
    target_id = torch.zeros(1, dtype=torch.int32, device=DEVICE)
    expected = _shrink(target, weights, target_id, 0.7, batch_invariant)[:, 0]
    for num_tokens in (31, 32, 33, 127, 128, 129):
        inputs = torch.randn((num_tokens, hidden_size), dtype=dtype, device=DEVICE)
        mapping = torch.randint(-1, 3, (num_tokens,), dtype=torch.int32, device=DEVICE)
        pos = num_tokens // 2
        inputs[pos], mapping[pos] = target[0], 0
        out = _shrink(inputs, weights, mapping, 0.7, batch_invariant)
        assert torch.equal(out[:, pos], expected), num_tokens
    assert len(reduce_grids) == 7


def test_batch_invariant_lora_ignores_tuned_shrink_config(
    vllm_runner, qwen3_lora_files, monkeypatch, tmp_path
):
    """BI generation must not read tuned LoRA files, even invalid ones."""
    gpu = torch.cuda.get_device_name().replace(" ", "_").replace("-", "_")
    (tmp_path / f"{gpu}_SHRINK.json").write_text("invalid JSON")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setenv("VLLM_TUNED_CONFIG_FOLDER", str(tmp_path))
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    with vllm_runner(
        "Qwen/Qwen3-0.6B",
        enable_lora=True,
        max_lora_rank=8,
        max_model_len=128,
        max_num_seqs=8,
        enforce_eager=True,
        enable_prefix_caching=False,
        kernel_config={"enable_flashinfer_autotune": False},
        gpu_memory_utilization=0.2,
    ) as runner:
        params = SamplingParams(temperature=0, seed=0, max_tokens=4, ignore_eos=True)
        lora = LoRARequest("Alice", 1, qwen3_lora_files)
        prompt = "Hello, my name is"
        (single,) = runner.llm.generate(
            [prompt], params, lora_request=lora, use_tqdm=False
        )
        batch = runner.llm.generate(
            [prompt] * 8, params, lora_request=lora, use_tqdm=False
        )
        assert len(single.outputs[0].token_ids) == 4
        assert all(
            output.outputs[0].token_ids == single.outputs[0].token_ids
            for output in batch
        )
