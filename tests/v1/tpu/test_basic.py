# SPDX-License-Identifier: Apache-2.0
"""A basic correctness check for TPUs

Run `pytest tests/v1/tpu/test_basic.py`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from vllm.platforms import current_platform
import torch
from vllm.v1.worker.tpu_model_runner import TPUModelRunner, _get_slot_slices

if TYPE_CHECKING:
    from tests.conftest import VllmRunner

MODELS = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    # TODO: Enable this models with v6e
    # "Qwen/Qwen2-7B-Instruct",
    # "meta-llama/Llama-3.1-8B",
]

TENSOR_PARALLEL_SIZES = [1]

# TODO: Enable when CI/CD will have a multi-tpu instance
# TENSOR_PARALLEL_SIZES = [1, 4]


@pytest.mark.skipif(not current_platform.is_tpu(),
                    reason="This is a basic test for TPU only")
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("max_tokens", [5])
@pytest.mark.parametrize("enforce_eager", [True])
@pytest.mark.parametrize("tensor_parallel_size", TENSOR_PARALLEL_SIZES)
def test_models(
    vllm_runner: type[VllmRunner],
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    max_tokens: int,
    enforce_eager: bool,
    tensor_parallel_size: int,
) -> None:
    prompt = "The next numbers of the sequence " + ", ".join(
        str(i) for i in range(1024)) + " are:"
    example_prompts = [prompt]

    with monkeypatch.context() as m:
        m.setenv("VLLM_USE_V1", "1")

        with vllm_runner(
                model,
                max_model_len=8192,
                enforce_eager=enforce_eager,
                gpu_memory_utilization=0.7,
                max_num_seqs=16,
                tensor_parallel_size=tensor_parallel_size) as vllm_model:
            vllm_outputs = vllm_model.generate_greedy(example_prompts,
                                                      max_tokens)
        output = vllm_outputs[0][1]
        assert "1024" in output

def test_creating_write_to_kvcache_slices1():
  block_numbers = torch.tensor([ 0,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10])
  block_offsets = torch.tensor([ 9,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11,  0,  1,  2,  3, 4,  5,  6,  7,  8,  9, 10])
  num_scheduled_tokens_per_req = [1, 12, 11]
  page_size=16

  slot_slices = _get_slot_slices(block_numbers, block_offsets, num_scheduled_tokens_per_req, page_size)

  assert slot_slices.shape[0] == 3
  assert slot_slices.shape[1] == 3
  assert torch.equal(slot_slices[0], torch.tensor([0, 5, 10]))  # physical_page_ids
  assert torch.equal(slot_slices[1], torch.tensor([9, 0, 0]))  # token_start_ids
  assert torch.equal(slot_slices[2], torch.tensor([1, 12, 11]))  # slice_size

  
def test_creating_write_to_kvcache_slices2():
  block_numbers = torch.tensor([ 5, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16])
  block_offsets = torch.tensor([ 7,  9, 10, 11, 12, 13, 14, 15,  0,  1,  2,  3,  4,  5,  6,  7,  8, 9, 10, 11, 12, 13, 14, 15,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9])
  num_scheduled_tokens_per_req = [1, 37, 26]
  page_size=16

  slot_slices = _get_slot_slices(block_numbers, block_offsets, num_scheduled_tokens_per_req, page_size)

  assert slot_slices.shape[0] == 3
  assert slot_slices.shape[1] == 6
  assert torch.equal(slot_slices[0], torch.tensor([5, 10, 11, 12, 15, 16]))  # physical_page_ids
  assert torch.equal(slot_slices[1], torch.tensor([7, 9, 0, 0, 0, 0]))  # token_start_ids
  assert torch.equal(slot_slices[2], torch.tensor([1, 7, 16, 14, 16, 10]))  # slice_size
