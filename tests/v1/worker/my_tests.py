# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm.v1.worker.tpu_model_runner import TPUModelRunner, _get_slot_slices

def test_creating_write_to_kvcache_slices():
  block_numbers = torch.tensor([ 0,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10])
  block_offsets = torch.tensor([ 9,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11,  0,  1,  2,  3, 4,  5,  6,  7,  8,  9, 10])
  num_scheduled_tokens_per_req = [1, 12, 11]
  page_size=16
  slot_slices = _get_slot_slices(block_numbers, block_offsets, num_scheduled_tokens_per_req, page_size)
  assert slot_slices.shape == 3
  assert slot_slices[0] == torch.tensor([0, 9, 1])
  assert slot_slices[1] == torch.tensor([5, 0, 12])
  assert slot_slices[2] == torch.tensor([10, 0, 12])


