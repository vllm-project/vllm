# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm.v1.worker.tpu_model_runner import TPUModelRunner, _get_slot_slices

def test_creating_write_to_kvcache_slices1():
  block_numbers = torch.tensor([ 0,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10])
  block_offsets = torch.tensor([ 9,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11,  0,  1,  2,  3, 4,  5,  6,  7,  8,  9, 10])
  num_scheduled_tokens_per_req = [1, 12, 11]
  page_size=16
  slot_slices = _get_slot_slices(block_numbers, block_offsets, num_scheduled_tokens_per_req, page_size)
  print(f'xw32 {slot_slices=}')
  assert slot_slices.shape[0] == 3
  assert torch.equal(slot_slices[0], torch.tensor([9, 1, 0])) # (start, size, physical_page_id)
  assert torch.equal(slot_slices[1], torch.tensor([0, 12, 5]))
  assert torch.equal(slot_slices[2], torch.tensor([0, 11, 10]))
  
def test_creating_write_to_kvcache_slices2():
  block_numbers = torch.tensor([ 5, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16])
  block_offsets = torch.tensor([ 7,  9, 10, 11, 12, 13, 14, 15,  0,  1,  2,  3,  4,  5,  6,  7,  8, 9, 10, 11, 12, 13, 14, 15,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9])
  num_scheduled_tokens_per_req = [1, 37, 26]
  page_size=16
  slot_slices = _get_slot_slices(block_numbers, block_offsets, num_scheduled_tokens_per_req, page_size)
  assert slot_slices.shape[0] == 6
  assert torch.equal(slot_slices[0], torch.tensor([7, 1, 5])) # (start, size, physical_page_id)
  assert torch.equal(slot_slices[1], torch.tensor([9, 7, 10]))
  assert torch.equal(slot_slices[2], torch.tensor([0, 16, 11]))
  assert torch.equal(slot_slices[3], torch.tensor([0, 14, 12]))
  assert torch.equal(slot_slices[4], torch.tensor([0, 16, 15]))
  assert torch.equal(slot_slices[5], torch.tensor([0, 10, 16]))
  # assert slot_slices[0] == torch.tensor([7, 1, 5]) # (start, size, physical_page_id)
  # assert slot_slices[1] == torch.tensor([9, 7, 10])
  # assert slot_slices[2] == torch.tensor([0, 16, 11])
  # assert slot_slices[3] == torch.tensor([0, 14, 12])
  # assert slot_slices[4] == torch.tensor([0, 16, 15])
  # assert slot_slices[5] == torch.tensor([0, 10, 16])
  
  


