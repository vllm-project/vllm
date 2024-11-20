import triton

import triton.language  as tl
import torch

def swap_blocks(src_tensor: torch.Tensor, dst_tensor: torch.Tensor, block_mapping_tensor: torch.Tensor):
  
    block_mapping = block_mapping_tensor.tolist()
    
    for key, value in block_mapping:
        src_block = src_tensor[key]
        dst_tensor[value]= src_block
        
