import torch

def sgmv_expand(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    b_seq_start_loc: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    batches: int,
    max_seq_length: int,
    token_nums: int,
    add_inputs: bool = False
):
    exploded_indices = torch.repeat_interleave(lora_indices_tensor, seq_len_tensor)

    bgmv_expand(
        inputs,
        lora_b_weights,
        output_tensor,
        exploded_indices,
        add_inputs
    )
    
    
def bgmv_expand(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    add_inputs: bool = True
):
    selected_loras = lora_b_weights[lora_indices_tensor].squeeze()
    inputs = inputs.to(dtype=torch.float16)
    outputs = torch.einsum("bi, boi -> bo", inputs, selected_loras)
    
    if add_inputs:
        output_tensor[:] += outputs[:]
    else:
        output_tensor[:] = outputs[:]
        
def sgmv_shrink(
    inputs: torch.Tensor,
    lora_a_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    b_seq_start_loc: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    batches: int,
    max_seq_length: int,
    token_nums: int,
    scaling: float,
):
    exploded_indices = torch.repeat_interleave(lora_indices_tensor, seq_len_tensor)

    bgmv_shrink(
        inputs,
        lora_a_weights,
        output_tensor,
        exploded_indices,
        scaling
    )
    
def bgmv_shrink(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    scaling: float = 1.0
):
    selected_loras = lora_b_weights[lora_indices_tensor].squeeze()
    outputs = torch.einsum("bi, boi -> bo", inputs, selected_loras)
    
    output_tensor[:] = scaling * outputs[:]
    
def sgmv_expand_slice(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    b_seq_start_loc: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    batches: int,
    max_seq_length: int,
    token_nums: int,
    slice_offset: int,
    slice_size: int,
    add_inputs: bool = False
):
    exploded_indices = torch.repeat_interleave(lora_indices_tensor, seq_len_tensor)

    bgmv_expand_slice(
        inputs,
        lora_b_weights,
        output_tensor,
        exploded_indices,
        slice_offset,
        slice_size,
        add_inputs
    )
    
    
def bgmv_expand_slice(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    slice_offset: int,
    slice_size: int,
    add_inputs: bool = True
):
    selected_loras = lora_b_weights[lora_indices_tensor].squeeze()
    inputs = inputs.to(dtype=torch.float16)
    outputs = torch.einsum("bi, boi -> bo", inputs, selected_loras)
    
    if add_inputs:
        output_tensor[:, slice_offset:slice_offset+slice_size] += outputs[:]
    else:
        output_tensor[:, slice_offset:slice_offset+slice_size] = outputs[:]