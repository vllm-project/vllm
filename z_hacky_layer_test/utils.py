from typing import Tuple
from py_bok import AttentionLayerDimensions, DeviceDataType, RotaryEmbedding
import torch


def to_torch_data_type(device_data_type: DeviceDataType) -> torch.dtype:
    """
    Convert DeviceDataType enum to corresponding PyTorch dtype.

    Args:
        device_data_type: DeviceDataType enum value

    Returns:
        Corresponding torch.dtype

    Raises:
        ValueError: If the DeviceDataType is not supported
    """
    dtype_mapping = {
        DeviceDataType.FP16: torch.float16,
        DeviceDataType.BF16: torch.bfloat16,
        DeviceDataType.FP32: torch.float32,
        DeviceDataType.INT8: torch.int8,
        DeviceDataType.INT16: torch.int16,
        DeviceDataType.INT32: torch.int32,
        DeviceDataType.UINT8: torch.uint8,
        DeviceDataType.FP8_E4M3: torch.float8_e4m3fn,
        DeviceDataType.FP8_E5M2: torch.float8_e5m2,
    }

    if device_data_type not in dtype_mapping:
        raise ValueError(f"Unsupported DeviceDataType: {device_data_type}")

    return dtype_mapping[device_data_type]


def identity_rotary_cos_sin(
    rotary_embedding_parameters: RotaryEmbedding,
) -> torch.Tensor:
    """
    Creates a rotary positional embedding cosine and sine cache for a given rotary embedding, where
    all the cosine values are ones and all the sine values are zero,
    which should create an identity embedding, i.e. an embedding that does nothing.
    """

    # One vector of ones for the cos values.
    cos_values = torch.ones(
        rotary_embedding_parameters.rotaryEmbeddingMaxPositions,
        rotary_embedding_parameters.rotaryEmbeddingDim,
        device=torch.device("cuda"),
        dtype=torch.float32,
    )

    # One vector of zeros for the sin values.
    sin_values = torch.zeros(
        rotary_embedding_parameters.rotaryEmbeddingMaxPositions,
        rotary_embedding_parameters.rotaryEmbeddingDim,
        device=torch.device("cuda"),
        dtype=torch.float32,
    )

    # Stack them together to get the cos and sin values for each position.
    result = torch.stack([cos_values, sin_values], dim=2)
    assert result.shape == (
        rotary_embedding_parameters.rotaryEmbeddingMaxPositions,
        rotary_embedding_parameters.rotaryEmbeddingDim,
        2,
    )
    return result


def generate_constant_attention_input(
    num_tokens: int,
    attention_layer_dimensions: AttentionLayerDimensions,
    data_type: DeviceDataType,
    value: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate constant attention input tensors (Q, K, V) filled with a specified value.

    Args:
        num_tokens: Number of tokens
        attention_layer_dimensions: Object with numQHeads, numKVHeads, headSize attributes
        data_type: Device data type (will be converted to torch dtype)
        value: Constant value to fill the tensors with

    Returns:
        Tuple of (q, k, v) tensors
    """
    device = torch.device("cuda")

    # Convert DeviceDataType to torch dtype
    dtype = to_torch_data_type(data_type)

    tensor_options = {
        "dtype": dtype,
        "device": device,
        "requires_grad": False,
    }

    # Create Q tensor: [num_tokens, num_q_heads, head_size]
    q = torch.full(
        (
            num_tokens,
            attention_layer_dimensions.numQHeads,
            attention_layer_dimensions.headSize,
        ),
        value,
        **tensor_options,
    )

    # Create K tensor: [num_tokens, num_kv_heads, head_size]
    k = torch.full(
        (
            num_tokens,
            attention_layer_dimensions.numKVHeads,
            attention_layer_dimensions.headSize,
        ),
        value,
        **tensor_options,
    )

    # Create V tensor: [num_tokens, num_kv_heads, head_size]
    v = torch.full(
        (
            num_tokens,
            attention_layer_dimensions.numKVHeads,
            attention_layer_dimensions.headSize,
        ),
        value,
        **tensor_options,
    )

    return q, k, v


def pack_attention_input(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """
    Pack attention input tensors (Q, K, V) into a single concatenated tensor.

    Args:
        q: Query tensor with shape [num_tokens, num_q_heads, head_dim]
        k: Key tensor with shape [num_tokens, num_kv_heads, head_dim]
        v: Value tensor with shape [num_tokens, num_kv_heads, head_dim]

    Returns:
        Packed tensor with shape [num_tokens, num_q_heads * head_dim + num_kv_heads * head_dim + num_kv_heads * head_dim]
    """
    num_tokens = q.size(0)

    # Flatten each tensor from [num_tokens, num_heads, head_dim] to [num_tokens, num_heads * head_dim]
    q_flat = q.view(num_tokens, -1)  # [num_tokens, num_q_heads * head_dim]
    k_flat = k.view(num_tokens, -1)  # [num_tokens, num_kv_heads * head_dim]
    v_flat = v.view(num_tokens, -1)  # [num_tokens, num_kv_heads * head_dim]

    # Concatenate along the second dimension
    return torch.cat([q_flat, k_flat, v_flat], dim=1)


def write_kv_cache_at_contiguous_offsets(
    k: torch.Tensor,
    v: torch.Tensor,
    sequence_lengths: torch.Tensor,
    kv_cache: torch.Tensor,
    max_num_blocks_per_sequence: int,
) -> torch.Tensor:
    """
    Write K and V tensors to KV cache at contiguous block offsets.

    Args:
        k: Key tensor with shape [num_tokens, num_kv_heads, head_dim]
        v: Value tensor with shape [num_tokens, num_kv_heads, head_dim]
        sequence_lengths: Tensor containing length of each sequence
        kv_cache: KV cache tensor to write to with shape [num_blocks, 2, num_kv_heads, tokens_per_block, head_dim]
        max_num_blocks_per_sequence: Maximum number of blocks per sequence

    Returns:
        Block offsets tensor with shape [num_sequences, 2, max_num_blocks_per_sequence]
    """
    # Check that the type of the kv_cache is the same as the type of the k and v tensors
    assert (
        kv_cache.dtype == k.dtype
    ), "The type of the kv_cache must be the same as the type of the k and v tensors"
    assert (
        kv_cache.dtype == v.dtype
    ), "The type of the kv_cache must be the same as the type of the k and v tensors"

    # Check that the sequence length of the k and v tensors is the same
    assert k.size(0) == v.size(0), "K and V tensors must have the same number of tokens"

    num_tokens_per_block = kv_cache.size(3)
    num_required_blocks = (
        ((sequence_lengths + num_tokens_per_block - 1) // num_tokens_per_block)
        .sum()
        .item()
    )
    num_available_blocks = kv_cache.size(0)

    assert num_required_blocks <= num_available_blocks, (
        f"The number of required blocks {num_required_blocks} is greater than "
        f"the number of available blocks {num_available_blocks}"
    )

    # Split k and v by sequence lengths
    sequence_lengths_cpu = sequence_lengths.to("cpu").to(torch.int64)
    sequence_lengths_list = sequence_lengths_cpu.tolist()

    k_split_by_sequence = k.split(sequence_lengths_list, dim=0)
    v_split_by_sequence = v.split(sequence_lengths_list, dim=0)

    current_block_offset = 0

    # Merge the '2' dimension of the kv_cache with the number of blocks
    # This makes indexing easier: [num_blocks * 2, num_kv_heads, tokens_per_block, head_dim]
    kv_cache_merged = kv_cache.reshape(
        kv_cache.size(0) * 2, kv_cache.size(2), kv_cache.size(3), kv_cache.size(4)
    )

    num_sequences = sequence_lengths.size(0)
    host_block_offsets = torch.empty(
        (num_sequences, 2, max_num_blocks_per_sequence), dtype=torch.int32, device="cpu"
    )

    for i in range(len(k_split_by_sequence)):
        sequence_length = sequence_lengths_list[i]

        # Process K tensor for this sequence
        sequence_k = k_split_by_sequence[i]
        k_offsets_row = host_block_offsets[i][0]
        current_block_offset = insert_k_or_v(
            current_block_offset,
            k_offsets_row,
            kv_cache_merged,
            sequence_k,
            sequence_length,
            num_tokens_per_block,
            kv_index=0,  # K is at index 0 in the merged cache
        )

        # Process V tensor for this sequence
        sequence_v = v_split_by_sequence[i]
        v_offsets_row = host_block_offsets[i][1]
        current_block_offset = insert_k_or_v(
            current_block_offset,
            v_offsets_row,
            kv_cache_merged,
            sequence_v,
            sequence_length,
            num_tokens_per_block,
            kv_index=1,  # V is at index 1 in the merged cache
        )

    return host_block_offsets.to(device="cuda", dtype=torch.int32)


def insert_k_or_v(
    current_block_offset: int,
    block_offsets: torch.Tensor,
    kv_cache_merged: torch.Tensor,
    kv_data: torch.Tensor,
    sequence_length: int,
    num_tokens_per_block: int,
    kv_index: int,
) -> int:
    """
    Insert K or V tensor data into the KV cache.

    This matches the original C++ insertKOrV function logic.

    Args:
        current_block_offset: Current block offset in the cache
        block_offsets: Row to store the block offsets for this sequence
        kv_cache_merged: Merged KV cache tensor [num_blocks*2, num_kv_heads, tokens_per_block, head_dim]
        kv_data: K or V tensor data for this sequence [seq_len, num_kv_heads, head_dim]
        sequence_length: Length of the sequence
        num_tokens_per_block: Number of tokens per block
        num_kv_heads: Number of KV heads
        head_dimension: Dimension of each head
        kv_index: 0 for K, 1 for V (determines offset in merged cache)

    Returns:
        Updated current_block_offset
    """
    sequence_token_offset = 0
    block_offset_in_sequence = 0

    while sequence_token_offset < sequence_length:
        num_tokens_to_copy = min(
            sequence_length - sequence_token_offset, num_tokens_per_block
        )

        # Calculate the actual block index in the merged cache
        # Each block has K at even indices (0, 2, 4, ...) and V at odd indices (1, 3, 5, ...)
        merged_block_index = current_block_offset * 2 + kv_index

        # Target slice in the cache: [num_kv_heads, num_tokens_to_copy, head_dim]
        target_slice = kv_cache_merged[
            merged_block_index,
            :,  # all KV heads
            :num_tokens_to_copy,  # only the tokens we're copying
            :,  # all head dimensions
        ]

        # Source slice from the sequence data: [num_tokens_to_copy, num_kv_heads, head_dim]
        source_slice = kv_data[
            sequence_token_offset : sequence_token_offset + num_tokens_to_copy
        ]

        # Transpose source to match target: [num_kv_heads, num_tokens_to_copy, head_dim]
        source_slice_transposed = source_slice.transpose(0, 1)

        # Copy the data
        target_slice.copy_(source_slice_transposed)

        # Update offsets
        sequence_token_offset += num_tokens_to_copy
        block_offsets[block_offset_in_sequence] = current_block_offset
        current_block_offset += 1
        block_offset_in_sequence += 1

    return current_block_offset
