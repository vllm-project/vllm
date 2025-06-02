import torch

from py_bok import (
    create_op,
    forward_inplace,
    AttentionLayerDimensions,
    DeviceDataType,
    RotaryEmbedding,
    RotaryScalingType,
    RotaryPositionalEmbeddingType,
)


def test_create_attention_op():
    stream = torch.cuda.Stream()

    # Create basic parameters for a smoke test

    # Model dimensions.
    num_layers = 1

    # Attention layer dimensions.
    num_q_heads = 96
    num_kv_heads = 8
    head_dim = 128
    dims = AttentionLayerDimensions(num_q_heads, num_kv_heads, head_dim)
    max_attention_window_size = 2048

    # Rotary embedding dimensions.
    rotary_embedding_dim = 128
    rotary_embedding_base = 10000
    rotary_embedding_max_positions = 2048
    rotary_embedding_scale = 0
    rotary_embedding = RotaryEmbedding(
        type=RotaryPositionalEmbeddingType.GPT_NEOX,
        rotaryEmbeddingDim=rotary_embedding_dim,
        rotaryEmbeddingBase=rotary_embedding_base,
        rotaryEmbeddingScale=rotary_embedding_scale,
        rotaryEmbeddingMaxPositions=rotary_embedding_max_positions,
        rotaryScalingType=RotaryScalingType.NONE,
    )
    # KV-cache dimensions.
    num_tokens_per_block = 32
    max_blocks_per_sequence = 32

    # Create the attention operation
    fp8_output_scaling = torch.tensor(
        [1.0], device=torch.device("cuda"), dtype=torch.float32
    )
    kv_scale_orig_quant = torch.tensor(
        [1.0], device=torch.device("cuda"), dtype=torch.float32
    )
    kv_scale_quant_orig = torch.tensor(
        [1.0], device=torch.device("cuda"), dtype=torch.float32
    )
    op = create_op(
        inputDataType=DeviceDataType.BF16,
        outputDataType=DeviceDataType.BF16,
        attentionLayerDimensions=dims,
        rotaryEmbedding=rotary_embedding,
        numTokensPerBlock=num_tokens_per_block,
        qScaling=1.0,
        maxAttentionWindowSize=max_attention_window_size,
        cyclicAttentionWindowSize=max_attention_window_size,
        maxNumBlocksPerSequence=max_blocks_per_sequence,
        fp8OutputScaling=fp8_output_scaling,
        kvScaleOrigQuant=kv_scale_orig_quant,
        kvScaleQuantOrig=kv_scale_quant_orig,
    )

    # Parameters of the particular op call.
    batch_size = 1
    context_length = 32
    sequence_length = context_length + 1
    context_lengths_host = torch.tensor([context_length], dtype=torch.uint32)
    context_lengths_device = context_lengths_host.cuda()
    sequence_lengths = torch.tensor(
        [sequence_length], device=torch.device("cuda"), dtype=torch.uint32
    )
    past_kv_lengths_host = torch.tensor([context_length], dtype=torch.uint32)
    kv_cache_block_offsets = torch.zeros(
        batch_size,
        max_blocks_per_sequence,
        2,
        device=torch.device("cuda"),
        dtype=torch.uint32,
    )

    q_size = num_q_heads * head_dim
    kv_size = num_kv_heads * head_dim
    qkv_size = q_size + 2 * kv_size
    input_tensor = torch.randn(batch_size, qkv_size, device=torch.device("cuda"))

    actual_kv_cache_pools = [
        torch.zeros(
            num_tokens_per_block * 2 * num_kv_heads * head_dim,
            device=torch.device("cuda"),
        )
        for _ in range(num_layers)
    ]
    kv_cache_pool_mapping_host = torch.zeros(num_layers, dtype=torch.uint32)
    kv_cache_pool_pointers_host = torch.tensor(
        [t.data_ptr() for t in actual_kv_cache_pools], dtype=torch.int64
    )
    host_is_generation_flags = torch.tensor([True], dtype=torch.bool)
    rotary_cos_sin = torch.zeros(
        rotary_embedding_max_positions,
        rotary_embedding_dim,
        2,
        device=torch.device("cuda"),
        dtype=torch.float32,
    )
    output_scaling = torch.tensor(
        [1.0], device=torch.device("cuda"), dtype=torch.float32
    )
    output = torch.zeros(batch_size, q_size, device=torch.device("cuda"))
    workspace = torch.zeros(1 << 30, device=torch.device("cuda"))
    layer_index = 0
    forward_inplace(
        op,
        input_tensor,  # 2d device tensor of floats with dimensions (*, *)
        context_lengths_host,  # 1d host tensor of uint32_t's
        context_lengths_device,  # 1d device tensor of uint32_t's
        sequence_lengths,  # 1d device tensor of uint32_t's
        past_kv_lengths_host,  # 1d host tensor of uint32_t's
        kv_cache_block_offsets,  # 3d device tensor of offsets in the pool of uint32_t's with dimensions (*, *, 2)
        kv_cache_pool_pointers_host,  # 1d host tensor of pointers as int64_t's
        kv_cache_pool_mapping_host,  # 1d host tensor of uint32_t's
        host_is_generation_flags,  # 1d device tensor of bools
        output_scaling,  # 1d device tensor of floats
        rotary_cos_sin,  # Device tensor of floats with dimensions (*, * , 2)
        layer_index,  # int
        output,  # Any 2d device tensor
        workspace,  # Any device tensor,
        stream.cuda_stream,  # The stream to run the kernels on.
    )
