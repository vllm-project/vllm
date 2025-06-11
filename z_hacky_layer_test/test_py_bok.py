from dataclasses import dataclass
import pytest
import torch

from py_bok import (
    create_op,
    forward_inplace,
    calculate_workspace_size,
    AttentionLayerDimensions,
    DeviceDataType,
    RotaryEmbedding,
    RotaryScalingType,
    RotaryPositionalEmbeddingType,
    PrefixCacheConfiguration,
)

from utils import (
    generate_constant_attention_input,
    identity_rotary_cos_sin,
    pack_attention_input,
    write_kv_cache_at_contiguous_offsets,
)


@dataclass(frozen=True)
class ContextRequest:
    sequence_length: int


@dataclass(frozen=True)
class GenerationRequest:
    sequence_length: int


type Request = ContextRequest | GenerationRequest


@dataclass(frozen=True)
class ForwardInplaceTestCase:
    num_layers: int
    max_batch_size: int
    max_num_tokens: int

    num_q_heads: int
    num_kv_heads: int
    head_dim: int

    rotary_embedding_dim: int
    rotary_embedding_base: int
    rotary_embedding_max_positions: int

    max_attention_window_size: int

    num_tokens_per_block: int
    max_num_blocks_per_sequence: int

    requests: tuple[Request, ...]

    output_scaling_factor: float


test_cases = [
    # ForwardInplaceTestCase(
    #     num_layers=1,
    #     max_batch_size=64,
    #     max_num_tokens=(1 << 15),
    #     num_q_heads=96,
    #     num_kv_heads=8,
    #     head_dim=128,
    #     rotary_embedding_dim=128,
    #     rotary_embedding_base=10000,
    #     rotary_embedding_max_positions=2048,
    #     max_attention_window_size=(1 << 15),
    #     num_tokens_per_block=32,
    #     max_num_blocks_per_sequence=512,
    #     requests=(ContextRequest(sequence_length=1024),),
    #     output_scaling_factor=1.0,
    # ),
    ForwardInplaceTestCase(
        num_layers=1,
        max_batch_size=64,
        max_num_tokens=(1 << 15),
        num_q_heads=96,
        num_kv_heads=8,
        head_dim=128,
        rotary_embedding_dim=128,
        rotary_embedding_base=10000,
        rotary_embedding_max_positions=2048,
        max_attention_window_size=(1 << 15),
        num_tokens_per_block=32,
        max_num_blocks_per_sequence=512,
        requests=(
            ContextRequest(sequence_length=1024),
            ContextRequest(sequence_length=1024),
        ),
        output_scaling_factor=1.0,
    ),
]


@pytest.mark.parametrize("test_case", test_cases)
def test_forward_inplace(test_case: ForwardInplaceTestCase):
    stream = torch.cuda.Stream()

    dims = AttentionLayerDimensions()
    dims.numQHeads = test_case.num_q_heads
    dims.numKVHeads = test_case.num_kv_heads
    dims.headSize = test_case.head_dim

    rotary_embedding = RotaryEmbedding()
    rotary_embedding.type = RotaryPositionalEmbeddingType.GPT_NEOX
    rotary_embedding.rotaryEmbeddingDim = test_case.rotary_embedding_dim
    rotary_embedding.rotaryEmbeddingBase = test_case.rotary_embedding_base
    rotary_embedding.rotaryEmbeddingScale = 0
    rotary_embedding.rotaryEmbeddingMaxPositions = (
        test_case.rotary_embedding_max_positions
    )
    rotary_embedding.rotaryScalingType = RotaryScalingType.NONE

    prefix_cache_configuration = PrefixCacheConfiguration()
    prefix_cache_configuration.numTokensPerBlock = test_case.num_tokens_per_block
    prefix_cache_configuration.maxNumBlocksPerSequence = (
        test_case.max_num_blocks_per_sequence
    )
    prefix_cache_configuration.dataType = DeviceDataType.FP8_E4M3

    fp8_output_scaling = torch.tensor(
        [1.0], device=torch.device("cuda"), dtype=torch.float32
    )
    kv_scale_orig_quant = torch.tensor(
        [1.0], device=torch.device("cuda"), dtype=torch.float32
    )
    kv_scale_quant_orig = torch.tensor(
        [1.0], device=torch.device("cuda"), dtype=torch.float32
    )
    multi_block_semaphores = torch.zeros(
        test_case.max_batch_size * test_case.num_q_heads,
        device=torch.device("cuda"),
        dtype=torch.int32,
    )

    # Create a representation of the fixed parameters of the attention operation.
    op = create_op(
        inputDataType=DeviceDataType.BF16,
        outputDataType=DeviceDataType.FP8_E4M3,
        attentionLayerDimensions=dims,
        rotaryEmbedding=rotary_embedding,
        prefixCacheConfiguration=prefix_cache_configuration,
        qScaling=1.0,
        maxAttentionWindowSize=test_case.max_attention_window_size,
        cyclicAttentionWindowSize=test_case.max_attention_window_size,
        fp8OutputScaling=fp8_output_scaling,
        kvScaleOrigQuant=kv_scale_orig_quant,
        kvScaleQuantOrig=kv_scale_quant_orig,
        multiBlockSemaphores=multi_block_semaphores,
    )

    # Calculate how much workspace memory is needed for the operation.
    workspace_size = calculate_workspace_size(
        op, test_case.max_num_tokens, test_case.max_batch_size
    )
    workspace = torch.zeros(
        workspace_size, device=torch.device("cuda"), dtype=torch.int8
    )

    batch_size = len(test_case.requests)

    q_dimension = test_case.num_q_heads * test_case.head_dim
    output = torch.zeros(
        batch_size,
        q_dimension,
        device=torch.device("cuda"),
        dtype=torch.int8,
    )

    # Create a representation of the rotary positional embedding cosine and sine cache.
    rotary_cos_sin = identity_rotary_cos_sin(rotary_embedding)

    output_scaling_factor = torch.tensor(
        [test_case.output_scaling_factor],
        device=torch.device("cuda"),
        dtype=torch.float32,
    )

    num_blocks_in_kv_cache = (
        test_case.max_batch_size * prefix_cache_configuration.maxNumBlocksPerSequence
    )
    actual_kv_cache_pools = [
        torch.zeros(
            num_blocks_in_kv_cache,
            2,
            test_case.num_kv_heads,
            prefix_cache_configuration.numTokensPerBlock,
            test_case.head_dim,
            device=torch.device("cuda"),
            dtype=torch.float8_e4m3fn,
        )
        for _ in range(test_case.num_layers)
    ]

    def _context_length(request: Request) -> int:
        match request:
            case ContextRequest(sequence_length=sequence_length):
                return sequence_length
            case GenerationRequest(sequence_length=sequence_length):
                return 0
            case _:
                raise ValueError(f"Invalid request: {request}")

    def _sequence_length(request: Request) -> int:
        match request:
            case ContextRequest(sequence_length=sequence_length):
                return sequence_length
            case GenerationRequest(sequence_length=sequence_length):
                return sequence_length
            case _:
                raise ValueError(f"Invalid request: {request}")

    context_lengths_host = torch.tensor(
        [_context_length(request) for request in test_case.requests],
        dtype=torch.int32,
    )

    sequence_lengths = torch.tensor(
        [_sequence_length(request) for request in test_case.requests],
        dtype=torch.int32,
    )
    sequence_lengths_device = sequence_lengths.cuda()
    past_kv_lengths_host = sequence_lengths - 1

    num_context_requests = sum(
        isinstance(request, ContextRequest) for request in test_case.requests
    )

    # Rough simulation of a realistic workload where the operation would be called for each layer.
    num_tokens = sequence_lengths.sum().item()
    for layer_index in range(test_case.num_layers):

        (q, k, v) = generate_constant_attention_input(
            num_tokens,
            dims,
            DeviceDataType.BF16,
            1.0,
        )

        qkv = pack_attention_input(q, k, v)

        kv_cache_block_offsets = write_kv_cache_at_contiguous_offsets(
            k.to(torch.float8_e4m3fn),
            v.to(torch.float8_e4m3fn),
            sequence_lengths_device,
            actual_kv_cache_pools[layer_index],
            test_case.max_num_blocks_per_sequence,
        )
        # Run the attention operation.
        stream.synchronize()
        forward_inplace(
            op,
            qkv,
            num_context_requests,
            context_lengths_host.to(torch.uint32),
            sequence_lengths_device.to(torch.uint32),
            past_kv_lengths_host.to(torch.uint32),
            kv_cache_block_offsets.to(torch.uint32),
            actual_kv_cache_pools[layer_index].data_ptr(),
            output_scaling_factor,
            rotary_cos_sin,
            output,
            workspace,
            stream.cuda_stream,
        )
        stream.synchronize()
