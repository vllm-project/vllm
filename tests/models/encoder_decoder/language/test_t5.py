"""Compare the outputs of HF and vLLM for T5 models using greedy sampling.
Based on tests/models/encoder_decoder/language/test_bart.py.

Run `pytest tests/models/encoder_decoder/language/test_t5.py`.
"""
from typing import List, Optional, Tuple, Type

import pytest
from transformers import AutoModelForSeq2SeqLM

from tests.kernels.utils import make_test_metadata
from vllm.attention.layer import Attention
from vllm.attention.selector import global_force_attn_backend_context_manager
from vllm.config import set_current_vllm_config

from ....conftest import (DecoderPromptType, ExplicitEncoderDecoderPrompt,
                          HfRunner, VllmRunner)
from ....utils import multi_gpu_test
from .conftest import compare_hf_vllm_logprobs
import torch
from vllm.model_executor.models.t5 import T5Attention, T5Config, AttentionType
from vllm.platforms import current_platform

@pytest.mark.parametrize(
    "model",
    [
        # pytest.param("google/t5-small",
                    #  marks=[pytest.mark.core_model, pytest.mark.cpu_model]),
        pytest.param("google-t5/t5-small"),
    ],
)
@pytest.mark.parametrize(
    "vllm_kwargs",
    [{
        "max_model_len": 512
    }]
    )
@pytest.mark.parametrize("dtype", ["float"])#, "bfloat16"])
@pytest.mark.parametrize("max_tokens", [64])
@pytest.mark.parametrize("num_logprobs", [5])
@pytest.mark.parametrize("decoder_prompt_type", list(DecoderPromptType))
def test_models(hf_runner, vllm_runner, example_encoder_decoder_prompts, model,
                dtype, max_tokens, num_logprobs, decoder_prompt_type, vllm_kwargs) -> None:
    # TODO force backend
    compare_hf_vllm_logprobs(
        hf_runner,
        vllm_runner,
        example_encoder_decoder_prompts[decoder_prompt_type],
        decoder_prompt_type,
        model,
        dtype=dtype,
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
        tensor_parallel_size=1,
        vllm_runner_kwargs=vllm_kwargs
    )


def ref_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    attn_weights = scale * torch.einsum("qhd,khd->hqk", query, key).float()
    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask.float()
    attn_weights = torch.softmax(attn_weights, dim=-1).to(value.dtype)
    out = torch.einsum("hqk,khd->qhd", attn_weights, value)
    return out


@pytest.fixture
def dist_init():
    from vllm.distributed import init_distributed_environment, cleanup_dist_env_and_memory, initialize_model_parallel
    import tempfile
    temp_file = tempfile.mkstemp()[1]
    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method=f"file://{temp_file}",
        local_rank=0,
        backend="nccl",
    )
    initialize_model_parallel(1, 1)
    yield
    cleanup_dist_env_and_memory()

# TODO more cases
@pytest.mark.parametrize("dtype", ["float", "bfloat16"])
def test_t5_bias_attention(dtype, dist_init) -> None:
    import random
    seed = 0
    MAX_SEQ_LEN = 32
    block_size = 16
    NUM_BLOCKS = 4321
    current_platform.seed_everything(seed)
    config = T5Config()

    # setup kv caches
    head_size = config.d_kv
    num_heads = (config.num_heads, config.num_heads)
    num_seqs = 1

    scale = float(1.0 / (head_size**0.5))
    num_query_heads, num_kv_heads = num_heads
    # query = torch.empty(num_seqs, num_query_heads, head_size, dtype=dtype)
    # query.uniform_(-scale, scale)

    assert num_query_heads % num_kv_heads == 0
    num_queries_per_kv = num_query_heads // num_kv_heads

    seq_lens = [random.randint(1, MAX_SEQ_LEN) for _ in range(num_seqs)]
    seq_lens[-1] = MAX_SEQ_LEN
    max_seq_len = max(seq_lens)
    # seq_lens = torch.tensor(seq_lens, dtype=torch.int)

    # Create the block tables.
    max_num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    block_tables_lst: List[List[int]] = []
    for _ in range(num_seqs):
        block_table = [
            random.randint(0, NUM_BLOCKS - 1)
            for _ in range(max_num_blocks_per_seq)
        ]
        block_tables_lst.append(block_table)

    block_tables = torch.tensor(block_tables_lst, dtype=torch.int)

    # Create the KV caches.
    kv_cache_dtype = 'auto'
    from vllm.utils import create_kv_caches_with_random
    key_caches, value_caches = create_kv_caches_with_random(NUM_BLOCKS, block_size, 1,
                                                num_kv_heads, head_size,
                                                kv_cache_dtype, dtype, seed,
                                                'cuda')
    key_cache, value_cache = key_caches[0], value_caches[0]

    # Using default kv_scale
    k_scale = v_scale = 1.0


    from vllm.attention.selector import _Backend
    x = torch.randn(num_seqs, max_seq_len, config.d_model, device='cuda', dtype=torch.float)
    with global_force_attn_backend_context_manager(_Backend.XFORMERS):

        from vllm.attention.backends.xformers import XFormersMetadata
        from vllm.attention.backends.xformers import XFormersBackend
        from vllm import LLM

        from vllm.forward_context import set_forward_context
        from vllm.config import VllmConfig

        vllm_config = VllmConfig()
        with set_current_vllm_config(vllm_config):    
            encoder_seq_start_loc = torch.zeros(len(seq_lens) + 1,
                                    dtype=torch.int32,
                                    device='cuda')
            meta = XFormersBackend.make_metadata(
                                    seq_lens=None,#seq_lens, 
                                    max_decode_seq_len=0, num_prefills=None, 
                                    num_prefill_tokens=None, num_decode_tokens=0, 
                                    seq_lens_tensor=None,#torch.tensor(seq_lens),
                                    slot_mapping=None,#torch.zeros(1),
                                    multi_modal_placeholder_index_maps=None, 
                                    max_prefill_seq_len=None,#MAX_SEQ_LEN, 
                                    use_cuda_graph=False,
                                    context_lens_tensor=None,
                                    # no block tables on encoder forward
                                    block_tables=torch.tensor([]).cuda(),
                                    # block_tables=block_tables,
                                    num_encoder_tokens=sum(seq_lens), encoder_seq_lens=seq_lens,encoder_seq_lens_tensor=torch.tensor(seq_lens).cuda(),
                                    max_encoder_seq_len=max(seq_lens), encoder_seq_start_loc=encoder_seq_start_loc)
        # # NOTE use compute_bias here
        # attn_bias = t5_attn.compute_bias(MAX_SEQ_LEN, MAX_SEQ_LEN)

        # same weights should be loaded
        # TODO load model without engine overhead
        llm = LLM(model="google-t5/t5-small", load_format='safetensors', enforce_eager=True, dtype='float')
        model = llm.llm_engine.model_executor.driver_worker.model_runner.model
        t5_attn = model.model.encoder.blocks[0].self_attn.SelfAttention
        print("\nTYPE", type(t5_attn))
        # TODO decoder
        # FIXME this is kinda close, maybe issue is not with xformers custom bias attn
        # t5_attn = T5Attention(config, AttentionType.ENCODER, has_relative_attention_bias=True).cuda()
        # t5_attn.has_relative_attention_bias = False
        assert t5_attn.has_relative_attention_bias
        from transformers import T5Tokenizer, T5ForConditionalGeneration
        from transformers.models.t5.modeling_t5 import T5Attention as HFT5Attention
        hfmodel = T5ForConditionalGeneration.from_pretrained('google-t5/t5-small', return_dict=True)
        print("My T5", t5_attn)
        # this must be set to call attn.impl.forward
        # vllm_config.compilation_config.static_forward_context[".attn"] = t5_attn.attn
        vllm_config.compilation_config.static_forward_context["model.encoder.blocks.0.self_attn.SelfAttention.attn"] = t5_attn.attn
        hf_attn = hfmodel.encoder.block[0].layer[0].SelfAttention.cuda()
        # hf_attn.has_relative_attention_bias = False
        assert hf_attn.has_relative_attention_bias
        # hf_attn = HFT5Attention(config, has_relative_attention_bias=True).cuda()


        with set_forward_context(meta, vllm_config):
            # kv_cache for xformers [2, num_blocks, block_size * num_kv_heads * head_size]
            kvc = torch.stack([key_cache.reshape(NUM_BLOCKS, -1), value_cache.reshape(NUM_BLOCKS, -1)], 0)
            output = t5_attn(x, kvc, meta)
        ref_output, *_ = hf_attn(x)

        atol, rtol = 1e-3, 1e-5
        # torch.testing.assert_close(output, ref_output.squeeze(), atol=atol, rtol=rtol)

        # **cross attn**
        t5_attn = model.model.decoder.blocks[0].cross_attn.EncDecAttention
        print("\nTYPE", type(t5_attn))
        assert not t5_attn.has_relative_attention_bias
        vllm_config.compilation_config.static_forward_context["model.decoder.blocks.0.cross_attn.EncDecAttention.attn"] = t5_attn.attn
        hf_attn = hfmodel.decoder.block[0].layer[1].EncDecAttention.cuda()
        assert not hf_attn.has_relative_attention_bias

        meta = XFormersBackend.make_metadata(
                        seq_lens=seq_lens, 
                        max_decode_seq_len=MAX_SEQ_LEN, num_prefills=0, 
                        num_prefill_tokens=0, num_decode_tokens=1, 
                        max_prefill_seq_len=None,
                        seq_lens_tensor=torch.tensor(seq_lens),
                        slot_mapping=None,#torch.zeros(1),
                        multi_modal_placeholder_index_maps=None, 
                        use_cuda_graph=False,
                        context_lens_tensor=None,
                        block_tables=torch.tensor([]).cuda(),
                        # block_tables=block_tables
                        )


        with set_forward_context(meta, vllm_config):
            output = t5_attn(x, kvc, meta)
        ref_output, *_ = hf_attn(x)

        torch.testing.assert_close(output, ref_output.squeeze(), atol=atol, rtol=rtol)



@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("distributed_executor_backend", ["ray", "mp"])
@pytest.mark.parametrize("model", ["google/t5-small"])
@pytest.mark.parametrize("dtype", ["float"])
@pytest.mark.parametrize("max_tokens", [64])
@pytest.mark.parametrize("num_logprobs", [5])
@pytest.mark.parametrize("decoder_prompt_type", [DecoderPromptType.CUSTOM])
def test_models_distributed(hf_runner, vllm_runner,
                            example_encoder_decoder_prompts,
                            distributed_executor_backend, model, dtype,
                            max_tokens, num_logprobs,
                            decoder_prompt_type) -> None:
    compare_hf_vllm_logprobs(
        hf_runner,
        vllm_runner,
        example_encoder_decoder_prompts[decoder_prompt_type],
        decoder_prompt_type,
        model,
        dtype=dtype,
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
        tensor_parallel_size=2,
        distributed_executor_backend=distributed_executor_backend,
    )
