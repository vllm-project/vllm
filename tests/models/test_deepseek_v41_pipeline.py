# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform


def _write_bf16_engram_checkpoint(path: Path) -> torch.Tensor:
    from safetensors.torch import save_file
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import TokenizersBackend

    from tests.utils import init_test_distributed_environment
    from vllm.config import EngramConfig, KernelConfig, ModelConfig, VllmConfig
    from vllm.distributed import cleanup_dist_env_and_memory
    from vllm.model_executor.model_loader.utils import initialize_model
    from vllm.transformers_utils.configs.deepseek_v41 import DeepseekV41Config
    from vllm.utils.network_utils import get_open_port
    from vllm.utils.torch_utils import set_default_torch_dtype

    config = DeepseekV41Config(
        text_config={
            "vocab_size": 128,
            "hidden_size": 128,
            "moe_intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "num_key_value_heads": 1,
            "head_dim": 512,
            "qk_rope_head_dim": 64,
            "q_lora_rank": 128,
            "o_lora_rank": 128,
            "o_groups": 1,
            "hidden_act": "silu",
            "swiglu_limit": 10.0,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 512,
            "rope_theta": 10000.0,
            "compress_rope_theta": 10000.0,
            "n_routed_experts": 2,
            "n_shared_experts": 1,
            "num_experts_per_tok": 1,
            "scoring_func": "sqrtsoftplus",
            "topk_method": "noaux_tc",
            "norm_topk_prob": True,
            "routed_scaling_factor": 1.0,
            "sliding_window": 128,
            "compress_ratios": [0, 1],
            "index_topk": 16,
            "index_n_heads": 32,
            "index_head_dim": 128,
            "kv_source_layer_ids": [1],
            "index_source_layer_ids": [1],
            "hc_mult": 4,
            "hc_sinkhorn_iters": 20,
            "hc_eps": 1e-6,
            "engram_layer_ids": [1],
            "engram_num_embeddings": [36],
            "engram_max_ngram_size": 2,
            "engram_vocab_size": 16,
            "engram_n_heads": 2,
            "engram_head_dim": 256,
            "engram_pad_token_id": 2,
            "engram_compressed_vocab_size": 128,
            "num_nextn_predict_layers": 0,
        },
        architectures=["DeepseekV41ForCausalLM"],
        dtype="bfloat16",
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=2,
        image_token_id=127,
        tie_word_embeddings=False,
    )
    config.save_pretrained(path)
    vocab = {f"token{i}": i for i in range(128)}
    tokenizer = TokenizersBackend(
        tokenizer_object=Tokenizer(WordLevel(vocab=vocab, unk_token="token0")),
        unk_token="token0",
        bos_token="token0",
        eos_token="token1",
        pad_token="token2",
    )
    tokenizer.save_pretrained(path)
    vllm_config = VllmConfig(
        model_config=ModelConfig(
            model=str(path),
            dtype="bfloat16",
            language_model_only=True,
            skip_tokenizer_init=True,
            max_model_len=256,
        ),
        engram_config=EngramConfig(cpu_offload=False),
        kernel_config=KernelConfig(enable_jit_warmup=False),
    )
    init_test_distributed_environment(1, 1, 0, str(get_open_port()))
    try:
        with torch.device("cuda"), set_default_torch_dtype(torch.bfloat16):
            model = initialize_model(vllm_config)
        generator = torch.Generator().manual_seed(0)
        weights = {}
        for name, param in model.named_parameters():
            if name.endswith("engram.embed_tokens.weight_scale_inv") or (
                param.is_meta
                and name not in ("image_start", "image_end", "image_newline")
            ):
                continue
            dtype = (
                torch.bfloat16 if "engram.embed_tokens.weight" in name else param.dtype
            )
            value = torch.randn(param.shape, generator=generator) * 0.02
            if "norm" in name or name.endswith(("q_weight", "k_weight")):
                value += 1
            value = value.to(dtype)
            name = name.removeprefix("language_model.").removeprefix("model.")
            name = name.replace(".experts.routed_experts.", ".experts.")
            if "compressor.fused_wkv_wgate" in name:
                weights[name.replace("fused_wkv_wgate", "wkv")] = value
            elif "fused_wqa_wkv" in name:
                for part, suffix in zip(value.split([128, 512]), ["wq_a", "wkv"]):
                    weights[name.replace("fused_wqa_wkv", suffix)] = part.contiguous()
            elif "experts.w13_weight" in name:
                for expert, tensor in enumerate(value):
                    for part, suffix in zip(tensor.chunk(2), ["w1", "w3"]):
                        weights[
                            name.replace(
                                "experts.w13_weight",
                                f"experts.{expert}.{suffix}.weight",
                            )
                        ] = part.contiguous()
            elif "experts.w2_weight" in name:
                for expert, tensor in enumerate(value):
                    weights[
                        name.replace("experts.w2_weight", f"experts.{expert}.w2.weight")
                    ] = tensor.contiguous()
            elif "gate_up_proj" in name:
                for part, suffix in zip(value.chunk(2), ["w1", "w3"]):
                    weights[name.replace("gate_up_proj", suffix)] = part.contiguous()
            else:
                name = name.replace("embed_tokens.weight", "embed.weight")
                name = name.replace("lm_head.weight", "head.weight")
                name = name.replace("e_score_correction_bias", "bias")
                weights[name] = value
        save_file(weights, str(path / "model.safetensors"))
        return weights["layers.1.engram.embed.weight"]
    finally:
        cleanup_dist_env_and_memory()


def test_bf16_engram_checkpoint_loads_and_generates(
    tmp_path, vllm_runner, enable_pickle
):
    """Unquantized tables load without scales and agree with CPU offload."""
    from tests.models.utils import check_logprobs_close
    from vllm import SamplingParams

    expected = _write_bf16_engram_checkpoint(tmp_path)

    def table_matches(model):
        table = model.language_model.model.layers[1].engram.embed_tokens.weight
        return table.dtype == torch.bfloat16 and torch.equal(table.cpu(), expected)

    outputs = []
    for offload in (False, True):
        with vllm_runner(
            str(tmp_path),
            dtype="bfloat16",
            language_model_only=True,
            skip_tokenizer_init=True,
            max_model_len=256,
            block_size=64,
            max_num_seqs=1,
            max_num_batched_tokens=256,
            enable_chunked_prefill=True,
            enforce_eager=True,
            kv_cache_memory_bytes=16 * 1024**2,
            gpu_memory_utilization=0.25,
            engram_config={"cpu_offload": offload},
            kernel_config={"enable_jit_warmup": False},
        ) as runner:
            assert runner.llm.apply_model(table_matches)[0]
            result = runner.llm.generate(
                {"prompt_token_ids": [5, 6, 7, 8]},
                SamplingParams(
                    temperature=0, max_tokens=4, ignore_eos=True, logprobs=5
                ),
            )[0].outputs[0]
            assert len(result.token_ids) == 4
            outputs.append((list(result.token_ids), result.text, result.logprobs))
    check_logprobs_close(
        outputs_0_lst=[outputs[0]],
        outputs_1_lst=[outputs[1]],
        name_0="GPU Engram",
        name_1="CPU Engram",
        always_check_logprobs=True,
    )
    assert outputs[0][0] == outputs[1][0]
    for gpu_logprobs, cpu_logprobs in zip(outputs[0][2], outputs[1][2]):
        assert gpu_logprobs.keys() == cpu_logprobs.keys()
        torch.testing.assert_close(
            torch.tensor([gpu_logprobs[token].logprob for token in gpu_logprobs]),
            torch.tensor([cpu_logprobs[token].logprob for token in gpu_logprobs]),
        )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
@pytest.mark.skip_global_cleanup
def test_stage_entry_injects_engram_before_mhc_pre(monkeypatch):
    model_module = import_module("vllm.models.deepseek_v41.nvidia.model")

    class StopAfterMhcPre(Exception):
        pass

    hidden = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    hashes = torch.arange(6).reshape(2, 3)
    mask = torch.tensor([True, False])
    pre_mix = torch.randn(2, 4)
    engram_calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]] = []
    mhc_pre_calls: list[tuple[torch.Tensor, torch.Tensor | None]] = []

    class Engram:
        layer_hash_index = 1

        def __call__(self, residual, layer_hashes, engram_mask):
            engram_calls.append((residual, layer_hashes, engram_mask))
            return residual + 10

    def mhc_pre(residual, *args, pre_mix=None, **kwargs):
        mhc_pre_calls.append((residual, pre_mix))
        raise StopAfterMhcPre

    layer = SimpleNamespace(
        mhc_stream=None,
        engram=Engram(),
        hc_attn_fn=None,
        hc_attn_scale=None,
        hc_attn_base=None,
        rms_norm_eps=1e-6,
        hc_eps=1e-6,
        hc_post_alpha=1.0,
        hc_sinkhorn_iters=1,
        attn_norm=SimpleNamespace(weight=None, variance_epsilon=1e-6),
    )
    monkeypatch.setattr(model_module, "mhc_pre_delayed_tilelang", mhc_pre)

    with pytest.raises(StopAfterMhcPre):
        model_module.DeepseekV4DecoderLayer.forward(
            layer,
            hidden,
            torch.arange(2),
            None,
            pre_mix=pre_mix,
            engram_hashes=hashes,
            engram_mask=mask,
        )

    assert len(engram_calls) == len(mhc_pre_calls) == 1
    engram_residual, layer_hashes, engram_mask = engram_calls[0]
    assert engram_residual is hidden
    assert engram_mask is mask
    torch.testing.assert_close(layer_hashes, hashes[:, 1])
    mhc_residual, mhc_pre_mix = mhc_pre_calls[0]
    torch.testing.assert_close(mhc_residual, hidden + 10)
    assert mhc_pre_mix is pre_mix
