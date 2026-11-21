# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare DSA SP against TP on the same DeepSeek V4 hash-MoE/CSA checkpoint."""

import pytest
import torch
from safetensors.torch import save_file
from transformers import DeepseekV4Config

from tests.models.utils import check_logprobs_close
from vllm import SamplingParams


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory, request):
    """Write one native-format layer with the kernel's real head dimensions."""
    path = tmp_path_factory.mktemp("dsv4_csa")
    hidden, heads, head_dim, index_heads, index_dim = 512, 16, 512, 64, 128
    vocab, intermediate, experts, topk = 512, 256, 8, 2
    q_rank, o_groups, o_rank, hc = 1024, 2, 128, 4
    config = DeepseekV4Config(
        architectures=["DeepseekV4ForCausalLM"],
        num_hidden_layers=1,
        layer_types=["compressed_sparse_attention"],
        mlp_layer_types=["hash_moe"],
        num_nextn_predict_layers=0,
        hidden_size=hidden,
        num_attention_heads=heads,
        num_key_value_heads=1,
        head_dim=head_dim,
        index_n_heads=index_heads,
        index_head_dim=index_dim,
        index_topk=512,
        q_lora_rank=q_rank,
        o_groups=o_groups,
        o_lora_rank=o_rank,
        hc_mult=hc,
        qk_rope_head_dim=64,
        sliding_window=128,
        tie_word_embeddings=False,
        n_routed_experts=experts,
        n_shared_experts=1,
        num_experts_per_tok=topk,
        moe_intermediate_size=intermediate,
        vocab_size=vocab,
        max_position_embeddings=4096,
        bos_token_id=0,
        eos_token_id=1,
    )
    fp8 = request.param == "fp8"
    if fp8:
        config.expert_dtype = "fp8"
        config.quantization_config = {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "expert_dtype": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
        }
    config.save_pretrained(path)
    rng = torch.Generator().manual_seed(17)

    def weight(*shape):
        return (torch.randn(shape, generator=rng) / shape[-1] ** 0.5).bfloat16()

    state = {
        "embed.weight": weight(vocab, hidden),
        "head.weight": weight(vocab, hidden),
        "norm.weight": torch.ones(hidden),
        "hc_head_fn": weight(hc, hc * hidden),
        "hc_head_base": weight(hc),
        "hc_head_scale": torch.ones(1),
    }
    layer = {
        "attn.wq_a.weight": weight(q_rank, hidden),
        "attn.wkv.weight": weight(head_dim, hidden),
        "attn.q_norm.weight": torch.ones(q_rank),
        "attn.kv_norm.weight": torch.ones(head_dim),
        "attn.wq_b.weight": weight(heads * head_dim, q_rank),
        "attn.wo_a.weight": weight(o_groups * o_rank, heads * head_dim // o_groups),
        "attn.wo_b.weight": weight(hidden, o_groups * o_rank),
        "attn.attn_sink": weight(heads),
        "attn.indexer.wq_b.weight": weight(index_heads * index_dim, q_rank),
        "attn.indexer.weights_proj.weight": weight(index_heads, hidden),
        "ffn.gate.weight": weight(experts, hidden),
        "ffn.gate.tid2eid": torch.rand(vocab, experts, generator=rng)
        .argsort(1)[:, :topk]
        .int()
        .contiguous(),
    }
    for branch in ("attn", "ffn"):
        layer[f"{branch}_norm.weight"] = torch.ones(hidden)
        layer[f"hc_{branch}_fn"] = weight((2 + hc) * hc, hc * hidden)
        layer[f"hc_{branch}_scale"] = torch.ones(3)
        layer[f"hc_{branch}_base"] = weight((2 + hc) * hc)
    for prefix, dim in (("attn", head_dim), ("attn.indexer", index_dim)):
        layer[f"{prefix}.compressor.norm.weight"] = torch.ones(dim)
        layer[f"{prefix}.compressor.ape"] = weight(4, 2 * dim)
        for name in ("wkv", "wgate"):
            layer[f"{prefix}.compressor.{name}.weight"] = weight(2 * dim, hidden)
    for name in ["shared_experts", *(f"experts.{i}" for i in range(experts))]:
        layer[f"ffn.{name}.w1.weight"] = weight(intermediate, hidden)
        layer[f"ffn.{name}.w2.weight"] = weight(hidden, intermediate)
        layer[f"ffn.{name}.w3.weight"] = weight(intermediate, hidden)
    if fp8:
        # Native FP8 V4 stores an E4M3 matrix and one FP32 scale per 128x128
        # block. The compressor, index weights projection and gate stay BF16.
        for name in list(layer):
            if not name.endswith(".weight"):
                continue
            prefix = name.removesuffix(".weight")
            if prefix.startswith(("ffn.experts.", "ffn.shared_experts.")) or prefix in (
                "attn.wq_a",
                "attn.wkv",
                "attn.wq_b",
                "attn.wo_a",
                "attn.wo_b",
                "attn.indexer.wq_b",
            ):
                matrix = layer[name]
                m, n = matrix.shape
                blocks = matrix.float().view(m // 128, 128, n // 128, 128)
                scales = (blocks.abs().amax(dim=(1, 3)) / 448).clamp_min(1e-12)
                layer[name] = (
                    (blocks / scales[:, None, :, None])
                    .to(torch.float8_e4m3fn)
                    .reshape(m, n)
                )
                layer[f"{prefix}.scale"] = scales
    state.update({f"layers.0.{k}": v for k, v in layer.items()})
    save_file(state, path / "model.safetensors", metadata={"format": "pt"})
    return str(path)


@pytest.mark.parametrize(
    ("layout", "use_ep", "checkpoint"),
    [("contiguous", True, "bf16"), ("zigzag", False, "fp8")],
    indirect=["checkpoint"],
)
def test_dsa_sp_keeps_v4_hash_routing_and_query_order(
    vllm_runner, checkpoint, layout, use_ep
):
    # The short request enters decode while the long one continues prefill.
    prompts = [
        {"prompt_token_ids": [2 + (i + length) % 500 for i in range(length)]}
        for length in (9, 527, 133)
    ]
    sampling = SamplingParams(temperature=0, max_tokens=4, ignore_eos=True, logprobs=8)
    runs = []
    for mode in (None, layout):
        with vllm_runner(
            checkpoint,
            tensor_parallel_size=2,
            enable_expert_parallel=use_ep,
            all2all_backend="allgather_reducescatter",
            attention_config={"dsa_sequence_parallel": mode},
            kv_cache_dtype="fp8",
            dtype="bfloat16",
            block_size=256,
            max_model_len=1024,
            max_num_batched_tokens=259,
            max_num_seqs=4,
            enable_chunked_prefill=True,
            skip_tokenizer_init=True,
            enforce_eager=True,
            gpu_memory_utilization=0.2,
        ) as runner:
            results = runner.llm.generate(prompts, sampling, use_tqdm=False)
            runs.append(
                [
                    (list(r.outputs[0].token_ids), "", r.outputs[0].logprobs)
                    for r in results
                ]
            )
    check_logprobs_close(
        outputs_0_lst=runs[0],
        outputs_1_lst=runs[1],
        name_0="TP",
        name_1=f"DSA SP ({layout})",
        always_check_logprobs=True,
        near_tie_tol=0.05,
    )
