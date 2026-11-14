# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Extend token sharding without changing TP/EP weights or model outputs."""

import json
import logging
from functools import partial
from pathlib import Path

import pytest
import regex as re
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, DeepseekV32Config

from tests.models.utils import check_logprobs_close
from vllm import SamplingParams
from vllm.lora.request import LoRARequest
from vllm.outputs import CompletionOutput


@pytest.fixture(scope="module")
def shardq_checkpoint(tmp_path_factory):
    path = tmp_path_factory.mktemp("deepseek_v32_shardq")
    config = DeepseekV32Config(
        vocab_size=1024,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=16,
        num_key_value_heads=16,
        kv_lora_rank=512,
        q_lora_rank=128,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        moe_intermediate_size=256,
        n_group=1,
        topk_group=1,
        index_n_heads=32,
        index_head_dim=128,
        index_topk=512,
        max_position_embeddings=8192,
        first_k_dense_replace=1,
        scoring_func="sigmoid",
        topk_method="noaux_tc",
    )
    config.architectures = ["DeepseekV32ForCausalLM"]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_config(config).to(torch.bfloat16)
    weights = {}
    # Transformers stores fused experts; write the checkpoint's per-expert
    # format so the real TP/EP weight loaders are exercised on both runs.
    for name, tensor in model.state_dict().items():
        if ".experts.gate_up_proj" in name:
            prefix = name.removesuffix("gate_up_proj")
            for expert, fused in enumerate(tensor):
                gate, up = fused.chunk(2)
                weights[f"{prefix}{expert}.gate_proj.weight"] = gate.contiguous()
                weights[f"{prefix}{expert}.up_proj.weight"] = up.contiguous()
        elif ".experts.down_proj" in name:
            prefix = name.removesuffix("down_proj")
            for expert, down in enumerate(tensor):
                weights[f"{prefix}{expert}.down_proj.weight"] = down.contiguous()
        else:
            weights[name] = tensor.contiguous()
    config.save_pretrained(path)
    save_file(weights, path / "model.safetensors", metadata={"format": "pt"})
    return str(path)


@pytest.fixture(scope="module")
def shardq_mtp_checkpoint(shardq_checkpoint, tmp_path_factory):
    path = tmp_path_factory.mktemp("deepseek_v32_shardq_mtp")
    config = DeepseekV32Config.from_pretrained(shardq_checkpoint)
    config.num_nextn_predict_layers = 1
    weights = load_file(Path(shardq_checkpoint) / "model.safetensors")
    for name, tensor in list(weights.items()):
        if name.startswith("model.layers.1."):
            weights[name.replace("model.layers.1.", "model.layers.2.")] = tensor.clone()
    width = config.hidden_size
    for norm in ("enorm", "hnorm", "shared_head.norm"):
        weights[f"model.layers.2.{norm}.weight"] = torch.ones(
            width, dtype=torch.bfloat16
        )
    weights["model.layers.2.shared_head.head.weight"] = weights[
        "lm_head.weight"
    ].clone()
    # Preserve the target state in the draft input to obtain accepted tokens
    # from this small random checkpoint without a pretrained MTP model.
    weights["model.layers.2.eh_proj.weight"] = torch.cat(
        (
            torch.zeros(width, width, dtype=torch.bfloat16),
            torch.eye(width, dtype=torch.bfloat16),
        ),
        dim=1,
    )
    config.save_pretrained(path)
    save_file(weights, path / "model.safetensors", metadata={"format": "pt"})
    return str(path)


@pytest.fixture(scope="module")
def shardq_lora(tmp_path_factory):
    path = tmp_path_factory.mktemp("deepseek_v32_shardq_lora")
    weights = {}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(73)
        for layer in range(2):
            name = f"base_model.model.model.layers.{layer}.self_attn.q_a_proj"
            weights[f"{name}.lora_A.weight"] = torch.randn(8, 256).bfloat16() * 0.05
            weights[f"{name}.lora_B.weight"] = torch.randn(128, 8).bfloat16() * 0.05
    save_file(weights, path / "adapter_model.safetensors")
    (path / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "r": 8,
                "lora_alpha": 8,
                "target_modules": ["q_a_proj"],
                "bias": "none",
            }
        )
    )
    return LoRARequest("shardq", 1, str(path))


def _layout(worker):
    model = worker.model_runner.model.model
    layer = model.layers[0]
    return (
        model.embed_tokens.tp_size,
        layer.self_attn.num_local_heads,
        layer.self_attn.q_b_proj.tp_size,
        layer.mlp.gate_up_proj.tp_size,
        model.layers[1].mlp.is_sequence_parallel,
        model.layers[1].mlp.shared_experts.gate_up_proj.tp_size,
        {name: (tuple(p.shape), p.dtype) for name, p in model.named_parameters()},
    )


def _check_local_execution(worker, *, sequence_parallel=True, min_tokens=64):
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.layers.attention.sparse_mla_attention import (
        SparseMLACommonMetadata,
    )

    model = worker.model_runner.model.model
    worker.shardq_observed_rows = []

    def check_rows(module, args, *, projection):
        metadata = get_forward_context().attn_metadata
        assert isinstance(metadata, dict)
        layer_metadata = metadata[model.layers[0].self_attn.layer_name]
        assert isinstance(layer_metadata, SparseMLACommonMetadata)
        full_rows = layer_metadata.num_actual_tokens
        local_rows = (
            (full_rows + 1) // 2
            if sequence_parallel and full_rows >= min_tokens
            else full_rows
        )
        expected = local_rows if projection == "A" else full_rows
        assert args[0].shape[0] == expected, (projection, full_rows, expected)
        worker.shardq_observed_rows.append((projection, full_rows, args[0].shape[0]))

    for layer in model.layers:
        layer.self_attn.fused_qkv_a_proj.register_forward_pre_hook(
            partial(check_rows, projection="A")
        )
        layer.self_attn.indexer.wk_weights_proj.register_forward_pre_hook(
            partial(check_rows, projection="A")
        )
        layer.self_attn.q_b_proj.register_forward_pre_hook(
            partial(check_rows, projection="B")
        )
        layer.mlp.register_forward_pre_hook(partial(check_rows, projection="MLP"))


def _observed_rows(worker):
    return worker.shardq_observed_rows


@pytest.mark.parametrize(
    ("quantization", "use_lora", "spec_method", "enforce_eager"),
    [
        (None, False, None, True),
        ("fp8", False, None, True),
        (None, True, None, True),
        (None, False, "ngram", True),
        (None, False, "mtp", True),
        (None, False, "mtp", False),
    ],
    ids=["bf16", "fp8", "lora", "ngram", "mtp", "mtp-cudagraph"],
)
def test_shardq_keeps_tp_weights_and_projects_local_rows(
    vllm_runner,
    shardq_checkpoint,
    shardq_mtp_checkpoint,
    shardq_lora,
    monkeypatch,
    caplog_vllm,
    quantization,
    use_lora,
    spec_method,
    enforce_eager,
):
    """Preserve TP generation with LoRA or accepted speculative tokens."""
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    caplog_vllm.set_level(logging.INFO, logger="vllm.v1.metrics.loggers")
    speculative_config = None
    if spec_method:
        speculative_config = {"method": spec_method, "num_speculative_tokens": 3}
        if spec_method == "mtp":
            speculative_config["index_share_for_mtp_iteration"] = True
        else:
            speculative_config.update(prompt_lookup_min=1, prompt_lookup_max=3)
    min_tokens = 4 if spec_method else 64
    max_tokens = 16 if spec_method else 4
    prompts = [
        {"prompt_token_ids": [2 + i % 997 for i in range(n)]} for n in (769, 17, 3)
    ]
    sampling = SamplingParams(
        temperature=0, max_tokens=max_tokens, ignore_eos=True, logprobs=16
    )
    outputs: list[list[CompletionOutput]] = []
    layouts = []
    for use_shardq in (False, True):
        with vllm_runner(
            shardq_mtp_checkpoint if spec_method == "mtp" else shardq_checkpoint,
            tensor_parallel_size=2,
            enable_expert_parallel=True,
            all2all_backend="allgather_reducescatter",
            skip_tokenizer_init=True,
            dtype="bfloat16",
            quantization=quantization,
            enable_lora=use_lora,
            max_loras=2,
            max_lora_rank=8,
            enforce_eager=enforce_eager,
            speculative_config=speculative_config,
            disable_log_stats=False,
            cudagraph_metrics=not enforce_eager,
            compilation_config={
                "cudagraph_mode": "FULL_AND_PIECEWISE",
                "prefill_cudagraph_capture_token_sizes": [64, 128, 255],
                "decode_cudagraph_capture_request_sizes": [1, 2, 4],
            }
            if not enforce_eager
            else None,
            max_model_len=1024,
            max_num_batched_tokens=255,
            max_num_seqs=4,
            enable_chunked_prefill=True,
            enable_prefix_caching=True,
            block_size=64,
            kv_cache_dtype="fp8_ds_mla",
            attention_config={
                "backend": "FLASHMLA_SPARSE",
                "sparse_mla_shard_q": use_shardq,
                "sparse_mla_shard_q_min_tokens": min_tokens,
            },
            gpu_memory_utilization=0.1,
        ) as runner:
            caplog_vllm.clear()
            layouts.append(runner.llm.collective_rpc(_layout))
            assert all(layout[:6] == (2, 8, 2, 2, False, 2) for layout in layouts[-1])
            enabled = runner.llm.collective_rpc(
                lambda worker: worker.model_runner.model.model.shardq_sequence_parallel
            )
            assert enabled == [use_shardq and not use_lora] * 2
            if use_shardq and enforce_eager:
                runner.llm.collective_rpc(
                    partial(
                        _check_local_execution,
                        sequence_parallel=not use_lora,
                        min_tokens=min_tokens,
                    )
                )
            runs: list[CompletionOutput] = []
            for batch in (prompts, prompts[:1]):
                result = runner.llm.generate(
                    batch,
                    sampling,
                    lora_request=([shardq_lora, None, shardq_lora][: len(batch)])
                    if use_lora
                    else None,
                    use_tqdm=False,
                )
                runs.extend(r.outputs[0] for r in result)
            if spec_method:
                metrics = {m.name: m for m in runner.llm.get_metrics()}
                assert metrics["vllm:spec_decode_num_accepted_tokens"].value > 0
            if not enforce_eager:
                runner.llm.llm_engine.do_log_stats()
                for mode in ("FULL", "PIECEWISE"):
                    assert re.search(rf"\|\s+{mode}\s+\|", caplog_vllm.text)
            if use_shardq and enforce_eager:
                for rows in runner.llm.collective_rpc(_observed_rows):
                    if spec_method:
                        # Four rows are one target token plus three draft tokens.
                        assert ("A", 4, 2) in rows
                    else:
                        assert ("A", 1, 1) in rows
                    if use_lora:
                        assert any(p == "A" and full >= 64 for p, full, _ in rows)
                    else:
                        assert any(
                            p == "A" and full > local > 1 for p, full, local in rows
                        )
            outputs.append(runs)
    assert layouts[0] == layouts[1]
    assert all(len(run.token_ids) == max_tokens for runs in outputs for run in runs)
    check_logprobs_close(
        outputs_0_lst=[(list(r.token_ids), "", r.logprobs) for r in outputs[0]],
        outputs_1_lst=[(list(r.token_ids), "", r.logprobs) for r in outputs[1]],
        name_0="TP",
        name_1="TP with extended ShardQ",
    )
