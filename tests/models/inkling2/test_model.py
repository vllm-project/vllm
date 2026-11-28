# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end smoke test of a tiny random-weight Inkling2 model (TP=1)."""

import copy
import json
from typing import Any

import numpy as np
import pytest
import torch

from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.models.inkling2.configs import (
    Inkling2DSparkConfig,
    Inkling2LoopMMConfig,
    Inkling2MMConfig,
    Inkling2ModelConfig,
)
from vllm.platforms import current_platform

if not (
    current_platform.is_device_capability_family(100)
    or current_platform.is_device_capability(90)
    or current_platform.is_device_capability_family(120)
):
    pytest.skip("Inkling2 requires Hopper or Blackwell", allow_module_level=True)

TINY_CONFIG: dict[str, Any] = {
    "architectures": ["Inkling2ForConditionalGeneration"],
    "model_type": "inkling2",
    "eos_token_id": 1,
    "text_config": {
        "torch_dtype": "bfloat16",
        "hidden_size": 512,
        "num_hidden_layers": 2,
        "local_layer_ids": [0],
        "sliding_window_size": 128,
        "vocab_size": 1024,
        "unpadded_vocab_size": 1000,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "n_routed_experts": 16,
        "num_experts_per_tok": 8,
        "intermediate_size": 256,
        "mlp_gate_softcap": 4.0,
        "mlp_up_softcap": 25.0,
    },
    "mtp_config": {
        "hidden_size": 512,
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "intermediate_size": 512,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1e6,
        "max_position_embeddings": 4096,
        "local_layer_ids": [0],
        "sliding_window_size": 128,
        "block_size": 8,
        "markov_rank": 32,
        "markov_head_type": "rnn_hidden",
        "target_layer_ids": [1, 2],
    },
}


# A looped variant: layer 0 is the prelude, layers 1-2 (one local, one full)
# run three passes, layer 3 is the coda; one-layer AttnRes blocks make every
# layer a block, so later passes read a two-block message bank.
TINY_LOOP_CONFIG: dict[str, Any] = {
    **TINY_CONFIG,
    "architectures": ["Inkling2LoopForConditionalGeneration"],
    "model_type": "inkling2_loop",
    "text_config": {
        **TINY_CONFIG["text_config"],
        # The loop runs the fused conv + mix kernels (hidden size 2304 or 4608).
        "hidden_size": 2304,
        "num_hidden_layers": 4,
        "local_layer_ids": [0, 1, 3],
        "n_layers_per_attn_res_block": 1,
        "loop_config": {
            "core_start": 1,
            "core_end": 3,
            "num_passes": 3,
            "gate_rank": 16,
            "is_varloop": True,
        },
    },
    "mtp_config": {
        **TINY_CONFIG["mtp_config"],
        "hidden_size": 2304,
        "target_layer_ids": [3, 8],
    },
}


def _tiny_looped_llm(tmp_path, text_overrides: dict[str, Any]) -> LLM:
    (tmp_path / "config.json").write_text(json.dumps(TINY_LOOP_CONFIG))
    return LLM(
        model=str(tmp_path),
        load_format="dummy",
        skip_tokenizer_init=True,
        tokenizer_mode="hf",
        gpu_memory_utilization=0.035,
        # These cache comparisons use at most 216 tokens in a single prefill.
        max_num_batched_tokens=256,
        max_model_len=2048,
        max_num_seqs=4,
        enable_prefix_caching=False,
        limit_mm_per_prompt={"image": 0},
        hf_overrides={"text_config": text_overrides},
    )


def _prompt_logprobs(llm: LLM, prompt: list[int]) -> list[float]:
    out = llm.generate(
        [{"prompt_token_ids": prompt}],
        SamplingParams(
            max_tokens=1, prompt_logprobs=0, ignore_eos=True, detokenize=False
        ),
    )[0]
    assert out.prompt_logprobs is not None
    return [
        entry[token].logprob
        for token, entry in zip(prompt[1:], out.prompt_logprobs[1:])
        if entry is not None
    ]


@pytest.mark.parametrize("text_overrides", [{}, {"loop_num_passes": 1}])
def test_tiny_looped_model_decode_matches_prefill(tmp_path, text_overrides):
    """Each loop pass keeps its own KV and conv-state caches: greedy decode
    steps (reading the caches of earlier tokens) give the tokens and
    logprobs of a single prefill over the same sequence, at the checkpoint's
    or an overridden pass count (variable-loop checkpoints)."""
    llm = _tiny_looped_llm(tmp_path, text_overrides)
    prompt = list(range(2, 202))
    decoded = llm.generate(
        [{"prompt_token_ids": prompt}],
        SamplingParams(
            max_tokens=16, temperature=0, logprobs=0, ignore_eos=True, detokenize=False
        ),
    )[0].outputs[0]
    full = llm.generate(
        [{"prompt_token_ids": prompt + list(decoded.token_ids)}],
        SamplingParams(
            max_tokens=1, prompt_logprobs=0, ignore_eos=True, detokenize=False
        ),
    )[0]
    assert full.prompt_logprobs is not None and decoded.logprobs is not None
    for step, token in enumerate(decoded.token_ids):
        step_logprob = decoded.logprobs[step][token].logprob
        prefill = full.prompt_logprobs[len(prompt) + step]
        assert prefill is not None and token in prefill
        assert abs(prefill[token].logprob - step_logprob) < 2e-2
        assert prefill[token].rank == 1


def test_tiny_looped_model_shared_cache(tmp_path):
    """With one KV cache shared by the loop passes, a single prefill (each
    pass writes its K/V before reading them) gives the per-pass caches'
    logprobs; later tokens read the last pass's K/V."""
    prompt = list(range(2, 202))
    llm = _tiny_looped_llm(tmp_path, {"loop_share_cache": True})
    shared = _prompt_logprobs(llm, prompt)
    del llm
    cleanup_dist_env_and_memory()
    separate = _prompt_logprobs(_tiny_looped_llm(tmp_path, {}), prompt)
    torch.testing.assert_close(shared, separate, atol=1e-3, rtol=0)


def test_loop_serving_options_validation(tmp_path):
    """The pass-count and shared-cache options need a variable-loop
    checkpoint, and DSpark the checkpoint's pass count; looped checkpoints
    declare model_type inkling2_loop and their own architecture."""
    text = {**TINY_LOOP_CONFIG["text_config"]}
    fixed = {**text, "loop_config": {**text["loop_config"], "is_varloop": False}}
    with pytest.raises(ValueError, match="is_varloop"):
        Inkling2ModelConfig(**fixed, loop_share_cache=True)
    with pytest.raises(ValueError, match="at least 2 passes"):
        Inkling2ModelConfig(
            **{**fixed, "loop_config": {**fixed["loop_config"], "num_passes": 1}}
        )
    with pytest.raises(ValueError, match="inkling2_loop"):
        Inkling2MMConfig(text_config=text)
    with pytest.raises(ValueError, match="architectures"):
        Inkling2LoopMMConfig(
            text_config=text, architectures=["Inkling2ForConditionalGeneration"]
        )
    config = Inkling2LoopMMConfig(
        text_config=text, mtp_config=TINY_LOOP_CONFIG["mtp_config"]
    )
    config.save_pretrained(tmp_path)
    restored = Inkling2LoopMMConfig.from_pretrained(tmp_path)
    assert restored.text_config.loop_config == config.text_config.loop_config
    assert restored.text_config.num_layer_visits == 8
    Inkling2DSparkConfig.from_target(config)
    config.text_config.loop_num_passes = 2
    with pytest.raises(ValueError, match="loop passes"):
        Inkling2DSparkConfig.from_target(config)


@pytest.mark.parametrize("looped", [False, True])
def test_portable_fp16_kv_config(tmp_path, looped):
    """Only SM12 Inkling2 accepts FP16 KV with a BF16 model."""
    from vllm.config import CacheConfig, ModelConfig, VllmConfig

    (tmp_path / "config.json").write_text(
        json.dumps(TINY_LOOP_CONFIG if looped else TINY_CONFIG)
    )
    model = ModelConfig(
        model=str(tmp_path),
        dtype="bfloat16",
        skip_tokenizer_init=True,
        tokenizer_mode="hf",
        max_model_len=2048,
    )
    cache = CacheConfig(cache_dtype="float16")
    if current_platform.is_device_capability_family(120):
        config = VllmConfig(model_config=model, cache_config=cache)
        assert config.model_config.dtype == torch.bfloat16
        assert config.cache_config.cache_dtype == "float16"
    else:
        with pytest.raises(ValueError, match="kv_cache_dtype"):
            VllmConfig(model_config=model, cache_config=cache)


@pytest.mark.parametrize("looped", [False, True])
@pytest.mark.parametrize(
    "draft_sample_method,kv_cache_dtype,moe_backend",
    [
        (None, "auto", "auto"),
        ("greedy", "auto", "auto"),
        ("probabilistic", "auto", "auto"),
    ]
    + (
        [
            (None, "float16", "auto"),
            (None, "fp8_e4m3", "auto"),
            ("greedy", "fp8_e4m3", "auto"),
            ("probabilistic", "fp8_e4m3", "auto"),
            (None, "fp8_e4m3", "humming"),
            ("greedy", "fp8_e4m3", "humming"),
        ]
        if current_platform.is_device_capability_family(120)
        else []
    ),
)
def test_tiny_model_generates(
    tmp_path, draft_sample_method, kv_cache_dtype, moe_backend, looped
):
    """One local and one full layer load (dummy weights) and decode with the
    paged KV and conv-state caches and CUDA graphs, optionally speculating
    with the DSpark draft (target feature capture, draft KV and graphs).
    Probabilistic drafting caches the draft's bf16 logits for rejection."""
    config = copy.deepcopy(TINY_LOOP_CONFIG if looped else TINY_CONFIG)
    (tmp_path / "config.json").write_text(json.dumps(config))
    llm = LLM(
        speculative_config=(
            None
            if draft_sample_method is None
            else {
                "method": "dspark",
                "num_speculative_tokens": 7,
                "draft_sample_method": draft_sample_method,
            }
        ),
        model=str(tmp_path),
        load_format="dummy",
        skip_tokenizer_init=True,
        tokenizer_mode="hf",
        kv_cache_dtype=kv_cache_dtype,
        kernel_config={"moe_backend": moe_backend},
        gpu_memory_utilization=0.10 if looped else 0.035,
        # Bound warmup memory and exercise chunking of the 298-token prompt.
        max_num_batched_tokens=256,
        max_model_len=2048,
        max_num_seqs=4,
        limit_mm_per_prompt={"image": 0},
        disable_log_stats=False,
    )
    prompts = [list(range(2, 300)), [5, 5, 5]]
    outputs = llm.generate(
        [{"prompt_token_ids": p} for p in prompts],
        SamplingParams(
            max_tokens=24,
            temperature=1.0 if draft_sample_method == "probabilistic" else 0,
            ignore_eos=True,
            detokenize=False,
        ),
    )
    for output in outputs:
        token_ids = output.outputs[0].token_ids
        assert len(token_ids) == 24
        assert all(0 <= t < 1000 for t in token_ids)
    if looped and draft_sample_method is None:
        single = llm.generate(
            [{"prompt_token_ids": prompts[1]}],
            SamplingParams(
                max_tokens=24, temperature=0, ignore_eos=True, detokenize=False
            ),
        )
        assert len(single[0].outputs[0].token_ids) == 24
        assert all(0 <= t < 1000 for t in single[0].outputs[0].token_ids)
    if draft_sample_method is not None:
        from tests.v1.e2e.spec_decode.utils import get_spec_decode_metric_value

        metrics = llm.get_metrics()
        drafted = get_spec_decode_metric_value(
            metrics, "vllm:spec_decode_num_draft_tokens"
        )
        accepted = get_spec_decode_metric_value(
            metrics, "vllm:spec_decode_num_accepted_tokens"
        )
        assert drafted > 0
        assert 0 <= accepted <= drafted


def _record_moe_routing(model: torch.nn.Module) -> None:
    """Redraw the routers (dummy weights route every token alike) and the
    loop message controllers (dummy ones leave every pass alike), then
    record, in call order, the torch reference routing of every MoE call's
    input (sorted expert ids per token)."""
    from vllm.models.inkling2.nvidia.loop import Inkling2LoopDecoderLayer
    from vllm.models.inkling2.nvidia.moe import Inkling2MoE
    from vllm.models.inkling2.nvidia.ops.linear import linear_k_groups
    from vllm.utils.gpu_sync_debug import gpu_sync_allowed

    calls: list[torch.Tensor] = []
    model.routing_calls = calls
    generator = torch.Generator(device="cuda").manual_seed(0)

    def hook(moe: Inkling2MoE, args: tuple) -> None:
        gate = moe.gate
        logits = linear_k_groups(args[0].contiguous(), gate.weight)
        ids = torch.topk(logits.sigmoid() + gate.bias, gate.topk).indices
        # The test's own reference copy; CI runs with VLLM_GPU_SYNC_CHECK=error.
        with gpu_sync_allowed():
            calls.append(ids.sort(-1).values.cpu())

    for module in model.modules():
        if isinstance(module, Inkling2LoopDecoderLayer):
            # Unit-scale gates: tanh(gain), tanh(forget) ~ O(1).
            down, gain_forget = module.controller_down, module.controller_gain_forget
            down.normal_(std=down.shape[-1] ** -0.5, generator=generator)
            gain_forget.normal_(std=gain_forget.shape[-1] ** -0.5, generator=generator)
        if isinstance(module, Inkling2MoE):
            module.gate.weight.normal_(generator=generator)
            module.gate.bias.zero_()
            module.register_forward_pre_hook(hook)


def _routing_calls(model: torch.nn.Module) -> list[torch.Tensor]:
    return model.routing_calls


@pytest.mark.parametrize("looped", [False, True])
def test_tiny_model_routed_experts(tmp_path, enable_pickle, looped):
    """Routed-experts capture keeps a row per layer visit, in execution
    order (a loop core layer has one per pass), each the experts that visit's
    MoE call routed every prompt token to."""
    config = TINY_LOOP_CONFIG if looped else TINY_CONFIG
    (tmp_path / "config.json").write_text(json.dumps(config))
    llm = LLM(
        model=str(tmp_path),
        load_format="dummy",
        skip_tokenizer_init=True,
        tokenizer_mode="hf",
        gpu_memory_utilization=0.035,
        max_model_len=2048,
        max_num_seqs=4,
        enforce_eager=True,
        enable_return_routed_experts=True,
        limit_mm_per_prompt={"image": 0},
    )
    # Looped: the prelude, 3 passes of the 2 core layers, then the coda.
    num_visits = 8 if looped else 2
    assert llm.llm_engine.model_config.hf_text_config.num_layer_visits == num_visits
    llm.apply_model(_record_moe_routing)
    prompt = list(range(2, 202))
    output = llm.generate(
        [{"prompt_token_ids": prompt}],
        SamplingParams(max_tokens=1, ignore_eos=True, detokenize=False),
    )[0]
    calls = llm.apply_model(_routing_calls)[0]
    # One prefill forward: one MoE call per layer visit.
    assert len(calls) == num_visits and all(len(c) == len(prompt) for c in calls)
    experts = output.outputs[0].routed_experts
    assert experts is not None and experts.shape[1:] == (num_visits, 8)
    n = len(prompt) - 1  # the last prompt token's row may be the sampled token's
    captured = torch.from_numpy(experts[:n].astype("int64")).sort(-1).values
    for visit, call in enumerate(calls):
        torch.testing.assert_close(captured[:, visit], call[:n], atol=0, rtol=0)
    if looped:
        # The passes of a core layer route differently (no row is reused).
        assert not torch.equal(captured[:, 1], captured[:, 3])


def _tiny_loop_objects_llm(tmp_path, monkeypatch, async_scheduling):
    """An in-process engine on the tiny loop model, writing to a dict store
    that `object_store` names."""
    from tests.distributed.aux_output_connector.utils import DictObjectStore

    # The engine and its worker run in this process, on the store they create.
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setattr(DictObjectStore, "created", [])
    (tmp_path / "config.json").write_text(json.dumps(TINY_LOOP_CONFIG))
    llm = LLM(
        model=str(tmp_path),
        load_format="dummy",
        skip_tokenizer_init=True,
        tokenizer_mode="hf",
        gpu_memory_utilization=0.035,
        max_model_len=2048,
        max_num_seqs=4,
        enforce_eager=True,
        enable_prefix_caching=True,
        async_scheduling=async_scheduling,
        aux_output_config={
            "enable_return_routed_experts": True,
            "object_key_prefix": "job",
            "object_store": f"{DictObjectStore.__module__}.DictObjectStore",
            "object_store_extra_config": {"bucket": "b"},
        },
        limit_mm_per_prompt={"image": 0},
    )
    [store] = DictObjectStore.created
    assert store.extra_config == {"bucket": "b"}
    return llm, store


@pytest.mark.parametrize("async_scheduling", [True, False], ids=["async", "sync"])
def test_tiny_loop_model_routed_experts_objects(
    tmp_path, enable_pickle, monkeypatch, async_scheduling
):
    """A namespace request stores a row per layer visit in its objects, as
    the inline output returns them. The engine runs in-process, so its next
    step hands the last request's writes to the worker."""
    from vllm.distributed.aux_output_connector.object_store import read_aux_rows

    llm, store = _tiny_loop_objects_llm(tmp_path, monkeypatch, async_scheduling)
    num_visits = llm.llm_engine.model_config.hf_text_config.num_layer_visits
    assert num_visits == 8
    llm.apply_model(_record_moe_routing)
    prompt = [{"prompt_token_ids": list(range(2, 202))}]

    def generate(**kwargs):
        params = SamplingParams(
            max_tokens=4, temperature=0, ignore_eos=True, detokenize=False, **kwargs
        )
        return llm.generate(prompt, params)[0].outputs[0]

    inline = generate().routed_experts
    listing = generate(aux_namespace="g/1").aux_objects
    generate()
    assert inline is not None and listing
    assert [start for _, start, _ in listing][0] == 0
    assert all(a[2] == b[1] for a, b in zip(listing, listing[1:]))
    rows = [
        read_aux_rows(store, [key], "routed_experts", timeout_s=30)[: end - start]
        for key, start, end in listing
    ]
    objects = np.concatenate(rows)
    assert objects.shape[1:] == (num_visits, 8)
    np.testing.assert_array_equal(objects, inline)


def _chunked_mixed_run(model: str, enforce_eager: bool) -> list[list[float]]:
    llm = LLM(
        model=model,
        load_format="dummy",
        skip_tokenizer_init=True,
        tokenizer_mode="hf",
        gpu_memory_utilization=0.035,
        max_model_len=2048,
        max_num_seqs=16,
        max_num_batched_tokens=256,
        enforce_eager=enforce_eager,
        # The 256-token piecewise graph is captured on 16 verify-sized requests.
        compilation_config={"cudagraph_capture_sizes": [8, 16, 256]},
        limit_mm_per_prompt={"image": 0},
    )
    # The short prompts decode while the long one prefills in 256-token chunks.
    prompts = [[7, 8, 9], [5, 5, 5], list(range(2, 702))]
    outputs = llm.generate(
        [{"prompt_token_ids": p} for p in prompts],
        SamplingParams(
            max_tokens=8,
            temperature=0,
            ignore_eos=True,
            detokenize=False,
            prompt_logprobs=0,
        ),
    )
    del llm
    logprobs = []
    for o in outputs:
        assert o.prompt_logprobs is not None
        logprobs.append(
            [next(iter(d.values())).logprob for d in o.prompt_logprobs[1:] if d]
        )
    return logprobs


def test_piecewise_graphs_match_eager(tmp_path):
    """Piecewise CUDA graphs are captured on a dummy decode-sized batch and
    replayed for any batch of the same token count, so attention must plan
    from the live batch: a chunked prompt prefilling alongside decodes runs
    and matches eager."""
    (tmp_path / "config.json").write_text(json.dumps(TINY_CONFIG))
    graph = _chunked_mixed_run(str(tmp_path), enforce_eager=False)
    eager = _chunked_mixed_run(str(tmp_path), enforce_eager=True)
    for g, e in zip(graph, eager):
        torch.testing.assert_close(torch.tensor(g), torch.tensor(e), rtol=0, atol=1e-2)


# `quantization_config` of the released checkpoints' config.json.
RELEASED_QUANTIZATION_CONFIGS = {
    "NVFP4": {"quant_method": "modelopt", "quant_algo": "NVFP4"},
    "MXFP8": {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "scale_fmt": "ue8m0",
        "weight_block_size": [1, 32],
        "modules_to_not_convert": [
            "model.llm.unembed",
            "model.visual",
            "model.llm.layers.*.attn.*",
        ],
    },
}


@pytest.mark.parametrize("quant_algo", ["NVFP4", "MXFP8"])
def test_released_checkpoint_quant_config(tmp_path, quant_algo):
    """Released checkpoints declare a ModelOpt (NVFP4) or FP8 (MXFP8)
    `quantization_config` in config.json; the per-tensor precisions in
    hf_quant_config.json win."""
    from vllm.config import LoadConfig, ModelConfig
    from vllm.model_executor.model_loader.weight_utils import get_quant_config
    from vllm.models.inkling2.quant_config import Inkling2QuantConfig

    config = {
        **TINY_CONFIG,
        "quantization_config": RELEASED_QUANTIZATION_CONFIGS[quant_algo],
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    experts = "model.llm.layers.{}.mlp.experts.w13_weight"
    mxfp8 = [0, 1] if quant_algo == "MXFP8" else [1]
    groups = [
        {"compute_dtype": "mxfp8", "tensors": [experts.format(i) for i in mxfp8]},
        {"compute_dtype": "bfloat16", "tensors": ["model.llm.embed_tokens.weight"]},
    ]
    if quant_algo == "NVFP4":
        groups.append({"compute_dtype": "nvfp4", "tensors": [experts.format(0)]})
    (tmp_path / "hf_quant_config.json").write_text(
        json.dumps({"quant_method": "inkling2", "model": {"parameter_groups": groups}})
    )
    model_config = ModelConfig(
        model=str(tmp_path), skip_tokenizer_init=True, tokenizer_mode="hf"
    )
    assert model_config.quantization == "inkling2"
    quant_config = get_quant_config(model_config, LoadConfig())
    assert isinstance(quant_config, Inkling2QuantConfig)
    assert [quant_config.is_mxfp8_layer(i) for i in range(2)] == [
        i in mxfp8 for i in range(2)
    ]


@pytest.mark.parametrize("tokens", [1, 8, 17])
def test_sm12x_logits_preserve_softcap_and_vocabulary(tmp_path, dist_init, tokens):
    """The low-M head still applies softcaps and excludes padded vocabulary."""
    from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
    from vllm.models.inkling2.nvidia.model import Inkling2LogitsProcessor

    (tmp_path / "config.json").write_text(json.dumps(TINY_CONFIG))
    config = VllmConfig(
        model_config=ModelConfig(
            model=str(tmp_path),
            dtype="bfloat16",
            skip_tokenizer_init=True,
            tokenizer_mode="hf",
            max_model_len=2048,
            hf_overrides={"head_dtype": "float32"},
        )
    )
    torch.manual_seed(0)
    with set_current_vllm_config(config), torch.device("cuda"):
        head = ParallelLMHead(1024, 512, params_dtype=torch.bfloat16)
        head.weight.data.normal_(std=0.02)
        candidate = Inkling2LogitsProcessor(1024, org_vocab_size=1000, soft_cap=3.0)
        control = LogitsProcessor(1024, org_vocab_size=1000, soft_cap=3.0)
        x = torch.randn(tokens, 512).bfloat16()
        actual, expected = candidate(head, x), control(head, x)
    assert actual.shape == (tokens, 1000)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    assert torch.equal(actual.argmax(-1), expected.argmax(-1))
