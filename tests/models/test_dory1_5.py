# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.exceptions import VLLMValidationError
from vllm.model_executor.models.dory1_5 import (
    Dory1_5LoopConditioning,
    Dory1_5LoopSeam,
)
from vllm.transformers_utils.configs.dory1_5 import (
    Dory1_5Config,
)


@pytest.mark.parametrize("seam", [False, True])
@torch.no_grad()
def test_loop_table_reload_preserves_captured_storage(seam):
    config = Dory1_5Config(
        hidden_size=8,
        loop_conditioning_hidden_size=4,
        loop_adapter_rank=0,
        n_recurrent_loops=2,
    )
    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))):
        module = (
            Dory1_5LoopSeam(config) if seam else Dory1_5LoopConditioning(config, "test")
        )
    steps = [0, 1, 0, 1, 2]
    conds = [torch.randn(4) for _ in steps]
    if seam:
        module.update(conds, steps)
    else:
        module.update(conds)
    captured_table = module.table.view_as(module.table)
    before = captured_table.clone()
    pointer = captured_table.data_ptr()
    head = module.entry_beta if seam else module.beta_head
    head.bias.add_(1)
    if seam:
        module.update(conds, steps)
    else:
        module.update(conds)
    assert module.table.data_ptr() == pointer
    torch.testing.assert_close(captured_table, module.table)
    torch.testing.assert_close(captured_table[:, 8:16], before[:, 8:16] + 1)
    assert "table" not in module.state_dict()


def _cache_layout(model):
    from vllm.model_executor.layers.attention import Attention
    from vllm.model_executor.layers.mamba.abstract import MambaBase

    return (
        sum(isinstance(m, Attention) for m in model.modules()),
        sum(isinstance(m, MambaBase) for m in model.modules()),
    )


@pytest.mark.usefixtures("enable_pickle")
@pytest.mark.parametrize("cache_mode", ["per_loop", "last_loop"])
@pytest.mark.parametrize("gdp_dims", [(64, 32), (128, 64)])
def test_loop_cache_modes(vllm_runner, tmp_path, cache_mode, gdp_dims):
    """Last-loop attention must retain independent GDP state through decoding."""
    from vllm import SamplingParams

    Dory1_5Config(
        architectures=["Dory1_5ForCausalLM"],
        vocab_size=64,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        n_input_layers=0,
        n_recurrent_layers=2,
        n_output_layers=0,
        n_recurrent_loops=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=gdp_dims[0],
        linear_value_head_dim=gdp_dims[1],
        loop_conditioning_hidden_size=8,
        loop_adapter_rank=0,
    ).save_pretrained(tmp_path)
    with vllm_runner(
        str(tmp_path),
        load_format="dummy",
        skip_tokenizer_init=True,
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=64,
        max_num_seqs=2,
        enable_prefix_caching=False,
        hf_overrides={"recurrent_kv_cache_mode": cache_mode},
    ) as runner:
        outputs = runner.llm.generate(
            [{"prompt_token_ids": [1, 2, 3]}, {"prompt_token_ids": [4, 5]}],
            SamplingParams(temperature=0, max_tokens=4, ignore_eos=True, logprobs=1),
        )
        for output in outputs:
            completion = output.outputs[0]
            assert len(completion.token_ids) == 4
            assert all(
                torch.isfinite(torch.tensor(step[token].logprob))
                for token, step in zip(completion.token_ids, completion.logprobs)
            )
        attention_caches = 2 if cache_mode == "per_loop" else 1
        assert runner.apply_model(_cache_layout) == [(attention_caches, 2)]


@pytest.mark.parametrize(
    "precision,enforce_eager,cache_mode",
    [
        # Regular CI covers all pairs; slow cases complete the three-way matrix.
        ("BF16", False, "per_loop"),
        ("BF16", True, "last_loop"),
        ("NVFP4", False, "last_loop"),
        ("NVFP4", True, "per_loop"),
        pytest.param("BF16", False, "last_loop", marks=pytest.mark.slow_test),
        pytest.param("BF16", True, "per_loop", marks=pytest.mark.slow_test),
        pytest.param("NVFP4", False, "per_loop", marks=pytest.mark.slow_test),
        pytest.param("NVFP4", True, "last_loop", marks=pytest.mark.slow_test),
    ],
)
def test_mixed_request_loops_match_separate_requests(
    vllm_runner, cache_mode, enforce_eager, precision
):
    """A graph for batch budget 3 must not run the engine's maximum budget 4."""
    from tests.models.utils import check_logprobs_close
    from vllm import SamplingParams

    prompts = [
        "The capital of France is",
        "Compute 12 + 30. The answer is",
        "Write a Python function that adds two numbers.",
    ]
    params = [
        SamplingParams(
            temperature=0,
            max_tokens=16,
            ignore_eos=True,
            logprobs=5,
            extra_args={"num_loops": count},
        )
        for count in (1, 2, 3)
    ]
    with vllm_runner(
        f"nvidia/NVIDIA-Nemotron-Labs-Dory1.5-4B-{precision}",
        dtype="bfloat16",
        max_model_len=512,
        max_num_seqs=4,
        max_num_batched_tokens=256,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        enforce_eager=enforce_eager,
        compilation_config={"cudagraph_capture_sizes": [1, 2, 4]},
        hf_overrides={"recurrent_kv_cache_mode": cache_mode},
    ) as runner:
        fixed = [
            runner.llm.generate([prompt], param)[0]
            for prompt, param in zip(prompts, params)
        ]
        mixed = runner.llm.generate(prompts, params)

        def results(outputs):
            return [
                (list(o.outputs[0].token_ids), o.outputs[0].text, o.outputs[0].logprobs)
                for o in outputs
            ]

        check_logprobs_close(
            outputs_0_lst=results(fixed),
            outputs_1_lst=results(mixed),
            name_0="separate",
            name_1="mixed",
        )

    with vllm_runner(
        f"nvidia/NVIDIA-Nemotron-Labs-Dory1.5-4B-{precision}",
        dtype="bfloat16",
        max_model_len=512,
        max_num_seqs=4,
        max_num_batched_tokens=256,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        enforce_eager=enforce_eager,
        compilation_config={"cudagraph_capture_sizes": [1, 2, 4]},
        hf_overrides={"recurrent_kv_cache_mode": cache_mode, "n_recurrent_loops": 2},
    ) as runner:
        from copy import deepcopy

        fixed_params = deepcopy(params[1])
        fixed_params.extra_args = None
        fixed_engine = runner.llm.generate([prompts[1]], fixed_params)
        check_logprobs_close(
            outputs_0_lst=results(fixed_engine),
            outputs_1_lst=results([mixed[1]]),
            name_0="fixed engine",
            name_1="request budget",
        )


@pytest.mark.parametrize("cache_mode", ["per_loop", "last_loop"])
def test_request_loops_isolate_prefixes_at_engine_admission(
    vllm_runner, monkeypatch, cache_mode
):
    """Cache keys depend on the budget even when a frontend supplies its own hashes."""
    import time

    from vllm import SamplingParams
    from vllm.v1.engine import EngineCoreRequest

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    def params(count):
        return SamplingParams(
            temperature=0,
            max_tokens=4,
            ignore_eos=True,
            extra_args={"num_loops": count},
            logprobs=5,
        )

    with vllm_runner(
        "nvidia/NVIDIA-Nemotron-Labs-Dory1.5-4B-BF16",
        dtype="bfloat16",
        max_model_len=1024,
        max_num_seqs=2,
        max_num_batched_tokens=256,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        enforce_eager=True,
        hf_overrides={"recurrent_kv_cache_mode": cache_mode},
    ) as runner:
        tokens = runner.llm.get_tokenizer().encode(
            "Explain the scientific observations. " * 100
        )[:401]
        prompt = {"prompt_token_ids": tokens, "cache_salt": "caller"}
        outputs = [runner.llm.generate([prompt], params(k))[0] for k in (1, 1, 2, 2)]
        hits = [o.num_cached_tokens for o in outputs]
        assert hits[0] == hits[2] == 0
        assert hits[1] > 0 and hits[3] > 0
        if cache_mode == "per_loop":
            from tests.models.utils import check_logprobs_close

            def result(o):
                c = o.outputs[0]
                return (list(c.token_ids), c.text, c.logprobs)

            check_logprobs_close(
                outputs_0_lst=[result(outputs[0]), result(outputs[2])],
                outputs_1_lst=[result(outputs[1]), result(outputs[3])],
                name_0="cold prefix",
                name_1="warm prefix",
            )
        core = runner.llm.llm_engine.engine_core.engine_core

        def request(count, hashes=None):
            return EngineCoreRequest(
                request_id=f"budget-{count}",
                prompt_token_ids=tokens,
                sampling_params=params(count),
                pooling_params=None,
                mm_features=None,
                arrival_time=time.time(),
                lora_request=None,
                cache_salt="caller",
                data_parallel_rank=None,
                block_hashes=hashes,
            )

        baseline, _ = core.preprocess_add_request(request(4))
        supplied_hashes = list(baseline.block_hashes)
        default_request = request(4, supplied_hashes)
        default_admitted, _ = core.preprocess_add_request(default_request)
        assert default_request.block_hashes is supplied_hashes
        assert default_admitted.block_hashes == baseline.block_hashes
        incoming = request(1, supplied_hashes)
        admitted, _ = core.preprocess_add_request(incoming)
        assert incoming.block_hashes is None
        assert admitted.block_hashes and admitted.block_hashes != baseline.block_hashes
        for invalid in (True, 0, 5, "2"):
            with pytest.raises(VLLMValidationError, match="num_loops"):
                core.preprocess_add_request(request(invalid))
            with pytest.raises(VLLMValidationError, match="num_loops"):
                runner.llm.generate(["test"], params(invalid))


@pytest.mark.parametrize(
    "case,match",
    [
        ("shape", "shape/dtype"),
        ("dtype", "shape/dtype"),
        ("missing", "Incomplete checkpoint"),
        ("duplicate", "Duplicate source tensor"),
        ("unknown", "Unexpected checkpoint tensor"),
    ],
)
def test_packed_loader_rejects_inconsistent_tensors(case, match):
    from vllm.model_executor.models.dory1_5 import _load_per_matrix_weights

    module = torch.nn.Linear(16, 8, dtype=torch.bfloat16)
    tensors = [(name, torch.ones_like(p)) for name, p in module.named_parameters()]
    if case == "shape":
        tensors[0] = ("weight", torch.ones(8, 8, dtype=torch.bfloat16))
    elif case == "dtype":
        tensors[0] = ("weight", tensors[0][1].float())
    elif case == "missing":
        tensors.pop()
    elif case == "duplicate":
        tensors.append(tensors[0])
    else:
        tensors.append(("missing.weight", tensors[0][1]))
    with pytest.raises(ValueError, match=match):
        _load_per_matrix_weights(module, tensors)


@pytest.mark.parametrize(
    "allocation,match",
    [
        (None, "allocation file was not found"),
        ({"formats": []}, "formats mapping"),
        ({"formats": {"model.layers.0.mixer.up_proj.weight": "FP3"}}, "Unsupported"),
        ({"formats": {"missing.weight": "NVFP4"}}, "only pack linear matrix weights"),
    ],
)
def test_packed_model_rejects_invalid_allocation(
    dist_init, tmp_path, allocation, match
):
    import json

    from vllm.config import ModelConfig
    from vllm.model_executor.models.dory1_5 import (
        Dory1_5ForCausalLM,
    )

    Dory1_5Config(
        architectures=["Dory1_5ForCausalLM"],
        vocab_size=64,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        layer_types=["mlp"],
        n_input_layers=0,
        n_recurrent_layers=1,
        n_output_layers=0,
        n_recurrent_loops=1,
        loop_conditioning_hidden_size=8,
        loop_adapter_rank=0,
        per_tensor_quantization={"version": 1, "allocation": "allocation.json"},
    ).save_pretrained(tmp_path)
    if allocation is not None:
        (tmp_path / "allocation.json").write_text(json.dumps(allocation))
    config = VllmConfig(
        model_config=ModelConfig(
            model=str(tmp_path),
            skip_tokenizer_init=True,
            dtype="bfloat16",
            max_model_len=64,
            enforce_eager=True,
        )
    )
    with set_current_vllm_config(config), pytest.raises(ValueError, match=match):
        Dory1_5ForCausalLM(vllm_config=config)
