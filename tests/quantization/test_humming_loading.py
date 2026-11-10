# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA Humming loading regressions, selected by the Humming loading CI job."""

import json
import math

import torch

from tests.models.utils import check_logprobs_close
from vllm.config import VllmConfig, set_current_vllm_config


@torch.inference_mode()
def test_humming_packed_experts_avoid_integer_expansion(
    dist_init, workspace_init, record_property
):
    """Opting out bounds loading memory and preserves the default kernel output."""
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.layers.fused_moe import FusedMoEFactory
    from vllm.model_executor.layers.quantization.humming import HummingConfig
    from vllm.utils.humming import HummingWeightSchema

    torch.manual_seed(29)
    experts, hidden, intermediate = 8, 512, 256
    schema = HummingWeightSchema(b_dtype="int3", weight_scale_group_size=128)
    checkpoint = {}
    for name, shape in (
        ("w13", (experts, 2 * intermediate, hidden)),
        ("w2", (experts, hidden, intermediate)),
    ):
        weights = torch.randn(shape, dtype=torch.bfloat16, device="cuda") / 10
        tensors = schema.quant_tensor(weights, schema, torch.bfloat16)
        # A zero code makes the default retain the original format, but only
        # after allocating the full INT32 projection to check its codes.
        tensors["weight"].view(-1)[0] = 0
        projections = ("gate_proj", "up_proj") if name == "w13" else ("down_proj",)
        for key, value in tensors.items():
            for projection, chunk in zip(projections, value.chunk(len(projections), 1)):
                for expert, tensor in enumerate(chunk):
                    checkpoint[f"{expert}.{projection}.{key}"] = tensor
    checkpoint_bytes = sum(tensor.nbytes for tensor in checkpoint.values())
    record_property("packed_checkpoint_bytes", checkpoint_bytes)
    x = torch.randn(17, hidden, dtype=torch.bfloat16, device="cuda") / 10
    logits = torch.randn(17, experts, device="cuda")
    outputs = []
    for overrides in ({}, {"moe_convert_symmetric_integer": False}):
        config = VllmConfig()
        quant = HummingConfig.from_config(
            {
                "quant_method": "humming",
                "weight_dtype": "int3",
                "input_dtype": "bfloat16",
                "weight_scale_group_size": 128,
                **overrides,
            }
        )
        torch.accelerator.synchronize()
        before = torch.accelerator.memory_allocated()
        torch.accelerator.reset_peak_memory_stats()
        with set_current_vllm_config(config), torch.device("cuda"):
            runner = FusedMoEFactory(
                num_experts=experts,
                top_k=2,
                hidden_size=hidden,
                intermediate_size=intermediate,
                params_dtype=torch.bfloat16,
                quant_config=quant,
                prefix="model.layers.0.mlp.experts",
            )
            layer = runner.routed_experts
            loaded = list(layer.load_weights(list(checkpoint.items())))
            assert len(loaded) == len(checkpoint)
            layer.quant_method.process_weights_after_loading(layer)
        torch.accelerator.synchronize()
        loading_peak = torch.accelerator.max_memory_allocated() - before
        record_property(
            "packed_loading_peak_bytes" if overrides else "default_loading_peak_bytes",
            loading_peak,
        )
        if overrides:
            # Permit packed copies and repacking, but not an INT32 projection.
            assert loading_peak < 4 * checkpoint_bytes
        with set_current_vllm_config(config), set_forward_context(None, config):
            outputs.append(runner(hidden_states=x, router_logits=logits).clone())
        del layer, runner
    assert torch.isfinite(outputs[1]).all()
    torch.testing.assert_close(outputs[1], outputs[0], rtol=0, atol=0)


def test_humming_packed_experts_generate_with_real_weights(
    vllm_runner, record_property, enable_pickle
):
    """The public loading option reaches MoE experts and preserves generation."""
    model_name = "RedHatAI/Qwen3-30B-A3B-quantized.w4a16"
    revision = "1c842a5000febac4b335c94a04789a9d4756a18e"
    prompts = [
        "The capital of France is",
        "Explain in one sentence why the sky appears blue.",
        "Write a Python function that adds two numbers.",
        "Continue the sequence: 2, 4, 6, 8,",
    ]

    def inspect_model(model):
        from vllm.model_executor.layers.quantization.humming import HummingMoEMethod

        return [
            (
                module.quant_method.processed,
                module.quant_method.quant_config.moe_convert_symmetric_integer,
            )
            for module in model.modules()
            if isinstance(getattr(module, "quant_method", None), HummingMoEMethod)
        ]

    outputs = []
    for convert in (True, False):

        def override(config, convert=convert):
            if not convert and hasattr(config, "quantization_config"):
                config.quantization_config["moe_convert_symmetric_integer"] = False
            return config

        with vllm_runner(
            model_name,
            revision=revision,
            tokenizer_revision=revision,
            trust_remote_code=False,
            load_format="auto",
            quantization="humming",
            hf_overrides=override,
            dtype="bfloat16",
            tensor_parallel_size=1,
            enable_expert_parallel=False,
            enable_eplb=False,
            enforce_eager=True,
            max_model_len=2048,
            max_num_seqs=1,
            kv_cache_memory_bytes=512 * 1024 * 1024,
            enable_prefix_caching=False,
            enable_chunked_prefill=True,
            seed=29,
        ) as runner:
            layers = runner.apply_model(inspect_model)
            assert layers == [[(True, convert)] * 48]
            result = runner.generate_greedy_logprobs(prompts, 32, 5)
            assert len(result) == len(prompts)
            for tokens, _, logprobs in result:
                assert tokens and logprobs
                assert all(
                    math.isfinite(value.logprob)
                    for step in logprobs
                    for value in step.values()
                )
            record_property(
                "default_outputs" if convert else "packed_outputs",
                json.dumps(
                    [
                        (
                            tokens,
                            text,
                            [{k: v.logprob for k, v in s.items()} for s in logs],
                        )
                        for tokens, text, logs in result
                    ]
                ),
            )
            outputs.append(result)

    check_logprobs_close(
        outputs_0_lst=outputs[0],
        outputs_1_lst=outputs[1],
        name_0="default",
        name_1="packed",
        always_check_logprobs=True,
    )
