# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm import LLM, SamplingParams
from vllm.transformers_utils.configs.sensenova_loop import SenseNovaLoopConfig

from ...utils import check_logprobs_close

pytestmark = pytest.mark.hybrid_model


@pytest.mark.slow_test
def test_public_checkpoint_matches_hf(hf_runner, vllm_runner, example_prompts):
    """Load the public export natively and compare against its HF implementation."""
    model = "sensenova/SenseNova-Loop-9B-Preview"
    prompts = example_prompts[:2]
    with hf_runner(model, dtype="bfloat16") as hf_model:
        hf_outputs = hf_model.generate_greedy_logprobs_limit(prompts, 16, 5)

    with vllm_runner(
        model,
        dtype="bfloat16",
        trust_remote_code=False,
        enable_chunked_prefill=True,
        max_model_len=512,
        max_num_seqs=1,
        gpu_memory_utilization=0.05,
        kv_cache_memory_bytes=256 * 1024**2,
        compilation_config={"cudagraph_capture_sizes": [1]},
    ) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy_logprobs(prompts, 16, 5)

    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )


@pytest.fixture
def loop_checkpoint(tmp_path):
    from safetensors.torch import save_file
    from transformers import Qwen3_5ForCausalLM

    config = SenseNovaLoopConfig(
        architectures=["SenseNovaLoopForConditionalGeneration"],
        text_config={
            "vocab_size": 256,
            "hidden_size": 256,
            "intermediate_size": 512,
            "num_hidden_layers": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 128,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 2,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "layer_types": ["linear_attention", "full_attention"] * 2,
            "max_position_embeddings": 128,
            "eos_token_id": 255,
        },
    )
    torch.manual_seed(0)
    weights = Qwen3_5ForCausalLM(config.text_config).bfloat16().state_dict()
    for layer in (1, 2):
        for router in ("mhc_attn", "mhc_ffn"):
            weights[f"model.layers.{layer}.{router}.weight"] = (
                torch.randn(8, 512) * 0.01
            )
            weights[f"model.layers.{layer}.{router}.bias"] = torch.randn(8) * 0.01
    # Match the public multimodal export, including unused vision and MTP weights.
    weights = {
        name.replace("model.", "model.language_model.", 1): value
        for name, value in weights.items()
    }
    weights["model.visual.patch_embed.proj.weight"] = torch.zeros(1)
    weights["mtp.fc.weight"] = torch.zeros(1)
    save_file(weights, tmp_path / "model.safetensors")
    config.loop_settings = {"start": 1, "end": 3, "eval_count": 2}
    config.mhc = {
        "start": 1,
        "end": 3,
        "streams": 2,
        "iterations": 20,
        "fixed_identity": False,
        "pre_scale": 1.0,
    }
    config.save_pretrained(tmp_path)
    return str(tmp_path)


@pytest.mark.parametrize("load_format", ["auto", "dummy"])
@pytest.mark.parametrize(
    "loop_settings, indices",
    [
        ({"start": 1, "end": 3, "eval_count": 3}, [0, 1, 2, 1, 2, 1, 2, 3]),
        (None, [0, 1, 2, 3]),
    ],
)
def test_loop_overrides_reach_loaded_layers(
    loop_checkpoint, monkeypatch, loop_settings, indices, load_format
):
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    llm = LLM(
        model=loop_checkpoint,
        skip_tokenizer_init=True,
        dtype="bfloat16",
        load_format=load_format,
        enforce_eager=True,
        max_model_len=32,
        max_num_seqs=2,
        gpu_memory_utilization=0.05,
        kv_cache_memory_bytes=16 * 1024**2,
        hf_overrides={
            "loop_settings": loop_settings,
            "mhc": {
                "start": 1,
                "end": 3,
                "streams": 2,
                "iterations": 20,
                "fixed_identity": False,
                "pre_scale": 0.5,
            },
        },
    )
    try:

        def inspect_layers(model):
            layers = model.model.layers
            assert len({id(layer) for layer in layers}) == len(layers)
            physical_layers: dict[int, torch.nn.Module] = {}
            for layer in layers:
                first = physical_layers.setdefault(layer.physical_index, layer)
                parameters = dict(first.named_parameters())
                assert all(
                    parameter is parameters[name]
                    for name, parameter in layer.named_parameters()
                )
                if layer.mhc_active:
                    assert layer.mhc_attn.weight.dtype == torch.float32
                    assert layer.mhc_ffn.weight.dtype == torch.float32
            return (
                [layer.physical_index for layer in layers],
                [layer.mhc_attn.pre_scale for layer in layers if layer.mhc_active],
            )

        assert llm.apply_model(inspect_layers) == [
            (indices, [0.5] * (len(indices) - 2))
        ]
        output = llm.generate(
            {"prompt_token_ids": [1, 2, 3]},
            SamplingParams(temperature=0, max_tokens=2, ignore_eos=True),
        )
        assert len(output[0].outputs[0].token_ids) == 2
    finally:
        llm.llm_engine.engine_core.shutdown()


def test_loop_decode_graph_matches_eager(loop_checkpoint, monkeypatch):
    from vllm.config.compilation import CompilationMode, CUDAGraphMode
    from vllm.platforms import current_platform

    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    outputs = []
    for eager in (True, False):
        llm = LLM(
            model=loop_checkpoint,
            skip_tokenizer_init=True,
            dtype="bfloat16",
            enforce_eager=eager,
            max_model_len=32,
            max_num_seqs=2,
            gpu_memory_utilization=0.05,
            kv_cache_memory_bytes=16 * 1024**2,
            compilation_config={"cudagraph_capture_sizes": [1, 2]},
        )
        try:

            def inspect_graphs(model):
                from vllm.compilation.counter import compilation_counter

                config = model.vllm_config.compilation_config
                return (
                    config.mode.name,
                    config.cudagraph_mode.name,
                    compilation_counter.num_cudagraph_captured,
                )

            mode, graph_mode, captures = llm.apply_model(inspect_graphs)[0]
            if eager:
                assert mode == CompilationMode.NONE.name
                assert graph_mode == CUDAGraphMode.NONE.name
            else:
                if current_platform.is_cuda():
                    assert mode == CompilationMode.NONE.name
                    assert graph_mode == CUDAGraphMode.FULL_DECODE_ONLY.name
                assert captures > 0
            output = llm.generate(
                [{"prompt_token_ids": [1, 2, 3]}, {"prompt_token_ids": [4, 5, 6]}],
                SamplingParams(temperature=0, max_tokens=4, ignore_eos=True),
            )
            outputs.append([request.outputs[0].token_ids for request in output])
        finally:
            llm.llm_engine.engine_core.shutdown()
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize(
    "offload",
    [
        {"cpu_offload_gb": 1},
        {"offload_group_size": 2, "offload_num_in_group": 1},
    ],
)
def test_loop_rejects_cpu_offload_before_loading(loop_checkpoint, offload):
    with pytest.raises(NotImplementedError, match="SenseNova Loop.*CPU offload"):
        llm = LLM(
            model=loop_checkpoint,
            skip_tokenizer_init=True,
            dtype="bfloat16",
            enforce_eager=True,
            max_model_len=32,
            gpu_memory_utilization=0.05,
            kv_cache_memory_bytes=16 * 1024**2,
            **offload,
        )
        llm.llm_engine.engine_core.shutdown()


def test_loop_rejects_invalid_mhc_override_before_loading(loop_checkpoint):
    settings = SenseNovaLoopConfig.from_pretrained(loop_checkpoint).mhc
    llm = None
    try:
        with pytest.raises(ValueError, match="Invalid mHC pre_scale"):
            llm = LLM(
                model=loop_checkpoint,
                skip_tokenizer_init=True,
                dtype="bfloat16",
                enforce_eager=True,
                max_model_len=32,
                max_num_seqs=1,
                gpu_memory_utilization=0.05,
                kv_cache_memory_bytes=16 * 1024**2,
                hf_overrides={"mhc": dict(settings, pre_scale=0)},
            )
    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()
