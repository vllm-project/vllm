# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.config import SpeculativeConfig
from vllm.model_executor.models.interfaces import supports_eagle3
from vllm.platforms import current_platform

DEVICE_TYPE = current_platform.device_type


@pytest.mark.parametrize(
    "model_path",
    [
        pytest.param(
            "nm-testing/SpeculatorLlama3-1-8B-Eagle3-converted-0717-quantized",
            id="llama3-eagle3-speculator",
        ),
        pytest.param(
            "nm-testing/Speculator-Qwen3-8B-Eagle3-converted-071-quantized",
            id="qwen3-eagle3-speculator",
        ),
        pytest.param(
            "nm-testing/Speculator-Qwen3-8B-Eagle3-converted-071-quantized-w4a16",
            id="qwen3-eagle3-speculator-w4a16-verifier",
            marks=pytest.mark.skipif(
                current_platform.is_rocm(),
                reason="The tests are skipped on rocm platform.",
            ),
        ),
    ],
)
def test_eagle3_speculators_model(
    vllm_runner, example_prompts, model_path, monkeypatch
):
    """Test Eagle3 speculators models properly initialize speculative decoding.

    This test verifies:
    1. Eagle3 support is detected for the model
    2. Speculative config is automatically initialized from embedded config
    3. The draft model path is correctly set to the speculators model
    4. Speculative tokens count is valid
    5. Text generation works with speculative decoding enabled
    """
    # Set environment variable for V1 engine serialization
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    with vllm_runner(model_path, dtype=torch.bfloat16) as vllm_model:
        # Verify Eagle3 support is detected
        eagle3_supported = vllm_model.apply_model(supports_eagle3)
        assert eagle3_supported, f"Eagle3 should be supported for {model_path}"

        vllm_config = vllm_model.llm.llm_engine.vllm_config

        assert isinstance(vllm_config.speculative_config, SpeculativeConfig), (
            "Speculative config should be initialized for speculators model"
        )

        spec_config = vllm_config.speculative_config
        assert spec_config.num_speculative_tokens > 0, (
            f"Expected positive speculative tokens, "
            f"got {spec_config.num_speculative_tokens}"
        )

        assert spec_config.model == model_path, (
            f"Draft model should be {model_path}, got {spec_config.model}"
        )

        vllm_outputs = vllm_model.generate_greedy(example_prompts, max_tokens=20)
        assert vllm_outputs, f"No outputs generated for speculators model {model_path}"


@pytest.fixture
def eagle3_fc_norm_model(tmp_path, dist_init):
    """Construct a small real draft head without downloading model weights."""
    from transformers import LlamaConfig

    from vllm.config import set_current_vllm_config
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.model_loader.utils import process_weights_after_loading
    from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
    from vllm.utils.torch_utils import set_random_seed

    def create(
        dtype=torch.bfloat16,
        num_aux=3,
        fc_norm=True,
        norm_before_fc=False,
        model_cls=Eagle3LlamaForCausalLM,
        use_aux=True,
    ):
        set_random_seed(0)
        config = dict(
            vocab_size=64,
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
        )
        target = tmp_path / "target"
        draft = tmp_path / (
            f"draft-{num_aux}-{fc_norm}-{norm_before_fc}-{use_aux}-{model_cls.__name__}"
        )
        LlamaConfig(
            **config, architectures=["LlamaForCausalLM"], num_hidden_layers=4
        ).save_pretrained(target)
        LlamaConfig(
            **config,
            architectures=["Eagle3LlamaForCausalLM"],
            num_hidden_layers=1,
            num_aux_hidden_states=num_aux,
            fc_norm=fc_norm,
            norm_before_fc=norm_before_fc,
            eagle_config={
                "eagle_aux_hidden_state_layer_ids": list(range(num_aux)),
                "use_aux_hidden_state": use_aux,
            },
        ).save_pretrained(draft)
        vllm_config = EngineArgs(
            model=str(target),
            skip_tokenizer_init=True,
            dtype="bfloat16",
            max_model_len=64,
            enforce_eager=True,
            speculative_config={
                "method": "eagle3",
                "model": str(draft),
                "num_speculative_tokens": 3,
            },
        ).create_engine_config()
        with set_current_vllm_config(vllm_config), torch.device(DEVICE_TYPE):
            model = model_cls(vllm_config=vllm_config).to(dtype=dtype)
            for param in model.parameters():
                if param.dtype.is_floating_point:
                    torch.nn.init.uniform_(param, -0.2, 0.2)
            process_weights_after_loading(
                model, vllm_config.model_config, torch.device(DEVICE_TYPE)
            )
        return model

    return create


def _combine_eagle3_aux(model, states):
    # On the base revision this exercises the incumbent caller operation.
    hook = getattr(model, "combine_aux_hidden_states", None)
    return hook(states) if hook else model.combine_hidden_states(torch.cat(states, -1))


@pytest.mark.parametrize("priority", [["vllm_c", "native"], ["native"]])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_aux", [1, 3])
@torch.inference_mode()
def test_eagle3_aux_combination_preserves_bytes(
    eagle3_fc_norm_model, priority, dtype, num_aux
):
    """Separate inputs must preserve the concat-first norm's values and inputs."""
    from vllm import ir

    model = eagle3_fc_norm_model(dtype=dtype, num_aux=num_aux)
    with ir.ops.rms_norm.set_priority(priority):
        cases = [
            (1, "contiguous"),
            (17, "contiguous"),
            (17, "row_stride"),
            (17, "feature_stride"),
            (17, "offset"),
        ]
        if priority == ["native"]:
            cases.append((0, "contiguous"))
        for rows, layout in cases:
            states = []
            for _ in range(num_aux):
                width = (
                    128
                    if layout == "feature_stride"
                    else 65
                    if layout in ("row_stride", "offset")
                    else 64
                )
                value = torch.randn(rows, width, dtype=dtype, device=DEVICE_TYPE)
                if layout == "feature_stride":
                    value = value[:, ::2]
                elif layout == "row_stride":
                    value = value[:, :64]
                elif layout == "offset":
                    value = value.flatten()[1 : 1 + rows * 64].view(rows, 64)
                states.append(value)
            originals = [value.clone() for value in states]
            expected = model.combine_hidden_states(torch.cat(states, -1))
            actual = _combine_eagle3_aux(model, states)
            assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
            for value, original in zip(states, originals):
                assert torch.equal(
                    value.contiguous().view(torch.uint8), original.view(torch.uint8)
                )


@torch.inference_mode()
def test_eagle3_aux_combination_fallbacks(eagle3_fc_norm_model):
    from vllm import ir
    from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM

    class OffsetHead(Eagle3LlamaForCausalLM):
        def combine_hidden_states(self, hidden_states):
            return super().combine_hidden_states(hidden_states) + 1

    for options in (
        {"fc_norm": False},
        {"norm_before_fc": True},
        {"model_cls": OffsetHead},
        {"use_aux": False},
        {},
    ):
        model = eagle3_fc_norm_model(dtype=torch.float32, **options)
        # Mixed input dtypes must be promoted by cat before normalization.
        states = [
            torch.randn(17, 64, device=DEVICE_TYPE, dtype=dtype)
            for dtype in (
                (torch.float32,) * 3
                if options
                else (torch.float16, torch.bfloat16, torch.float32)
            )
        ]
        with ir.ops.rms_norm.set_priority(["native"]):
            expected = model.combine_hidden_states(torch.cat(states, -1))
            actual = _combine_eagle3_aux(model, states)
            assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.parametrize("priority", [["vllm_c", "native"], ["native"]])
@torch.inference_mode()
def test_eagle3_fc_norm_avoids_raw_aux_allocation(eagle3_fc_norm_model, priority):
    """Concatenate normalized states once, without a raw-state copy."""
    from vllm import ir

    model = eagle3_fc_norm_model()
    states = [
        torch.randn(4096, 64, device=DEVICE_TYPE, dtype=torch.bfloat16)
        for _ in range(3)
    ]
    concatenations = []
    with ir.ops.rms_norm.set_priority(priority):
        for combine in (
            lambda: model.combine_hidden_states(torch.cat(states, -1)),
            lambda: _combine_eagle3_aux(model, states),
        ):
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU]
            ) as profile:
                output = combine()
                if output.device.type != "cpu":
                    torch.accelerator.synchronize()
            cat = next(
                event for event in profile.key_averages() if event.key == "aten::cat"
            )
            concatenations.append(cat.count)
            del output
    assert concatenations == [2, 1], (priority, concatenations)
