# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers import PretrainedConfig

from vllm.config.ec_transfer import ECRole, ECTransferConfig
from vllm.config.model import ModelConfig
from vllm.config.multimodal import MultiModalConfig
from vllm.config.vllm import VllmConfig
from vllm.transformers_utils.model_arch_config_convertor import (
    ModelArchConfigConvertorBase,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum


def test_mm_encoder_attn_backend_str_conversion():
    config = MultiModalConfig(mm_encoder_attn_backend="FLASH_ATTN")  # type: ignore[arg-type]
    assert config.mm_encoder_attn_backend == AttentionBackendEnum.FLASH_ATTN


def test_mm_encoder_attn_backend_invalid():
    with pytest.raises(ValueError):
        MultiModalConfig(mm_encoder_attn_backend="not_a_backend")  # type: ignore[arg-type]


def test_mm_hasher_algorithm_invalid():
    with pytest.raises(ValueError, match="mm_hasher_algorithm"):
        MultiModalConfig(mm_hasher_algorithm="md5")  # type: ignore[arg-type]


def test_mm_encoder_attn_backend_hash_updates():
    base_hash = MultiModalConfig().compute_hash()
    overridden_hash = MultiModalConfig(
        mm_encoder_attn_backend=AttentionBackendEnum.FLASH_ATTN
    ).compute_hash()
    assert base_hash != overridden_hash


def test_language_model_only_does_not_affect_mm_hash():
    """language_model_only does not affect the ViT computation graph,
    so it should not change the multimodal config hash."""
    base_hash = MultiModalConfig().compute_hash()
    lm_only_hash = MultiModalConfig(language_model_only=True).compute_hash()
    assert base_hash == lm_only_hash


def test_language_model_only_affects_model_hash():
    """language_model_only affects the LM computation graph,
    so it should change the model config hash."""
    model = "llava-hf/llava-1.5-7b-hf"
    base_hash = ModelConfig(model).compute_hash()
    lm_only_hash = ModelConfig(model, language_model_only=True).compute_hash()
    assert base_hash != lm_only_hash


def test_zero_mm_limits_affect_model_hash():
    """Zero per-modality limits disable the MM input path (the model is
    called with input_ids instead of inputs_embeds), so they must change
    the model config hash to avoid reusing incompatible compiled artifacts."""
    model = "llava-hf/llava-1.5-7b-hf"
    base_hash = ModelConfig(model).compute_hash()
    disabled_hash = ModelConfig(model, limit_mm_per_prompt={"image": 0}).compute_hash()
    assert base_hash != disabled_hash


def test_nonzero_mm_limits_do_not_affect_model_hash():
    """Nonzero limits only cap the number of items per prompt; they do not
    change the computation graph, so they should not invalidate caches."""
    model = "llava-hf/llava-1.5-7b-hf"
    base_hash = ModelConfig(model).compute_hash()
    capped_hash = ModelConfig(model, limit_mm_per_prompt={"image": 5}).compute_hash()
    assert base_hash == capped_hash


def test_enable_mm_embeds_affects_model_hash():
    """enable_mm_embeds keeps the inputs_embeds path alive even when all
    modality limits are zero, so it is part of the same graph predicate."""
    model = "llava-hf/llava-1.5-7b-hf"
    base_hash = ModelConfig(model, limit_mm_per_prompt={"image": 0}).compute_hash()
    embeds_hash = ModelConfig(
        model, limit_mm_per_prompt={"image": 0}, enable_mm_embeds=True
    ).compute_hash()
    assert base_hash != embeds_hash


@pytest.mark.parametrize("backend_arg", ["video_backend", "backend"])
def test_use_gpu_video_backend_from_media_io_kwargs(backend_arg: str):
    config = MultiModalConfig(
        media_io_kwargs={"video": {backend_arg: "pynvvideocodec"}}
    )

    assert config.use_gpu_video_backend()


def test_mm_encoder_fp8_scale_path_requires_fp8():
    with pytest.raises(ValueError, match="mm_encoder_attn_dtype"):
        MultiModalConfig(mm_encoder_fp8_scale_path="/tmp/scales.json")


def test_mm_encoder_attn_dtype_hash_updates(tmp_path):
    scale_file = tmp_path / "scales.json"
    scale_file.write_text("{}")
    base_hash = MultiModalConfig().compute_hash()
    fp8_hash = MultiModalConfig(mm_encoder_attn_dtype="fp8").compute_hash()
    fp8_static_hash = MultiModalConfig(
        mm_encoder_attn_dtype="fp8",
        mm_encoder_fp8_scale_path=str(scale_file),
    ).compute_hash()
    assert base_hash != fp8_hash
    assert fp8_hash != fp8_static_hash


def _make_mm_prefix_model_config(
    *,
    language_model_only: bool = False,
) -> ModelConfig:
    model_config = MagicMock(spec=ModelConfig)
    model_config.multimodal_config = MultiModalConfig(
        language_model_only=language_model_only
    )
    # Bind real helper methods onto the mock.
    model_config._supports_multimodal_for_mm_prefix = (
        ModelConfig._supports_multimodal_for_mm_prefix.__get__(
            model_config, ModelConfig
        )
    )
    return model_config


@pytest.mark.parametrize("supports_mm", [True, False])
def test_supports_multimodal_for_mm_prefix_uses_registry(supports_mm: bool):
    model_config = _make_mm_prefix_model_config()

    with patch(
        "vllm.multimodal.MULTIMODAL_REGISTRY.supports_multimodal_inputs",
        return_value=supports_mm,
    ) as mocked:
        assert model_config._supports_multimodal_for_mm_prefix() is supports_mm
        mocked.assert_called_once_with(model_config)

    # Sticky cache — registry must not be consulted again.
    with patch(
        "vllm.multimodal.MULTIMODAL_REGISTRY.supports_multimodal_inputs",
        side_effect=AssertionError("should use cache"),
    ):
        assert model_config._supports_multimodal_for_mm_prefix() is supports_mm


def test_supports_multimodal_for_mm_prefix_before_multimodal_config():
    model_config = _make_mm_prefix_model_config()
    model_config.multimodal_config = None

    assert model_config._supports_multimodal_for_mm_prefix() is True
    assert not hasattr(model_config, "_supports_multimodal_inputs_cached")


def test_language_model_only_disables_via_supports_multimodal_inputs():
    """language_model_only zeros all limits, so registry reports text-only."""
    model_config = _make_mm_prefix_model_config(language_model_only=True)

    with patch(
        "vllm.multimodal.MULTIMODAL_REGISTRY.supports_multimodal_inputs",
        return_value=False,
    ):
        assert model_config._supports_multimodal_for_mm_prefix() is False


def test_convertor_clears_mm_prefix_when_multimodal_disabled():
    hf_config = PretrainedConfig(
        model_type="gemma3",
        architectures=["Gemma3ForConditionalGeneration"],
    )
    hf_config.is_mm_prefix_lm = True
    convertor = ModelArchConfigConvertorBase(hf_config, hf_config)

    assert convertor.is_mm_prefix_lm(supports_multimodal=True) is True
    assert convertor.is_mm_prefix_lm(supports_multimodal=False) is False

    enabled = convertor.convert(supports_multimodal=True)
    disabled = convertor.convert(supports_multimodal=False)
    assert enabled.is_mm_prefix_lm is True
    assert disabled.is_mm_prefix_lm is False


def test_sticky_cache_survives_text_subconfig_regeneration():
    """with_hf_config deepcopies the cached decision onto text submodules."""
    model_config = _make_mm_prefix_model_config()
    with patch(
        "vllm.multimodal.MULTIMODAL_REGISTRY.supports_multimodal_inputs",
        return_value=False,
    ):
        assert model_config._supports_multimodal_for_mm_prefix() is False

    # Simulate deepcopy onto a Gemma4ForCausalLM-like config that would
    # otherwise fail registry lookup / return False incorrectly.
    text_config = _make_mm_prefix_model_config()
    text_config._supports_multimodal_inputs_cached = (
        model_config._supports_multimodal_inputs_cached
    )
    with patch(
        "vllm.multimodal.MULTIMODAL_REGISTRY.supports_multimodal_inputs",
        side_effect=AssertionError("must not re-query registry"),
    ):
        assert text_config._supports_multimodal_for_mm_prefix() is False


@pytest.mark.parametrize(
    ("device", "expected"),
    [
        (None, None),
        ("cpu", "cpu"),
        ("cuda", "cuda"),
        # Callers compare against a bare device type, so an indexed device and a
        # torch.device must normalise to the same thing as "cuda".
        ("cuda:1", "cuda"),
        (torch.device("cuda", 1), "cuda"),
    ],
)
def test_mm_processor_device_type_normalizes(device: object, expected: str | None):
    kwargs = {} if device is None else {"device": device}
    config = MultiModalConfig(mm_processor_kwargs=kwargs)
    assert config.get_mm_processor_device_type() == expected


def _validate_mm_processor_device(*, device: str, ec_role: ECRole | None) -> None:
    ec_config = (
        None
        if ec_role is None
        # `is_ec_producer`/`is_ec_consumer` are False without a connector, so an
        # unset one would make every role look like "no EC role at all".
        else ECTransferConfig(ec_connector="ECExampleConnector", ec_role=ec_role)
    )
    MultiModalConfig(
        mm_processor_kwargs={"device": device}
    ).validate_mm_processor_device(ec_config)


@pytest.mark.parametrize("ec_role", [None, "ec_producer"])
def test_bad_mm_processor_device_rejected(ec_role: ECRole | None):
    """A bad device must fail during startup, not mid-request.

    Rejected for every role, and on a CPU-only platform too, so a typo can never
    silently fall through to running the processor somewhere unintended.
    """
    with (
        patch("vllm.platforms.current_platform.device_type", "cpu"),
        pytest.raises(ValueError, match='Invalid "device" in mm_processor_kwargs'),
    ):
        _validate_mm_processor_device(device="not-a-device", ec_role=ec_role)


@pytest.mark.parametrize("ec_role", [None, "ec_consumer", "ec_both"])
def test_accelerator_mm_processor_rejected_outside_encoder_instance(
    ec_role: ECRole | None,
):
    """Only an encode-only EPD instance has the device to itself.

    Every other role runs the language model in the same process, so frontend
    accelerator work would contend with the forward pass and allocate outside
    the memory profiled for the KV cache.
    """
    with (
        patch("vllm.platforms.current_platform.device_type", "cuda"),
        pytest.raises(ValueError, match="also runs the language model"),
    ):
        _validate_mm_processor_device(device="cuda", ec_role=ec_role)


def test_accelerator_mm_processor_allowed_on_encoder_instance():
    with patch("vllm.platforms.current_platform.device_type", "cuda"):
        _validate_mm_processor_device(device="cuda", ec_role="ec_producer")


def test_cpu_mm_processor_needs_no_ec_role():
    """The gate only applies to the accelerator, so CPU stays unrestricted."""
    with patch("vllm.platforms.current_platform.device_type", "cuda"):
        _validate_mm_processor_device(device="cpu", ec_role=None)


@pytest.mark.parametrize(
    ("flag", "kwargs", "expected"),
    [
        # "auto" is only settled once the EC role is known, so it must leave no
        # device behind for `VllmConfig` to mistake for an explicit request.
        ("auto", None, None),
        (None, None, None),
        ("cpu", None, "cpu"),
        ("cuda", None, "cuda"),
        # Any explicit value other than "cpu" means "whatever this platform
        # calls its accelerator".
        ("gpu", None, "cuda"),
        # An explicit device in the kwargs wins over the convenience flag.
        ("cpu", {"device": "cuda"}, "cuda"),
        ("cuda", {"device": "cpu"}, "cpu"),
    ],
)
def test_fold_mm_processor_device(
    flag: str | None, kwargs: dict | None, expected: str | None
):
    with patch("vllm.platforms.current_platform.device_type", "cuda"):
        folded = MultiModalConfig.fold_mm_processor_device(kwargs, flag)

    config = MultiModalConfig(mm_processor_kwargs=folded or {})
    assert config.get_mm_processor_device_type() == expected


def _resolve_mm_processor_device(
    *,
    ec_role: ECRole | None,
    mm_tensor_ipc: str = "torch_shm",
    device: str | None = None,
) -> str | None:
    """Run the `auto` resolution and report where the processor ended up."""
    mm_config = MultiModalConfig(
        mm_processor_kwargs={} if device is None else {"device": device},
        mm_tensor_ipc=mm_tensor_ipc,  # type: ignore[arg-type]
    )
    model_config = MagicMock(spec=ModelConfig)
    model_config.multimodal_config = mm_config
    vllm_config = MagicMock(spec=VllmConfig)
    vllm_config.model_config = model_config
    vllm_config.ec_transfer_config = (
        None
        if ec_role is None
        else ECTransferConfig(ec_connector="ECExampleConnector", ec_role=ec_role)
    )

    with patch("vllm.platforms.current_platform.device_type", "cuda"):
        VllmConfig._resolve_mm_processor_device(vllm_config)
    return mm_config.get_mm_processor_device_type()


def test_auto_mm_processor_device_uses_accelerator_on_encoder_instance():
    """The one deployment that has the device to itself and can hand it over."""
    assert _resolve_mm_processor_device(ec_role="ec_producer") == "cuda"


@pytest.mark.parametrize("ec_role", [None, "ec_consumer", "ec_both"])
def test_auto_mm_processor_device_stays_on_cpu_off_encoder_instance(
    ec_role: ECRole | None,
):
    """Every other role runs the language model in the same process."""
    assert _resolve_mm_processor_device(ec_role=ec_role) is None


def test_auto_mm_processor_device_needs_a_device_capable_transport():
    """Other transports serialize host bytes, so the output is copied back."""
    assert (
        _resolve_mm_processor_device(ec_role="ec_producer", mm_tensor_ipc="direct_rpc")
        is None
    )


def test_auto_mm_processor_device_leaves_an_explicit_request_alone():
    assert _resolve_mm_processor_device(ec_role="ec_producer", device="cpu") == "cpu"


def test_vllm_config_runs_the_mm_processor_device_check():
    """Startup must reach the check; the rule itself is covered above.

    Guards the wiring, which no other test would notice going missing.
    """
    model_config = MagicMock(spec=ModelConfig)
    model_config.multimodal_config = MultiModalConfig(
        mm_processor_kwargs={"device": "cuda"}
    )
    vllm_config = MagicMock(spec=VllmConfig)
    vllm_config.model_config = model_config
    vllm_config.ec_transfer_config = None

    with (
        patch("vllm.platforms.current_platform.device_type", "cuda"),
        pytest.raises(ValueError, match="also runs the language model"),
    ):
        VllmConfig._validate_mm_processor_device(vllm_config)
