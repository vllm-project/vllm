# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from pathlib import Path

import pytest

from vllm.distributed.ec_transfer.ec_connector.tiering.layout import (
    ECCacheNamespace,
    build_ec_cache_key,
    get_ec_entry_path,
    get_ec_namespace_path,
    get_ec_temporary_path,
)

pytestmark = pytest.mark.cpu_test


def _namespace() -> ECCacheNamespace:
    return ECCacheNamespace(
        encoder_config_hash="a" * 64,
    )


def test_namespace_is_deterministic_and_compatibility_scoped() -> None:
    namespace = _namespace()

    assert namespace.digest() == _namespace().digest()
    assert (
        namespace.digest() != replace(namespace, encoder_config_hash="b" * 64).digest()
    )


@pytest.fixture
def model_namespace(tmp_path, monkeypatch):
    from vllm.config import ModelConfig, VllmConfig
    from vllm.config.ec_transfer import ECTransferConfig
    from vllm.distributed.ec_transfer.ec_connector.tiering.scheduler import (
        ECTieringScheduler,
    )

    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")

    def namespace(*, dtype="float16", output_width=32, config_options=None, **options):
        model = ModelConfig(
            model=str(tmp_path),
            dtype=dtype,
            skip_tokenizer_init=True,
            enforce_eager=True,
            max_model_len=128,
            **options,
        )
        ec = ECTransferConfig(ec_connector="ECTieringConnector", ec_role="ec_both")
        ec.mm_encoder_output_widths = {"image": output_width}
        config = VllmConfig(
            model_config=model, ec_transfer_config=ec, **(config_options or {})
        )
        return ECTieringScheduler._build_namespace(config)

    return namespace


def test_namespace_uses_effective_hf_encoder_config(tmp_path, model_namespace):
    from transformers import LlavaConfig

    LlavaConfig(architectures=["LlavaForConditionalGeneration"]).save_pretrained(
        tmp_path
    )

    def namespace(overrides):
        return model_namespace(hf_overrides=overrides)

    def select_last_layer(config):
        config.vision_feature_layer = -1
        return config

    previous = namespace({"vision_feature_layer": -2})
    current = namespace({"vision_feature_layer": -1})
    assert previous.digest() != current.digest()
    assert current.digest() == namespace({"vision_feature_layer": -1}).digest()
    assert current.digest() == namespace(select_last_layer).digest()
    nested = namespace(
        {"vision_feature_layer": -1, "vision_config": {"hidden_act": "relu"}}
    )
    assert nested.digest() != current.digest()
    assert previous.digest() != model_namespace(dtype="bfloat16").digest()
    assert previous.digest() != model_namespace(revision="revision-2").digest()
    assert (
        previous.digest()
        == model_namespace(
            hf_overrides={"vision_feature_layer": -2}, output_width=64
        ).digest()
    )
    assert (
        previous.digest() != namespace({"text_config": {"hidden_size": 8192}}).digest()
    )
    assert build_ec_cache_key(previous.digest(), "image", "image", 2, 128) != (
        build_ec_cache_key(current.digest(), "image", "image", 2, 128)
    )


@pytest.mark.parametrize(
    ("config_class", "architecture"),
    [
        ("LlavaConfig", "LlavaForConditionalGeneration"),
        ("Qwen2_5_VLConfig", "Qwen2_5_VLForConditionalGeneration"),
    ],
)
def test_namespace_isolates_encoder_precision(
    tmp_path, model_namespace, config_class, architecture
):
    import transformers

    from vllm.config.quantization import resolve_quantization_config

    getattr(transformers, config_class)(architectures=[architecture]).save_pretrained(
        tmp_path
    )
    base = model_namespace().digest()
    fp8_options = {"mm_encoder_attn_dtype": "fp8"}
    fp8 = model_namespace(**fp8_options).digest()
    assert fp8 != base
    assert fp8 == model_namespace(**fp8_options).digest()
    assert (
        fp8
        != model_namespace(
            **fp8_options, mm_encoder_fp8_scale_save_path=str(tmp_path / "saved.json")
        ).digest()
    )
    scales = tmp_path / "scales.json"
    scales.write_text("{}")
    assert (
        fp8
        != model_namespace(
            **fp8_options, mm_encoder_fp8_scale_path=str(scales)
        ).digest()
    )
    quantization = "fp8_per_tensor"
    quantized = model_namespace(quantization=quantization).digest()
    assert quantized != base
    assert (
        quantized
        == model_namespace(
            quantization=quantization,
            quantization_config=resolve_quantization_config(quantization, None),
        ).digest()
    )
    assert (
        quantized
        != model_namespace(
            quantization=quantization,
            quantization_config=resolve_quantization_config(
                quantization, {"ignore": ["*visual*"]}
            ),
        ).digest()
    )


def test_namespace_isolates_processing_and_kernel_config(tmp_path, model_namespace):
    from transformers import Qwen3VLConfig

    from vllm.config import KernelConfig, ParallelConfig

    Qwen3VLConfig(architectures=["Qwen3VLForConditionalGeneration"]).save_pretrained(
        tmp_path
    )
    base = model_namespace().digest()
    assert base == model_namespace().digest()
    for options in (
        {"mm_processor_kwargs": {"image_mean": [0.0, 0.0, 0.0]}},
        {"mm_device_do_normalize": False},
        {"video_pruning_rate": 0.5},
    ):
        assert base != model_namespace(**options).digest()
    assert (
        model_namespace(video_pruning_rate=0.5).digest()
        != model_namespace(
            video_pruning_rate=0.5, video_pruning_method="vidcom2"
        ).digest()
    )
    assert (
        base
        != model_namespace(
            config_options={"kernel_config": KernelConfig(linear_backend="torch")}
        ).digest()
    )
    assert (
        base
        == model_namespace(
            config_options={"parallel_config": ParallelConfig(tensor_parallel_size=2)}
        ).digest()
    )


def test_cache_key_includes_item_shape_and_identity() -> None:
    namespace_digest = _namespace().digest()
    cache_key = build_ec_cache_key(
        namespace_digest,
        identifier="image-hash",
        modality="image",
        num_tokens=16,
        file_size_bytes=16 * 3584 * 2,
    )

    assert cache_key == build_ec_cache_key(
        namespace_digest,
        identifier="image-hash",
        modality="image",
        num_tokens=16,
        file_size_bytes=16 * 3584 * 2,
    )
    assert cache_key != build_ec_cache_key(
        namespace_digest,
        identifier="lora:image-hash",
        modality="image",
        num_tokens=16,
        file_size_bytes=16 * 3584 * 2,
    )
    assert cache_key != build_ec_cache_key(
        namespace_digest,
        identifier="image-hash",
        modality="image",
        num_tokens=16,
        file_size_bytes=16 * 3584 * 2 + 4096,
    )


def test_entry_path_is_versioned_and_hash_prefix_sharded() -> None:
    namespace_digest = _namespace().digest()
    cache_key = "a1b2" + "0" * 60

    path = get_ec_entry_path(Path("/cache"), namespace_digest, cache_key)

    assert path == (
        get_ec_namespace_path(Path("/cache"), namespace_digest)
        / "a1b"
        / "20"
        / f"{cache_key}.ec"
    )
    temporary_path = get_ec_temporary_path(path, "write-operation-1", "0" * 32)
    assert temporary_path.parent == (
        get_ec_namespace_path(Path("/cache"), namespace_digest) / ".tmp" / ("0" * 32)
    )
    assert temporary_path.name.startswith(f".{path.name}.{'0' * 32}.")
    assert temporary_path.suffix == ".tmp"
