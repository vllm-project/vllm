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
            config_options={
                "kernel_config": KernelConfig(
                    enable_rocm_segmented_attn_autotune=True,
                )
            }
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


@pytest.mark.parametrize("tower", [False, True])
def test_request_adapter_name_and_path_isolate_all_encoder_cache_keys(
    tmp_path, monkeypatch, tower
):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import (
        CLIPImageProcessor,
        LlamaConfig,
        LlavaConfig,
        LlavaProcessor,
        TokenizersBackend,
    )

    from vllm.config import ModelConfig, VllmConfig
    from vllm.config.ec_transfer import ECTransferConfig
    from vllm.config.lora import LoRAConfig
    from vllm.inputs import mm_input
    from vllm.lora.request import LoRARequest
    from vllm.multimodal.inputs import MultiModalKwargsItems, PlaceholderRange
    from vllm.sampling_params import SamplingParams
    from vllm.v1.engine.input_processor import InputProcessor

    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    model_path = tmp_path / "model"
    text = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=4,
    )
    LlavaConfig(
        architectures=["LlavaForConditionalGeneration"],
        text_config=text.to_dict(),
        vision_config={
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "image_size": 8,
            "patch_size": 4,
        },
        image_token_index=3,
    ).save_pretrained(model_path)
    tokenizer = TokenizersBackend(
        tokenizer_object=Tokenizer(
            WordLevel(
                {"<unk>": 0, "<s>": 1, "</s>": 2, "<image>": 3}, unk_token="<unk>"
            )
        ),
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        additional_special_tokens=["<image>"],
    )
    LlavaProcessor(
        CLIPImageProcessor(
            size={"shortest_edge": 8}, crop_size={"height": 8, "width": 8}
        ),
        tokenizer,
        patch_size=4,
        vision_feature_select_strategy="default",
    ).save_pretrained(model_path)
    config = VllmConfig(
        model_config=ModelConfig(
            model=str(model_path),
            dtype="float16",
            skip_tokenizer_init=False,
            enforce_eager=True,
            max_model_len=128,
        ),
        ec_transfer_config=ECTransferConfig(
            ec_connector="ECTieringConnector", ec_role="ec_both"
        ),
        lora_config=LoRAConfig(enable_tower_connector_lora=tower, max_lora_rank=8),
    )
    processor = InputProcessor(config)

    from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler import embedding_cache
    from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
    from vllm.v1.request import Request

    def process(name, path, adapter_id):
        result = processor.process_inputs(
            f"request-{adapter_id}",
            mm_input(
                [3] * 4,
                MultiModalKwargsItems({"image": [None]}),
                {"image": ["cat"]},
                {"image": [PlaceholderRange(offset=0, length=4)]},
            ),
            SamplingParams(max_tokens=1),
            ("generate",),
            lora_request=LoRARequest(name, adapter_id, path),
        )
        return Request.from_engine_core_request(result, None)

    first = process("vision", "/models/v1", 1)
    different_path = process("vision", "/models/v2", 2)
    different_name = process("other", "/models/v1", 3)
    repeated = process("vision", "/models/v1", 4)
    identifier = first.mm_features[0].identifier
    assert identifier == repeated.mm_features[0].identifier

    gpu_cache = EncoderCacheManager(cache_size=16)
    gpu_cache.allocate(first, 0)
    cpu_cache = embedding_cache.EmbeddingCache(num_blocks=4)
    cpu_cache.alloc(identifier, 1, (4, 1))
    cpu_cache.mark_ready(identifier)
    key = build_ec_cache_key(_namespace().digest(), identifier, "image", 4, 16)
    for request in (different_path, different_name):
        other = request.mm_features[0].identifier
        assert (other != identifier) == tower
        assert gpu_cache.check_and_update_cache(request, 0) == (not tower)
        assert (cpu_cache.get(other) is not None) == (not tower)
        assert (
            build_ec_cache_key(_namespace().digest(), other, "image", 4, 16) != key
        ) == tower
    assert gpu_cache.check_and_update_cache(repeated, 0)
    assert cpu_cache.get(repeated.mm_features[0].identifier) is not None


def test_namespace_isolates_encoder_lora_config_without_reading_weights(
    tmp_path, model_namespace
):
    from transformers import LlavaConfig

    from vllm.config.lora import LoRAConfig

    LlavaConfig(architectures=["LlavaForConditionalGeneration"]).save_pretrained(
        tmp_path
    )

    def namespace(dtype, **options):
        options.setdefault(
            "default_mm_loras", {"image": str(tmp_path / "absent-adapter")}
        )
        return model_namespace(
            config_options={
                "lora_config": LoRAConfig(
                    enable_tower_connector_lora=True,
                    lora_dtype=dtype,
                    **options,
                )
            },
        ).digest()

    first = namespace("float16")
    assert first == namespace("float16")
    assert first != namespace("bfloat16")
    assert first == namespace("float16", max_cpu_loras=4)
    assert first == namespace(
        "float16", default_mm_loras={"image": str(tmp_path / "another-adapter")}
    )
    assert first != namespace("float16", target_modules=["q_proj"])
