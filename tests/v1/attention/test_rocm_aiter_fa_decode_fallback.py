# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch

MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "vllm"
    / "v1"
    / "attention"
    / "backends"
    / "rocm_aiter_fa.py"
)


def make_module(name: str, *, package: bool = False, **attrs) -> ModuleType:
    module = ModuleType(name)
    if package:
        module.__path__ = []
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


class StubAttentionMetadataBuilder:
    @classmethod
    def __class_getitem__(cls, _item):
        return cls


class StubAttentionType:
    DECODER = object()
    ENCODER_DECODER = object()


@contextmanager
def loaded_backend_module(unified_calls: list[dict[str, object]]):
    def unified_attention(**kwargs):
        unified_calls.append(kwargs)

    modules = {
        "vllm": make_module("vllm", package=True),
        "vllm._aiter_ops": make_module(
            "vllm._aiter_ops",
            rocm_aiter_ops=SimpleNamespace(is_shuffle_kv_cache_enabled=lambda: False),
        ),
        "vllm.config": make_module(
            "vllm.config",
            VllmConfig=object,
            get_layers_from_vllm_config=lambda *args, **kwargs: {},
        ),
        "vllm.config.cache": make_module("vllm.config.cache", CacheDType=object),
        "vllm.logger": make_module(
            "vllm.logger",
            init_logger=lambda _name: SimpleNamespace(
                info=lambda *args, **kwargs: None
            ),
        ),
        "vllm.model_executor": make_module("vllm.model_executor", package=True),
        "vllm.model_executor.layers": make_module(
            "vllm.model_executor.layers",
            package=True,
        ),
        "vllm.model_executor.layers.attention": make_module(
            "vllm.model_executor.layers.attention",
            Attention=object,
        ),
        "vllm.platforms": make_module(
            "vllm.platforms",
            current_platform=SimpleNamespace(
                is_rocm=lambda: False,
                fp8_dtype=lambda: torch.float8_e4m3fn,
            ),
        ),
        "vllm.platforms.interface": make_module(
            "vllm.platforms.interface",
            DeviceCapability=object,
        ),
        "vllm.utils": make_module("vllm.utils", package=True),
        "vllm.utils.math_utils": make_module(
            "vllm.utils.math_utils",
            cdiv=lambda a, b: (a + b - 1) // b,
        ),
        "vllm.utils.platform_utils": make_module(
            "vllm.utils.platform_utils",
            num_compute_units=lambda: 1,
        ),
        "vllm.utils.torch_utils": make_module(
            "vllm.utils.torch_utils",
            is_quantized_kv_cache=lambda _dtype: False,
        ),
        "vllm.v1": make_module("vllm.v1", package=True),
        "vllm.v1.attention": make_module("vllm.v1.attention", package=True),
        "vllm.v1.attention.backend": make_module(
            "vllm.v1.attention.backend",
            AttentionBackend=object,
            AttentionCGSupport=SimpleNamespace(UNIFORM_BATCH="UNIFORM_BATCH"),
            AttentionImpl=object,
            AttentionLayer=object,
            AttentionMetadataBuilder=StubAttentionMetadataBuilder,
            AttentionType=StubAttentionType,
            CommonAttentionMetadata=object,
            MultipleOf=object,
        ),
        "vllm.v1.attention.backends": make_module(
            "vllm.v1.attention.backends",
            package=True,
        ),
        "vllm.v1.attention.backends.utils": make_module(
            "vllm.v1.attention.backends.utils",
            split_decodes_prefills_and_extends=lambda *args, **kwargs: (
                0,
                0,
                0,
                0,
                0,
                0,
            ),
        ),
        "vllm.v1.attention.ops": make_module(
            "vllm.v1.attention.ops",
            package=True,
        ),
        "vllm.v1.attention.ops.merge_attn_states": make_module(
            "vllm.v1.attention.ops.merge_attn_states",
            merge_attn_states=lambda **kwargs: None,
        ),
        "vllm.v1.kv_cache_interface": make_module(
            "vllm.v1.kv_cache_interface",
            AttentionSpec=object,
        ),
        "aiter": make_module("aiter", package=True),
        "aiter.ops": make_module("aiter.ops", package=True),
        "aiter.ops.triton": make_module("aiter.ops.triton", package=True),
        "aiter.ops.triton.unified_attention": make_module(
            "aiter.ops.triton.unified_attention",
            unified_attention=unified_attention,
        ),
    }

    spec = spec_from_file_location("rocm_aiter_fa_under_test", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    with patch.dict("sys.modules", modules):
        spec.loader.exec_module(module)
        yield module


def build_decode_case(
    module: ModuleType, *, head_size: int, sliding_window: int | None
):
    impl = module.AiterFlashAttentionImpl(
        num_heads=1,
        head_size=head_size,
        scale=1.0,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=sliding_window,
        kv_cache_dtype="auto",
    )
    layer = SimpleNamespace(
        _k_scale=torch.ones(1, 1),
        _v_scale=torch.ones(1, 1),
    )
    metadata = SimpleNamespace(
        num_actual_tokens=1,
        num_actual_kv_tokens=1,
        num_decodes=1,
        num_decode_tokens=1,
        num_prefills=0,
        num_prefill_tokens=0,
        num_extends=0,
        num_extend_tokens=0,
        use_cascade=False,
        decode_metadata=SimpleNamespace(max_query_len=1),
        prefill_metadata=None,
        extend_metadata=None,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([8], dtype=torch.int32),
        max_seq_len=8,
        block_table=torch.tensor([[0]], dtype=torch.int32),
        slot_mapping=torch.tensor([0], dtype=torch.int32),
        k_scale=None,
        v_scale=None,
    )
    query = torch.randn(1, 1, head_size)
    key = torch.randn(1, 1, head_size)
    value = torch.randn(1, 1, head_size)
    kv_cache = torch.randn(2, 1, 16, 1, head_size)
    output = torch.zeros_like(query)
    return impl, layer, metadata, query, key, value, kv_cache, output


@pytest.mark.parametrize(
    ("head_size", "sliding_window", "expect_unified"),
    [
        (32, None, True),
        (128, 256, True),
        (128, None, False),
    ],
    ids=[
        "small-head-uses-unified",
        "sliding-window-uses-unified",
        "large-head-without-window-keeps-paged-attention",
    ],
)
def test_decode_fallback_branch_selection(
    head_size: int,
    sliding_window: int | None,
    expect_unified: bool,
):
    unified_calls: list[dict[str, object]] = []
    paged_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    with loaded_backend_module(unified_calls) as module:
        (
            impl,
            layer,
            metadata,
            query,
            key,
            value,
            kv_cache,
            output,
        ) = build_decode_case(
            module,
            head_size=head_size,
            sliding_window=sliding_window,
        )

        def paged_attention_v1(*args, **kwargs):
            paged_calls.append((args, kwargs))

        with patch.object(
            torch.ops,
            "aiter",
            SimpleNamespace(paged_attention_v1=paged_attention_v1),
            create=True,
        ):
            impl.forward(layer, query, key, value, kv_cache, metadata, output)

    assert bool(unified_calls) is expect_unified
    assert bool(paged_calls) is (not expect_unified)

    if expect_unified:
        expected_window = (
            (-1, -1) if sliding_window is None else (sliding_window - 1, 0)
        )
        assert unified_calls[0]["max_seqlen_q"] == 1
        assert unified_calls[0]["window_size"] == expected_window
    else:
        assert paged_calls[0][0][-1] == 0
