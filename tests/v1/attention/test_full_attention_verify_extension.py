# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contract tests for the out-of-tree full-attention verify extension."""

from dataclasses import FrozenInstanceError, dataclass, replace

import pytest
import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config import SpeculativeConfig
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
)
from vllm.v1.attention.verify_extension import (
    FullAttentionVerifyBackend,
    FullAttentionVerifyExtension,
    FullAttentionVerifyMetadata,
    FullAttentionVerifyRows,
    build_full_attention_verify_metadata,
    dispatch_full_attention_verify,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.spec_decode import registry
from vllm.v1.spec_decode.registry import (
    SpeculativeMethodSpec,
    register_speculative_method,
)
from vllm.v1.worker.utils import AttentionGroup

DEVICE = torch.device("cpu")
BLOCK_SIZE = 16
NAME = "unit_test_full_attention_method"
FACTORY = f"{__name__}:make_extension"
INVALID_FACTORY = f"{__name__}:make_invalid_extension"
NONE_FACTORY = f"{__name__}:make_no_extension"
SPECULATOR = f"{__name__}:DummySpeculator"
created: list["RecordingExtension"] = []


class DummySpeculator:
    pass


class RecordingExtension(FullAttentionVerifyExtension):
    def __init__(self, vllm_config):
        self.vllm_config = vllm_config
        self.rows: list[FullAttentionVerifyRows] = []

    def supports_backend(self, backend: FullAttentionVerifyBackend) -> bool:
        return (
            backend.name == FakeFullAttentionBackend.get_name()
            and backend.module == FakeFullAttentionBackend.__module__
            and backend.qualname == FakeFullAttentionBackend.__qualname__
        )

    def forward(self, rows: FullAttentionVerifyRows) -> None:
        self.rows.append(rows)
        rows.output.fill_(7)


def make_extension(vllm_config):
    extension = RecordingExtension(vllm_config)
    created.append(extension)
    return extension


def make_invalid_extension(vllm_config):
    return object()


def make_no_extension(vllm_config):
    return None


@dataclass(frozen=True)
class RecordingMetadata:
    verify_extension: FullAttentionVerifyExtension
    verify: FullAttentionVerifyMetadata


class RecordingMetadataBuilder(AttentionMetadataBuilder[RecordingMetadata]):
    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.verify_extension: FullAttentionVerifyExtension | None = None
        self.verify_backend: FullAttentionVerifyBackend | None = None

    def set_full_attention_verify_extension(
        self,
        extension: FullAttentionVerifyExtension,
        backend: FullAttentionVerifyBackend,
    ) -> None:
        self.verify_extension = extension
        self.verify_backend = backend

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> RecordingMetadata:
        assert self.verify_backend is not None
        assert self.verify_extension is not None
        return RecordingMetadata(
            verify_extension=self.verify_extension,
            verify=build_full_attention_verify_metadata(
                self.verify_backend, common_attn_metadata
            ),
        )


class FakeFullAttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "FAKE_FULL_ATTN"

    @staticmethod
    def get_impl_cls() -> type[AttentionImpl]:
        return AttentionImpl

    @staticmethod
    def get_builder_cls() -> type[RecordingMetadataBuilder]:
        return RecordingMetadataBuilder

    @classmethod
    def supports_full_attention_verify_extension(cls) -> bool:
        return True


class UnwiredBackend(FakeFullAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "UNWIRED"

    @classmethod
    def supports_full_attention_verify_extension(cls) -> bool:
        return False


class UnsupportedCapableBackend(FakeFullAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "UNSUPPORTED_CAPABLE"


@pytest.fixture(autouse=True)
def clear_created():
    created.clear()


@pytest.fixture
def registered(monkeypatch):
    monkeypatch.setattr(registry, "_REGISTRY", {})

    def _register(factory: str = FACTORY):
        register_speculative_method(
            NAME,
            SpeculativeMethodSpec(
                SPECULATOR,
                full_attn_verify_extension=factory,
            ),
        )

    return _register


def _vllm_config(method: str):
    vllm_config = create_vllm_config(block_size=BLOCK_SIZE)
    vllm_config.speculative_config = SpeculativeConfig(
        method=method, num_speculative_tokens=3
    )
    return vllm_config


def _kv_cache_spec(*, is_draft: bool = False) -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.float16,
        is_draft=is_draft,
    )


def _group(
    backend: type[AttentionBackend] = FakeFullAttentionBackend,
    *,
    is_draft: bool = False,
) -> AttentionGroup:
    return AttentionGroup(
        backend=backend,
        layer_names=["model.layers.0.self_attn.attn"],
        kv_cache_spec=_kv_cache_spec(is_draft=is_draft),
        kv_cache_group_id=0,
    )


def _rows(
    metadata: FullAttentionVerifyMetadata, output: torch.Tensor
) -> FullAttentionVerifyRows:
    num_tokens = output.shape[0]
    return FullAttentionVerifyRows(
        layer_name="model.layers.0.self_attn.attn",
        query=torch.zeros_like(output),
        key=torch.zeros(num_tokens, 1, 8),
        value=torch.zeros(num_tokens, 1, 8),
        kv_cache=torch.zeros(2, 1, BLOCK_SIZE, 16),
        output=output,
        metadata=metadata,
        num_heads=2,
        num_kv_heads=1,
        head_size=8,
        head_size_v=8,
        scale=8**-0.5,
        kv_cache_dtype="auto",
        attn_type=AttentionType.DECODER,
        sliding_window=None,
        logits_soft_cap=None,
        alibi_slopes=None,
        sinks=None,
        q_scale=torch.ones(1),
        k_scale=torch.ones(1),
        v_scale=torch.ones(1),
    )


def _set_attribute(obj: object, name: str, value: object) -> None:
    setattr(obj, name, value)


def test_builtin_method_does_not_create_or_attach_an_extension():
    group = _group()
    group.create_metadata_builders(_vllm_config("ngram"), DEVICE)

    builder = group.get_metadata_builder()
    assert isinstance(builder, RecordingMetadataBuilder)
    assert builder.verify_extension is None
    assert created == []


def test_factory_builder_metadata_and_dispatch_contract(registered):
    registered()
    vllm_config = _vllm_config(NAME)
    group = _group()
    group.create_metadata_builders(vllm_config, DEVICE, num_metadata_builders=2)

    builder = group.get_metadata_builder()
    assert isinstance(builder, RecordingMetadataBuilder)
    assert len(created) == 1
    extension = created[0]
    assert builder.verify_extension is extension
    second_builder = group.get_metadata_builder(1)
    assert isinstance(second_builder, RecordingMetadataBuilder)
    assert second_builder.verify_extension is extension
    assert extension.vllm_config is vllm_config

    common = create_common_attn_metadata(
        BatchSpec(seq_lens=[20], query_lens=[4]), BLOCK_SIZE, DEVICE
    ).replace(
        positions=torch.arange(4),
        is_prefilling=torch.zeros(1, dtype=torch.bool),
    )
    metadata = group.build_metadata(common)
    assert metadata.verify_extension is extension
    assert metadata.verify.query_start_loc is common.query_start_loc
    assert metadata.verify.seq_lens is common.seq_lens
    assert metadata.verify.block_table is common.block_table_tensor
    assert metadata.verify.slot_mapping is common.slot_mapping

    output = torch.zeros(4, 2, 8)
    rows = _rows(metadata.verify, output)
    dispatch_full_attention_verify(metadata.verify_extension, rows)

    assert extension.rows == [rows]
    assert torch.equal(output, torch.full_like(output, 7))
    with pytest.raises(FrozenInstanceError):
        _set_attribute(metadata.verify, "max_query_len", 5)
    with pytest.raises(FrozenInstanceError):
        _set_attribute(rows, "scale", 1.0)


def test_dispatch_refuses_metadata_from_another_backend(registered):
    registered()
    group = _group()
    group.create_metadata_builders(_vllm_config(NAME), DEVICE)
    builder = group.get_metadata_builder()
    assert isinstance(builder, RecordingMetadataBuilder)
    extension = builder.verify_extension
    assert extension is not None

    common = create_common_attn_metadata(
        BatchSpec(seq_lens=[20], query_lens=[2]), BLOCK_SIZE, DEVICE
    )
    metadata = group.build_metadata(common)
    assert metadata.verify_extension is extension
    verify_metadata = replace(
        metadata.verify,
        backend=FullAttentionVerifyBackend("OTHER", "other.module", "Backend"),
    )
    output = torch.zeros(2, 2, 8)
    rows = _rows(verify_metadata, output)

    with pytest.raises(RuntimeError, match="cannot dispatch"):
        dispatch_full_attention_verify(metadata.verify_extension, rows)
    assert not output.any()


def test_backend_without_forward_wiring_is_rejected(registered):
    registered()
    with pytest.raises(ValueError, match="does not wire"):
        _group(UnwiredBackend).create_metadata_builders(_vllm_config(NAME), DEVICE)


def test_extension_must_support_the_exact_backend(registered):
    registered()
    with pytest.raises(ValueError, match="does not support"):
        _group(UnsupportedCapableBackend).create_metadata_builders(
            _vllm_config(NAME), DEVICE
        )


def test_factory_result_is_checked(registered):
    registered(INVALID_FACTORY)
    with pytest.raises(TypeError, match="must return FullAttentionVerifyExtension"):
        _group().create_metadata_builders(_vllm_config(NAME), DEVICE)


def test_none_factory_does_not_require_backend_wiring(registered):
    registered(NONE_FACTORY)
    group = _group(UnwiredBackend)
    group.create_metadata_builders(_vllm_config(NAME), DEVICE)

    builder = group.get_metadata_builder()
    assert isinstance(builder, RecordingMetadataBuilder)
    assert builder.verify_extension is None


def test_draft_attention_group_does_not_get_target_verify_extension(registered):
    registered()
    group = _group(is_draft=True)
    group.create_metadata_builders(_vllm_config(NAME), DEVICE)

    builder = group.get_metadata_builder()
    assert isinstance(builder, RecordingMetadataBuilder)
    assert builder.verify_extension is None
    assert created == []
