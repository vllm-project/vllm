# SPDX-License-Identifier: Apache-2.0

# Standard
from types import ModuleType, SimpleNamespace
from typing import Any, cast
import sys

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.platform import torch_ops as py_ops
from lmcache.v1.platform.musa import native_kv_transfer as musa_native


def _make_native_module() -> ModuleType:
    """Create a fake native module with the Stage2 LMCache ABI."""
    module = ModuleType("musa_aiter")
    module.native_lmcache_kv_transfer_abi_version = lambda: 1  # type: ignore[attr-defined]
    module.lmcache_kv_paged_to_buffer = lambda *args, **kwargs: True  # type: ignore[attr-defined]
    module.lmcache_kv_buffer_to_paged = lambda *args, **kwargs: True  # type: ignore[attr-defined]
    module.lmcache_mla_paged_to_buffer = lambda *args, **kwargs: True  # type: ignore[attr-defined]
    module.lmcache_mla_buffer_to_paged = lambda *args, **kwargs: True  # type: ignore[attr-defined]
    return module


def _make_block_shape_desc(
    *,
    num_layers: int = 2,
    num_blocks: int = 4,
    block_size: int = 4,
    num_heads: int = 2,
    head_size: int = 8,
) -> py_ops.PageBufferShapeDesc:
    """Create a page-buffer shape descriptor for block-transfer tests."""
    shape_desc = py_ops.PageBufferShapeDesc()
    shape_desc.nl = num_layers
    shape_desc.nb = num_blocks
    shape_desc.bs = block_size
    shape_desc.nh = num_heads
    shape_desc.hs = head_size
    shape_desc.element_size = torch.empty((), dtype=torch.float32).element_size()
    shape_desc.kv_size = 2
    return shape_desc


def test_native_transfer_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Native MUSA transfer is opt-in so Stage1 behavior remains unchanged."""
    monkeypatch.delenv("LMCACHE_MUSA_NATIVE_KV_TRANSFER", raising=False)

    assert musa_native.is_native_musa_kv_transfer_enabled() is False


def test_native_transfer_module_lives_under_musa_platform() -> None:
    """The native KV-transfer adapter belongs to the MUSA platform package."""
    assert musa_native.__name__ == "lmcache.v1.platform.musa.native_kv_transfer"


def test_musa_device_ops_overrides_multi_layer_block_kv_transfer() -> None:
    """MusaDeviceOps overrides multi_layer_block_kv_transfer via MRO."""
    # First Party
    from lmcache.v1.platform import torch_ops
    from lmcache.v1.platform.musa.device_ops import MusaDeviceOps

    assert MusaDeviceOps.device_type == "musa"
    # The override is on the class itself, not the torch baseline.
    assert (
        MusaDeviceOps.multi_layer_block_kv_transfer
        is not torch_ops.multi_layer_block_kv_transfer
    )


def test_native_transfer_enabled_by_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The opt-in environment variable enables native dispatch attempts."""
    monkeypatch.setenv("LMCACHE_MUSA_NATIVE_KV_TRANSFER", "1")

    assert musa_native.is_native_musa_kv_transfer_enabled() is True


def test_load_native_module_returns_none_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing optional ``musa_aiter`` must not break LMCache imports."""
    monkeypatch.setitem(cast(dict[str, Any], sys.modules), "musa_aiter", None)

    assert musa_native.load_native_musa_module() is None


def test_check_native_abi_requires_expected_symbols() -> None:
    """LMCache accepts only the Stage2 LMCache-compatible native ABI."""
    assert musa_native.check_native_abi(_make_native_module()) is True


def test_check_native_abi_rejects_missing_symbol() -> None:
    """The adapter falls back when the optional native module is incomplete."""
    module = SimpleNamespace(native_lmcache_kv_transfer_abi_version=lambda: 1)

    assert musa_native.check_native_abi(module) is False


def test_check_native_abi_rejects_non_callable_symbol() -> None:
    """Native symbols must be callable to be considered ABI-compatible."""
    module = _make_native_module()
    module.lmcache_kv_buffer_to_paged = object()  # type: ignore[attr-defined]

    assert musa_native.check_native_abi(module) is False


def test_try_native_to_gpu_calls_non_mla_symbol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-MLA scatter uses the LMCache-compatible buffer-to-paged call."""
    calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
    module = _make_native_module()

    def _buffer_to_paged(*args: Any, **kwargs: Any) -> bool:
        calls.append(("buffer_to_paged", args, kwargs))
        return True

    module.lmcache_kv_buffer_to_paged = _buffer_to_paged  # type: ignore[attr-defined]
    monkeypatch.setattr(musa_native, "load_native_musa_module", lambda: module)
    monkeypatch.setattr(musa_native, "_native_tensors_ready", lambda *args: True)
    monkeypatch.setenv("LMCACHE_MUSA_NATIVE_KV_TRANSFER", "1")

    used = musa_native.try_native_to_gpu(
        use_mla=False,
        memory_tensor=torch.empty(2, 1, 4, 8),
        kvcaches=[torch.empty(2, 2, 2, 2, 4)],
        slot_mapping=torch.arange(4),
        start=0,
        end=4,
        skip_prefix_n_tokens=0,
        block_size=2,
        num_heads=2,
        head_size=4,
    )

    assert used is True
    assert calls[0][0] == "buffer_to_paged"
    assert torch.equal(calls[0][1][2], torch.arange(4))
    assert calls[0][1][3] == 0


def test_try_native_to_gpu_returns_true_for_empty_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty native transfer is a completed no-op after ABI is ready."""
    module = _make_native_module()
    monkeypatch.setattr(musa_native, "load_native_musa_module", lambda: module)
    monkeypatch.setenv("LMCACHE_MUSA_NATIVE_KV_TRANSFER", "1")

    assert musa_native.try_native_to_gpu(
        use_mla=False,
        memory_tensor=torch.empty(2, 1, 4, 8),
        kvcaches=[torch.empty(2, 2, 2, 2, 4)],
        slot_mapping=torch.arange(4),
        start=0,
        end=4,
        skip_prefix_n_tokens=4,
        block_size=2,
        num_heads=2,
        head_size=4,
    )


def test_try_native_to_gpu_rejects_cpu_tensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native dispatch must not consume CPU tensors through musa_aiter fallback."""
    module = _make_native_module()
    monkeypatch.setattr(musa_native, "load_native_musa_module", lambda: module)
    monkeypatch.setenv("LMCACHE_MUSA_NATIVE_KV_TRANSFER", "1")

    used = musa_native.try_native_to_gpu(
        use_mla=False,
        memory_tensor=torch.empty(2, 1, 4, 8),
        kvcaches=[torch.empty(2, 2, 2, 2, 4)],
        slot_mapping=torch.arange(4),
        start=0,
        end=4,
        skip_prefix_n_tokens=0,
        block_size=2,
        num_heads=2,
        head_size=4,
    )

    assert used is False


def test_try_native_from_gpu_falls_back_on_native_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native failures must return False so callers keep the torch fallback."""
    module = _make_native_module()

    def _paged_to_buffer(*args: Any, **kwargs: Any) -> bool:
        raise RuntimeError("native boom")

    module.lmcache_kv_paged_to_buffer = _paged_to_buffer  # type: ignore[attr-defined]
    monkeypatch.setattr(musa_native, "load_native_musa_module", lambda: module)
    monkeypatch.setattr(musa_native, "_native_tensors_ready", lambda *args: True)
    monkeypatch.setenv("LMCACHE_MUSA_NATIVE_KV_TRANSFER", "1")

    used = musa_native.try_native_from_gpu(
        use_mla=False,
        memory_tensor=torch.empty(2, 1, 4, 8),
        kvcaches=[torch.empty(2, 2, 2, 2, 4)],
        slot_mapping=torch.arange(4),
        start=0,
        end=4,
        block_size=2,
        num_heads=2,
        head_size=4,
    )

    assert used is False


def test_try_native_from_gpu_rejects_cpu_tensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native gather must not consume CPU tensors through musa_aiter fallback."""
    module = _make_native_module()
    monkeypatch.setattr(musa_native, "load_native_musa_module", lambda: module)
    monkeypatch.setenv("LMCACHE_MUSA_NATIVE_KV_TRANSFER", "1")

    used = musa_native.try_native_from_gpu(
        use_mla=False,
        memory_tensor=torch.empty(2, 1, 4, 8),
        kvcaches=[torch.empty(2, 2, 2, 2, 4)],
        slot_mapping=torch.arange(4),
        start=0,
        end=4,
        block_size=2,
        num_heads=2,
        head_size=4,
    )

    assert used is False


def test_native_block_gather_uses_device_staging_for_cpu_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Block-level native gather stages CPU output and expands slot IDs."""
    paged_layers = [torch.zeros(4, 4, 16) for _ in range(2)]
    out = torch.zeros(2, 8, 16)
    captured: dict[str, Any] = {}

    monkeypatch.setenv("LMCACHE_MUSA_NATIVE_KV_TRANSFER", "1")
    monkeypatch.setattr(
        musa_native, "_is_musa_block_transfer_candidate", lambda *_args: True
    )

    def _fake_native_from_gpu(**kwargs: Any) -> bool:
        memory_tensor = kwargs["memory_tensor"]
        captured["used_cpu_out_directly"] = memory_tensor is out
        captured["slot_mapping"] = kwargs["slot_mapping"].cpu()
        memory_tensor.fill_(7.0)
        return True

    monkeypatch.setattr(musa_native, "try_native_from_gpu", _fake_native_from_gpu)

    used = musa_native.try_native_multi_layer_block_kv_transfer(
        paged_layers=paged_layers,
        object_tensors=[out],
        block_ids=[2, 4],
        direction=py_ops.TransferDirection.D2H,
        shape_desc=_make_block_shape_desc(head_size=16),
        lmcache_chunk_size=8,
        engine_kv_format=py_ops.EngineKVFormat.NL_X_NB_BS_HS,
        skip_prefix_n_blocks=0,
    )

    assert used is True
    assert captured["used_cpu_out_directly"] is False
    assert torch.equal(
        captured["slot_mapping"], torch.tensor([8, 9, 10, 11, 16, 17, 18, 19])
    )
    assert torch.all(out == 7.0)


def test_native_block_scatter_uses_device_staging_for_cpu_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Block-level native scatter must not pass CPU/SHM chunks directly."""
    paged_layers = [torch.zeros(4, 4, 16) for _ in range(2)]
    chunk = torch.zeros(2, 8, 16)
    captured: dict[str, bool] = {}

    monkeypatch.setenv("LMCACHE_MUSA_NATIVE_KV_TRANSFER", "1")
    monkeypatch.setattr(
        musa_native, "_is_musa_block_transfer_candidate", lambda *_args: True
    )

    def _fake_native_to_gpu(**kwargs: Any) -> bool:
        captured["used_cpu_chunk_directly"] = kwargs["memory_tensor"] is chunk
        return True

    monkeypatch.setattr(musa_native, "try_native_to_gpu", _fake_native_to_gpu)

    used = musa_native.try_native_multi_layer_block_kv_transfer(
        paged_layers=paged_layers,
        object_tensors=[chunk],
        block_ids=[0, 1],
        direction=py_ops.TransferDirection.H2D,
        shape_desc=_make_block_shape_desc(head_size=16),
        lmcache_chunk_size=8,
        engine_kv_format=py_ops.EngineKVFormat.NL_X_NB_BS_HS,
        skip_prefix_n_blocks=0,
    )

    assert used is True
    assert captured["used_cpu_chunk_directly"] is False


def test_native_block_scatter_uses_whole_block_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Block-level native scatter receives skip offsets in whole-block tokens."""
    paged_layers = [torch.zeros(4, 4, 16) for _ in range(2)]
    chunk = torch.zeros(2, 8, 16)
    captured: dict[str, int] = {}

    monkeypatch.setenv("LMCACHE_MUSA_NATIVE_KV_TRANSFER", "1")
    monkeypatch.setattr(
        musa_native, "_is_musa_block_transfer_candidate", lambda *_args: True
    )

    def _fake_native_to_gpu(**kwargs: Any) -> bool:
        captured["skip_prefix_n_tokens"] = kwargs["skip_prefix_n_tokens"]
        return True

    monkeypatch.setattr(musa_native, "try_native_to_gpu", _fake_native_to_gpu)

    used = musa_native.try_native_multi_layer_block_kv_transfer(
        paged_layers=paged_layers,
        object_tensors=[chunk],
        block_ids=[0, 1],
        direction=py_ops.TransferDirection.H2D,
        shape_desc=_make_block_shape_desc(head_size=16),
        lmcache_chunk_size=8,
        engine_kv_format=py_ops.EngineKVFormat.NL_X_NB_BS_HS,
        skip_prefix_n_blocks=1,
    )

    assert used is True
    assert captured["skip_prefix_n_tokens"] == 4


def test_native_block_gather_skips_native_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Block-level native gather is not attempted when env opt-in is disabled."""
    paged_layers = [torch.zeros(4, 4, 16) for _ in range(2)]

    monkeypatch.delenv("LMCACHE_MUSA_NATIVE_KV_TRANSFER", raising=False)
    monkeypatch.setattr(
        musa_native, "_is_musa_block_transfer_candidate", lambda *_args: True
    )

    def _raise_if_called(**_kwargs: Any) -> bool:
        raise AssertionError("native path should not be called when disabled")

    monkeypatch.setattr(musa_native, "try_native_from_gpu", _raise_if_called)

    used = musa_native.try_native_multi_layer_block_kv_transfer(
        paged_layers=paged_layers,
        object_tensors=[torch.zeros(2, 8, 16)],
        block_ids=[0, 1],
        direction=py_ops.TransferDirection.D2H,
        shape_desc=_make_block_shape_desc(head_size=16),
        lmcache_chunk_size=8,
        engine_kv_format=py_ops.EngineKVFormat.NL_X_NB_BS_HS,
        skip_prefix_n_blocks=0,
    )

    assert used is False


def test_native_block_transfer_rejects_unvalidated_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unsupported MUSA layouts fall back below the device-layer entry."""
    monkeypatch.setenv("LMCACHE_MUSA_NATIVE_KV_TRANSFER", "1")
    monkeypatch.setattr(
        musa_native, "_is_musa_block_transfer_candidate", lambda *_args: True
    )

    used = musa_native.try_native_multi_layer_block_kv_transfer(
        paged_layers=[torch.zeros(2, 4, 2, 4, 8)],
        object_tensors=[torch.zeros(2, 2, 8, 16)],
        block_ids=[0, 1],
        direction=py_ops.TransferDirection.H2D,
        shape_desc=_make_block_shape_desc(),
        lmcache_chunk_size=8,
        engine_kv_format=py_ops.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS,
        skip_prefix_n_blocks=0,
    )

    assert used is False


def test_musa_ops_block_transfer_entry_dispatches_to_musa_platform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MusaDeviceOps dispatches to native transfer when tensor-backed."""
    # First Party
    from lmcache.v1.platform.musa.device_ops import MusaDeviceOps

    captured: dict[str, Any] = {}

    def _fake_musa_block_transfer(**kwargs: Any) -> bool:
        captured.update(kwargs)
        return True

    monkeypatch.setattr(
        musa_native,
        "try_native_multi_layer_block_kv_transfer",
        _fake_musa_block_transfer,
    )

    paged_layers = [torch.zeros(4, 4, 16) for _ in range(2)]
    object_tensors = [torch.zeros(2, 8, 16)]
    MusaDeviceOps().multi_layer_block_kv_transfer(
        paged_layers,
        object_tensors,
        [0, 1],
        "musa",
        py_ops.TransferDirection.D2H,
        _make_block_shape_desc(head_size=16),
        8,
        py_ops.EngineKVFormat.NL_X_NB_BS_HS,
        0,
    )

    assert captured["paged_layers"] is paged_layers
    assert captured["object_tensors"] is object_tensors
    assert captured["block_ids"] == [0, 1]
