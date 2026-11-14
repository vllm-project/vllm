# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tempfile
from unittest.mock import Mock

import huggingface_hub.constants
import pytest
import torch
from huggingface_hub.utils import LocalEntryNotFoundError

import vllm.model_executor.model_loader.weight_utils as weight_utils
from vllm.model_executor.model_loader.weight_utils import (
    download_weights_from_hf,
    drop_checkpoint_cache,
    initialize_single_dummy_weight,
    load_attention_sink,
    maybe_remap_kv_scale_name,
)
from vllm.platforms import current_platform


@pytest.mark.parametrize(
    "checkpoint_heads,runtime_heads,tp_size",
    [(8, 4, 2), (8, 8, 1), (8, 6, 2), (16, 12, 4)],
)
def test_attention_sink_loader_shapes(
    monkeypatch, checkpoint_heads, runtime_heads, tp_size
):
    """Each rank loads its checkpoint heads and disables only padded heads."""
    monkeypatch.setattr(
        weight_utils, "get_tensor_model_parallel_world_size", lambda: tp_size
    )
    checkpoint = torch.arange(checkpoint_heads, dtype=torch.float32)
    for tp_rank in range(tp_size):
        monkeypatch.setattr(
            weight_utils, "get_tensor_model_parallel_rank", lambda rank=tp_rank: rank
        )
        param = torch.nn.Parameter(torch.zeros(runtime_heads), requires_grad=False)
        load_attention_sink(param, checkpoint)
        local_heads = checkpoint.chunk(tp_size)[tp_rank]
        expected = torch.cat(
            (
                local_heads,
                torch.full((runtime_heads - local_heads.numel(),), -torch.inf),
            )
        )
        assert torch.equal(param, expected)


@pytest.mark.parametrize(
    "checkpoint_shape,runtime_shape,error",
    [
        ((2, 4), (4,), RuntimeError),
        ((8,), (2, 2), AssertionError),
        ((7,), (4,), ValueError),
        ((8,), (3,), AssertionError),
    ],
)
def test_attention_sink_loader_rejects_invalid_shapes(
    monkeypatch, checkpoint_shape, runtime_shape, error
):
    """Invalid sink shapes fail before changing the runtime parameter."""
    monkeypatch.setattr(weight_utils, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(weight_utils, "get_tensor_model_parallel_rank", lambda: 0)
    param = torch.nn.Parameter(torch.ones(runtime_shape), requires_grad=False)
    with pytest.raises(error):
        load_attention_sink(param, torch.zeros(checkpoint_shape))
    assert torch.equal(param, torch.ones(runtime_shape))


@pytest.mark.parametrize("param_dtype", [torch.bfloat16, torch.float32])
def test_attention_sink_loader_keeps_param_dtype(monkeypatch, param_dtype):
    """A checkpoint dtype different from the runtime parameter is cast, not copied."""
    monkeypatch.setattr(weight_utils, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(weight_utils, "get_tensor_model_parallel_rank", lambda: 0)
    param = torch.nn.Parameter(torch.zeros(4, dtype=param_dtype), requires_grad=False)
    load_attention_sink(param, torch.arange(4, dtype=torch.float32))
    assert param.dtype == param_dtype
    assert torch.equal(param.float(), torch.arange(4, dtype=torch.float32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_attention_sink_loader_accepts_cpu_checkpoint(monkeypatch):
    """Sinks are normally loaded from CPU checkpoints into device parameters."""
    monkeypatch.setattr(weight_utils, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(weight_utils, "get_tensor_model_parallel_rank", lambda: 1)
    param = torch.nn.Parameter(
        torch.full((6,), 123.0, device="cuda"), requires_grad=False
    )
    # The test suite sets the default device, so pin the checkpoint to CPU.
    checkpoint = torch.arange(8, dtype=torch.float32, device="cpu")
    load_attention_sink(param, checkpoint)
    assert param.device.type == "cuda"
    expected = torch.arange(4, 8, dtype=torch.float32, device="cpu")
    assert torch.equal(param[:4].cpu(), expected)
    assert torch.isinf(param[4:]).all() and (param[4:] < 0).all()


@pytest.mark.parametrize(
    "num_heads,dtype", [(4, torch.float32), (8, torch.bfloat16), (6, None)]
)
def test_make_attention_sink_registers_loader_and_padding(num_heads, dtype):
    """The factory is the single place that creates and registers a sink."""
    sink = weight_utils.make_attention_sink(num_heads, dtype)
    assert sink.shape == (num_heads,)
    assert sink.dtype == (dtype or torch.get_default_dtype())
    assert sink.requires_grad is False
    assert torch.isinf(sink).all() and (sink < 0).all()
    assert sink.weight_loader is load_attention_sink


def test_download_weights_from_hf():
    with tempfile.TemporaryDirectory() as tmpdir:
        # assert LocalEntryNotFoundError error is thrown
        # if offline is set and model is not cached
        huggingface_hub.constants.HF_HUB_OFFLINE = True
        with pytest.raises(LocalEntryNotFoundError):
            download_weights_from_hf(
                "facebook/opt-125m",
                allow_patterns=["*.safetensors", "*.bin"],
                cache_dir=tmpdir,
            )

        # download the model
        huggingface_hub.constants.HF_HUB_OFFLINE = False
        download_weights_from_hf(
            "facebook/opt-125m",
            allow_patterns=["*.safetensors", "*.bin"],
            cache_dir=tmpdir,
        )

        # now it should work offline
        huggingface_hub.constants.HF_HUB_OFFLINE = True
        assert (
            download_weights_from_hf(
                "facebook/opt-125m",
                allow_patterns=["*.safetensors", "*.bin"],
                cache_dir=tmpdir,
            )
            is not None
        )


class TestMaybeRemapKvScaleName:
    """Tests for maybe_remap_kv_scale_name covering all checkpoint formats."""

    PARAMS_DICT = {
        "model.layers.0.self_attn.attn.k_scale": None,
        "model.layers.0.self_attn.attn.v_scale": None,
        "model.layers.0.self_attn.attn.q_scale": None,
        "model.layers.0.self_attn.qkv_proj.weight": None,
    }

    def test_qkv_proj_k_scale(self):
        """Qwen3-MoE / llm-compressor format: qkv_proj.k_scale -> attn.k_scale
        Regression test for https://github.com/vllm-project/vllm/issues/25047"""
        result = maybe_remap_kv_scale_name(
            "model.layers.0.self_attn.qkv_proj.k_scale", self.PARAMS_DICT
        )
        assert result == "model.layers.0.self_attn.attn.k_scale"

    def test_qkv_proj_v_scale(self):
        """Qwen3-MoE / llm-compressor format: qkv_proj.v_scale -> attn.v_scale
        Regression test for https://github.com/vllm-project/vllm/issues/25047"""
        result = maybe_remap_kv_scale_name(
            "model.layers.0.self_attn.qkv_proj.v_scale", self.PARAMS_DICT
        )
        assert result == "model.layers.0.self_attn.attn.v_scale"

    def test_modelopt_k_proj_k_scale(self):
        """ModelOpt format: k_proj.k_scale -> attn.k_scale."""
        result = maybe_remap_kv_scale_name(
            "model.layers.0.self_attn.k_proj.k_scale", self.PARAMS_DICT
        )
        assert result == "model.layers.0.self_attn.attn.k_scale"

    def test_modelopt_v_proj_v_scale(self):
        """ModelOpt format: v_proj.v_scale -> attn.v_scale."""
        result = maybe_remap_kv_scale_name(
            "model.layers.0.self_attn.v_proj.v_scale", self.PARAMS_DICT
        )
        assert result == "model.layers.0.self_attn.attn.v_scale"

    def test_deprecated_kv_scale(self):
        """Old format: kv_scale -> attn.k_scale (deprecated)."""
        result = maybe_remap_kv_scale_name(
            "model.layers.0.self_attn.kv_scale", self.PARAMS_DICT
        )
        assert result == "model.layers.0.self_attn.attn.k_scale"

    def test_default_bare_k_scale(self):
        """Default format: .k_scale -> .attn.k_scale."""
        result = maybe_remap_kv_scale_name(
            "model.layers.0.self_attn.k_scale", self.PARAMS_DICT
        )
        assert result == "model.layers.0.self_attn.attn.k_scale"

    def test_non_scale_name_unchanged(self):
        """Non-scale names should be returned unchanged."""
        name = "model.layers.0.self_attn.qkv_proj.weight"
        result = maybe_remap_kv_scale_name(name, self.PARAMS_DICT)
        assert result == name

    def test_nvfp4_modelopt_k_proj_k_scale(self):
        """ModelOpt NVFP4 format (e.g. nvidia/Qwen3-30B-A3B-NVFP4):
        k_proj.k_scale -> attn.k_scale.
        Validates that NVFP4 checkpoints are not broken by this change."""
        result = maybe_remap_kv_scale_name(
            "model.layers.0.self_attn.k_proj.k_scale", self.PARAMS_DICT
        )
        assert result == "model.layers.0.self_attn.attn.k_scale"

    def test_nvfp4_modelopt_v_proj_v_scale(self):
        """ModelOpt NVFP4 format (e.g. nvidia/Qwen3-30B-A3B-NVFP4):
        v_proj.v_scale -> attn.v_scale.
        Validates that NVFP4 checkpoints are not broken by this change."""
        result = maybe_remap_kv_scale_name(
            "model.layers.0.self_attn.v_proj.v_scale", self.PARAMS_DICT
        )
        assert result == "model.layers.0.self_attn.attn.v_scale"

    def test_qwen3_vl_moe_qkv_proj_k_scale(self):
        """Qwen3-VL-MoE uses the same fused qkv_proj naming as Qwen3-MoE.
        Regression test for qwen3_vl_moe.py fix (same bug as #25047)."""
        result = maybe_remap_kv_scale_name(
            "model.layers.0.self_attn.qkv_proj.k_scale", self.PARAMS_DICT
        )
        assert result == "model.layers.0.self_attn.attn.k_scale"

    def test_qwen3_vl_moe_qkv_proj_v_scale(self):
        """Qwen3-VL-MoE uses the same fused qkv_proj naming as Qwen3-MoE.
        Regression test for qwen3_vl_moe.py fix (same bug as #25047)."""
        result = maybe_remap_kv_scale_name(
            "model.layers.0.self_attn.qkv_proj.v_scale", self.PARAMS_DICT
        )
        assert result == "model.layers.0.self_attn.attn.v_scale"

    def test_nvfp4_weight_scale_not_remapped(self):
        """NVFP4 weight_scale should not be touched by remap (not a kv scale)."""
        name = "model.layers.0.self_attn.k_proj.weight_scale"
        result = maybe_remap_kv_scale_name(name, self.PARAMS_DICT)
        assert result == name

    def test_nvfp4_input_scale_not_remapped(self):
        """NVFP4 input_scale should not be touched by remap (not a kv scale)."""
        name = "model.layers.0.self_attn.k_proj.input_scale"
        result = maybe_remap_kv_scale_name(name, self.PARAMS_DICT)
        assert result == name

    def test_missing_target_returns_none(self):
        """If remapped name not in params_dict, return None."""
        empty_params: dict[str, None] = {}
        result = maybe_remap_kv_scale_name(
            "model.layers.0.self_attn.qkv_proj.k_scale", empty_params
        )
        assert result is None


def test_checkpoint_weight_mapper_discards_serialized_g_idx():
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizationConfig,
    )

    mapper = QuantizationConfig.get_checkpoint_weight_mapper()

    assert mapper._map_name("model.layers.0.g_idx") is None
    assert mapper._map_name("model.layers.0.qweight") == "model.layers.0.qweight"


class TestKvCacheScaleMapper:
    """The `WeightsMapper` returned by `get_cache_scale_mapper` replaces the
    per-model `maybe_remap_kv_scale_name` calls. It must remap the same set of
    checkpoint formats (the non-`params_dict`-dependent ones) and be idempotent
    so it composes safely with a model's own qkv/gate_up `hf_to_vllm_mapper`."""

    def _mapper(self):
        # `get_cache_scale_mapper` does not use `self`; call it on the base
        # class to get the default (non-config-specific) mapper.
        from vllm.model_executor.layers.quantization.base_config import (
            QuantizationConfig,
        )

        return QuantizationConfig.get_cache_scale_mapper()

    def _map(self, name: str) -> str | None:
        return self._mapper()._map_name(name)

    @pytest.mark.parametrize(
        "name,expected",
        [
            # Qwen3-MoE / llm-compressor fused qkv_proj
            (
                "model.layers.0.self_attn.qkv_proj.k_scale",
                "model.layers.0.self_attn.attn.k_scale",
            ),
            (
                "model.layers.0.self_attn.qkv_proj.v_scale",
                "model.layers.0.self_attn.attn.v_scale",
            ),
            # ModelOpt / NVFP4 k_proj/v_proj
            (
                "model.layers.0.self_attn.k_proj.k_scale",
                "model.layers.0.self_attn.attn.k_scale",
            ),
            (
                "model.layers.0.self_attn.v_proj.v_scale",
                "model.layers.0.self_attn.attn.v_scale",
            ),
            # deprecated fused kv_scale and bare scales
            (
                "model.layers.0.self_attn.kv_scale",
                "model.layers.0.self_attn.attn.k_scale",
            ),
            (
                "model.layers.0.self_attn.k_scale",
                "model.layers.0.self_attn.attn.k_scale",
            ),
            # NemotronH mixer
            (
                "model.layers.0.mixer.k_proj.k_scale",
                "model.layers.0.mixer.attn.k_scale",
            ),
            # already in vLLM form -> unchanged (idempotent)
            (
                "model.layers.0.self_attn.attn.k_scale",
                "model.layers.0.self_attn.attn.k_scale",
            ),
            # non-kv scales must not be touched
            (
                "model.layers.0.self_attn.k_proj.weight_scale",
                "model.layers.0.self_attn.k_proj.weight_scale",
            ),
            (
                "model.layers.0.self_attn.k_proj.input_scale",
                "model.layers.0.self_attn.k_proj.input_scale",
            ),
            # regular weights untouched
            (
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.self_attn.q_proj.weight",
            ),
        ],
    )
    def test_remap(self, name, expected):
        assert self._map(name) == expected

    @pytest.mark.parametrize(
        "name",
        [
            "model.layers.0.self_attn.k_scale",
            "model.layers.0.self_attn.k_proj.k_scale",
            "model.layers.0.self_attn.qkv_proj.v_scale",
            "model.layers.0.mixer.k_proj.k_scale",
        ],
    )
    def test_idempotent(self, name):
        once = self._map(name)
        assert once is not None
        assert self._map(once) == once

    def test_composes_with_qkv_mapper(self):
        """Applied together with a model's qkv/gate_up mapper, the regex scale
        rules run before the substr rename, so scales are normalized to `.attn.`
        and regular projections are still fused correctly."""
        from vllm.model_executor.models.utils import WeightsMapper

        model_mapper = WeightsMapper(
            orig_to_new_substr={
                ".q_proj": ".qkv_proj.q",
                ".k_proj": ".qkv_proj.k",
                ".v_proj": ".qkv_proj.v",
            }
        )
        # AutoWeightsLoader does `mapper |= cache_scale_mapper`
        combined = model_mapper | self._mapper()

        assert (
            combined._map_name("model.layers.0.self_attn.q_proj.weight")
            == "model.layers.0.self_attn.qkv_proj.q.weight"
        )
        assert (
            combined._map_name("model.layers.0.self_attn.k_proj.k_scale")
            == "model.layers.0.self_attn.attn.k_scale"
        )
        assert (
            combined._map_name("model.layers.0.self_attn.k_scale")
            == "model.layers.0.self_attn.attn.k_scale"
        )


@pytest.mark.parametrize("fails", [False, True])
def test_drop_checkpoint_cache_releases_local_weight_files(
    tmp_path, monkeypatch, fails
):
    """Evict only weights; cache eviction errors must not prevent loading."""
    for name in ("weights.safetensors", "config.json"):
        (tmp_path / name).touch()
    fadvise = Mock(side_effect=OSError("cache eviction failed") if fails else None)
    monkeypatch.setattr("os.posix_fadvise", fadvise)
    drop_checkpoint_cache(str(tmp_path))
    fadvise.assert_called_once()


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_dummy_fp8_weight_is_not_degenerate(dtype):
    """+-1e-3 underflows in FP8, so 8-bit floats default to +-1."""
    param = torch.empty(4096, dtype=dtype)
    initialize_single_dummy_weight(param)
    values = param.float()
    assert values.isfinite().all()
    assert values.abs().max() <= 1.0
    assert (values != 0).float().mean() > 0.99


@pytest.mark.parametrize(
    ("dtype", "name"),
    [(torch.float8_e8m0fnu, "weight_scale_inv"), (torch.uint8, "w13_weight_scale")],
)
def test_dummy_e8m0_scale_is_fixed_exponent(dtype, name):
    """E8M0 scales, including uint8-stored ones, get a finite 2**-7."""
    param = torch.empty(64, dtype=dtype)
    initialize_single_dummy_weight(param, name=name)
    assert (param.view(torch.uint8) == 120).all()


@pytest.mark.parametrize("dtype", [torch.uint8, torch.int8, torch.float4_e2m1fn_x2])
def test_dummy_packed_weight_gets_random_bytes(dtype):
    """Packed weights are filled deterministically instead of left uninitialized."""
    # Some backends can't fill packed FP4 directly, so view zeroed bytes.
    params = [torch.zeros(4096, dtype=torch.uint8).view(dtype) for _ in range(2)]
    for param in params:
        initialize_single_dummy_weight(param, name="w13_weight")
    raw = params[0].view(torch.uint8)
    assert raw.unique().numel() > 250
    assert torch.equal(raw, params[1].view(torch.uint8))


@pytest.mark.skipif(current_platform.is_rocm(), reason="ROCm zeroes integer params")
def test_dummy_wide_integer_param_is_untouched():
    """Wider integer params may be indices, so random values are unsafe."""
    param = torch.arange(16, dtype=torch.int32)
    initialize_single_dummy_weight(param, name="g_idx")
    assert torch.equal(param, torch.arange(16, dtype=torch.int32))


def test_dummy_norm_weights_are_neutral(default_vllm_config):
    """Dummy norms stay identity-like, keeping activations at a realistic scale."""
    from vllm.model_executor.layers.layernorm import (
        GemmaRMSNorm,
        LayerNorm,
        RMSNorm,
        RMSNormGated,
    )

    expected = {
        RMSNorm(16): {"weight": 1.0},
        GemmaRMSNorm(16): {"weight": 0.0},
        LayerNorm(16): {"weight": 1.0, "bias": 0.0},
        RMSNormGated(16): {"weight": 1.0},
    }
    for norm, values in expected.items():
        for name, param in norm.named_parameters():
            param.data.uniform_(-5, 5)
            initialize_single_dummy_weight(param, name=name)
            assert torch.all(param == values[name]), (type(norm).__name__, name)


if __name__ == "__main__":
    test_download_weights_from_hf()


@pytest.mark.parametrize("workers", [1, 3])
def test_streaming_weights_match_lazy_and_survive_close(tmp_path, workers):
    import torch
    from safetensors.torch import save_file

    from vllm.model_executor.model_loader.weight_utils import (
        safetensors_weights_iterator,
        streaming_safetensors_weights_iterator,
    )

    paths = []
    for i in range(5):
        path = str(tmp_path / f"shard-{i}.safetensors")
        save_file({f"weight_{i}": torch.arange(37) + i}, path)
        paths.append(path)
    expected = dict(safetensors_weights_iterator(paths, False, "lazy"))
    actual = dict(
        streaming_safetensors_weights_iterator(
            paths,
            False,
            prefetch_file_window=workers,
            io_threads_per_file=2,
            block_size=31,
        )
    )
    assert actual.keys() == expected.keys()
    for name in expected:
        assert torch.equal(actual[name], expected[name])


def test_streaming_prefetch_respects_input_file_order(tmp_path):
    import torch
    from safetensors.torch import save_file

    from vllm.model_executor.model_loader.weight_utils import (
        streaming_safetensors_weights_iterator,
    )

    first = str(tmp_path / "z.safetensors")
    second = str(tmp_path / "a.safetensors")
    save_file({"first": torch.ones(1)}, first)
    save_file({"second": torch.ones(1)}, second)

    weights = streaming_safetensors_weights_iterator(
        [first, second], False, prefetch_file_window=1
    )
    assert [name for name, _ in weights] == ["first", "second"]


def test_streaming_reads_at_most_the_prefetch_window_concurrently(
    tmp_path, monkeypatch
):
    """The prefetch window bounds how many shards are read at once.

    The shards are read ahead of the consumer: the next shard is read
    while the current one is still being handed out, so the storage is
    busy while the loader copies the weights to the device.
    """
    import threading

    import torch
    from safetensors.torch import save_file

    from vllm.model_executor.model_loader import weight_utils as wu

    paths = []
    for i in range(6):
        path = str(tmp_path / f"{i}.safetensors")
        save_file({"weight": torch.arange(9)}, path)
        paths.append(path)

    real_read_shard = wu.read_shard
    lock = threading.Lock()
    second_started = threading.Event()
    in_flight = 0
    max_in_flight = 0

    def counting_read_shard(path, **kwargs):
        nonlocal in_flight, max_in_flight
        with lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
        if path == paths[1]:
            second_started.set()
        try:
            if path == paths[0]:
                # Hold the first shard until the second one has started,
                # so the two reads overlap no matter how fast the disk
                # is.
                assert second_started.wait(10), "The next shard was not read ahead"
            return real_read_shard(path, **kwargs)
        finally:
            with lock:
                in_flight -= 1

    monkeypatch.setattr(wu, "read_shard", counting_read_shard)
    weights = list(
        wu.streaming_safetensors_weights_iterator(paths, False, prefetch_file_window=2)
    )
    assert [name for name, _ in weights] == ["weight"] * 6
    assert max_in_flight == 2


def test_streaming_closes_its_handles_when_the_consumer_stops_early(
    tmp_path, monkeypatch
):
    """The reader owns the file handles, so it closes them when the
    consumer walks away from the iterator.
    """
    import torch
    from safetensors.torch import save_file

    from vllm.model_executor.model_loader import threaded_buffer_reader as tbr
    from vllm.model_executor.model_loader import weight_utils as wu

    paths = []
    for i in range(4):
        path = str(tmp_path / f"{i}.safetensors")
        save_file({"weight": torch.arange(9)}, path)
        paths.append(path)

    real_open = open
    handles = []

    def track_open(path, *args, **kwargs):
        handle = real_open(path, *args, **kwargs)
        handles.append(handle)
        return handle

    monkeypatch.setattr(tbr, "open", track_open, raising=False)
    iterator = wu.streaming_safetensors_weights_iterator(
        paths, False, prefetch_file_window=2
    )
    name, tensor = next(iterator)
    assert name == "weight"
    assert torch.equal(tensor, torch.arange(9))
    iterator.close()
    assert all(handle.closed for handle in handles)


@pytest.mark.parametrize("failure", ["empty", "missing", "truncated", "io"])
@pytest.mark.parametrize("prefetch_only", [False, True])
def test_streaming_read_failures_propagate_before_mapping(
    tmp_path, monkeypatch, failure, prefetch_only
):
    from unittest.mock import patch

    from vllm.model_executor.model_loader import weight_utils as wu

    path = tmp_path / "broken.safetensors"
    expected: type[Exception] = EOFError
    if failure != "missing":
        path.write_bytes(b"" if failure == "empty" else b"data")
    else:
        expected = FileNotFoundError
    # Exercise the portable pread fallback, including EOF and OS errors.
    monkeypatch.delattr(wu.os, "preadv", raising=False)
    if failure == "truncated":
        monkeypatch.setattr(wu.os, "pread", lambda *args: b"")
    elif failure == "io":
        expected = OSError

        def fail(*args):
            raise OSError("read failed")

        monkeypatch.setattr(wu.os, "pread", fail)
    if prefetch_only:
        # The prefetch path parses and maps the shard, so the failure
        # must arrive before it does either.
        with patch.object(wu, "BufferSafeTensors") as mapped:
            with pytest.raises(expected):
                wu.prefetch_checkpoints([str(path)])
            mapped.assert_not_called()
    else:
        with pytest.raises(expected):
            list(wu.streaming_safetensors_weights_iterator([str(path)], False))


@pytest.mark.parametrize("use_preadv", [False, True])
def test_streaming_retries_short_reads_and_filters_experts(
    tmp_path, monkeypatch, use_preadv
):
    import torch
    from safetensors.torch import save_file

    from vllm.model_executor.model_loader import weight_utils as wu

    path = str(tmp_path / "weights.safetensors")
    weights = {
        f"model.layers.0.mlp.experts.{i}.weight": torch.arange(7) for i in range(2)
    }
    save_file(weights, path)
    monkeypatch.delattr(wu.os, "preadv", raising=False)
    real_read = wu.os.pread
    monkeypatch.setattr(
        wu.os, "pread", lambda fd, n, off: real_read(fd, min(n, 3), off)
    )
    if use_preadv:

        def short_preadv(fd, buffers, offset):
            data = real_read(fd, min(len(buffers[0]), 3), offset)
            buffers[0][: len(data)] = data
            return len(data)

        monkeypatch.setattr(wu.os, "preadv", short_preadv, raising=False)
    result = dict(
        wu.safetensors_weights_iterator(
            [path],
            False,
            "streaming",
            local_expert_ids={1},
            safetensors_prefetch_block_size=19,
        )
    )
    assert set(result) == {"model.layers.0.mlp.experts.1.weight"}
    assert torch.equal(next(iter(result.values())), torch.arange(7))


def test_streaming_consumes_shards_in_input_order(tmp_path, monkeypatch):
    import threading

    import torch
    from safetensors.torch import save_file

    from vllm.model_executor.model_loader import weight_utils as wu

    slow = str(tmp_path / "0.safetensors")
    fast = str(tmp_path / "1.safetensors")
    save_file({"slow": torch.zeros(11)}, slow)
    save_file({"fast": torch.ones(3)}, fast)
    fast_read = threading.Event()
    real_read_shard = wu.read_shard

    def gated_read_shard(path, **kwargs):
        if path == fast:
            result = real_read_shard(path, **kwargs)
            fast_read.set()
            return result
        # The first shard is held until the second one has been read, so
        # a reader that consumed in completion order would hand out the
        # second shard first.
        assert fast_read.wait(10), "The second shard was not read first"
        return real_read_shard(path, **kwargs)

    monkeypatch.setattr(wu, "read_shard", gated_read_shard)
    weights = wu.streaming_safetensors_weights_iterator(
        [slow, fast],
        False,
        prefetch_file_window=2,
    )
    try:
        name, tensor = next(weights)
        assert name == "slow"
        assert torch.equal(tensor, torch.zeros(11))
        assert [name for name, _ in weights] == ["fast"]
    finally:
        weights.close()


@pytest.mark.parametrize(
    "option",
    [
        "prefetch_file_window",
        "io_threads_per_file",
        "block_size",
    ],
)
def test_streaming_rejects_invalid_limits(option):
    from vllm.model_executor.model_loader.weight_utils import (
        streaming_safetensors_weights_iterator,
    )

    with pytest.raises(ValueError, match="must be >= 1"):
        list(streaming_safetensors_weights_iterator([], False, **{option: 0}))


def test_prefetch_reads_every_byte_without_mapping(tmp_path, monkeypatch):
    """The prefetch path warms the page cache without parsing anything.

    It reads every byte of the shard, so the storage has the whole file
    ready, and it does not map the tensors: mapping is the loader's
    job, and doing it here would tie the memory down for nothing.
    """
    import threading
    from pathlib import Path

    import torch
    from safetensors.torch import save_file

    from vllm.model_executor.model_loader import weight_utils as wu

    paths = []
    for i in range(5):
        path = str(tmp_path / f"{i}.safetensors")
        save_file({f"weight_{i}": torch.arange(37)}, path)
        paths.append(path)
    reads: dict[str, list[tuple[int, int]]] = {path: [] for path in paths}
    handles = {}
    opened = []
    lock = threading.Lock()
    real_open = open
    real_pread = wu.os.pread
    # Exercise the portable pread path.
    monkeypatch.delattr(wu.os, "preadv", raising=False)

    def track_open(path, *args, **kwargs):
        handle = real_open(path, *args, **kwargs)
        with lock:
            handles[handle.fileno()] = path
            opened.append(handle)
        return handle

    def record_pread(fd, count, offset):
        data = real_pread(fd, count, offset)
        with lock:
            reads[handles[fd]].append((offset, len(data)))
        return data

    def map_after_read(*args, **kwargs):
        raise AssertionError("The prefetch path must not map tensors")

    monkeypatch.setattr(wu, "open", track_open, raising=False)
    monkeypatch.setattr(wu.os, "pread", record_pread)
    monkeypatch.setattr(wu, "BufferSafeTensors", map_after_read)

    assert wu.prefetch_checkpoints(
        paths, prefetch_file_window=2, io_threads_per_file=3, block_size=19
    ) == (2, 3)
    for path in paths:
        end = 0
        for offset, count in sorted(reads[path]):
            assert offset == end
            end += count
        assert end == Path(path).stat().st_size
    assert all(handle.closed for handle in opened)


def _safetensors_layout(path: str) -> dict[str, tuple[int, int]]:
    """Return where each tensor's bytes live in the file.

    The offsets are absolute file offsets, so a test can say which
    bytes a read touched.
    """
    import json
    import struct

    with open(path, "rb") as file:
        header_len = struct.unpack("<Q", file.read(8))[0]
        header = json.loads(file.read(header_len))
    data_start = 8 + header_len
    layout = {}
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        start, end = spec["data_offsets"]
        layout[name] = (data_start + start, end - start)
    return layout


def test_streaming_reads_each_kept_tensor_exactly_once(tmp_path, monkeypatch):
    """The reader fetches the bytes of the tensors it keeps, once each.

    It reads the tensors one by one into its own buffers instead of
    mapping the whole shard, so the bytes of the tensors it keeps are
    read exactly once and the bytes of the tensors it rejects are never
    read at all.
    """
    import threading

    import torch
    from safetensors.torch import save_file

    from vllm.model_executor.model_loader import threaded_buffer_reader as tbr
    from vllm.model_executor.model_loader import weight_utils as wu

    paths = []
    for i in range(5):
        path = str(tmp_path / f"{i}.safetensors")
        save_file(
            {f"weight_{i}a": torch.arange(37), f"weight_{i}b": torch.arange(11)}, path
        )
        paths.append(path)

    layout = {path: _safetensors_layout(path) for path in paths}
    reads: dict[str, list[tuple[int, int]]] = {path: [] for path in paths}
    fd_path: dict[int, str] = {}
    opened = []
    lock = threading.Lock()
    real_open = open
    real_pread = wu.os.pread
    # Exercise the portable pread path.
    monkeypatch.delattr(wu.os, "preadv", raising=False)

    def track_open(path, *args, **kwargs):
        handle = real_open(path, *args, **kwargs)
        with lock:
            fd_path[handle.fileno()] = path
            opened.append(handle)
        return handle

    def record_pread(fd, count, offset):
        data = real_pread(fd, count, offset)
        with lock:
            reads[fd_path[fd]].append((offset, len(data)))
        return data

    monkeypatch.setattr(tbr, "open", track_open, raising=False)
    monkeypatch.setattr(wu.os, "pread", record_pread)

    weights = dict(
        wu.streaming_safetensors_weights_iterator(
            paths,
            False,
            prefetch_file_window=2,
            io_threads_per_file=3,
            block_size=19,
        )
    )
    assert len(weights) == 2 * len(paths)
    for path in paths:
        data_start = min(offset for offset, _ in layout[path].values())
        data_reads = sorted(
            (offset, count) for offset, count in reads[path] if offset >= data_start
        )
        assert data_reads == sorted(layout[path].values())
    assert all(handle.closed for handle in opened)


def test_streaming_reads_only_the_tensors_it_is_asked_for(tmp_path, monkeypatch):
    """A filtered reader must not fetch the tensors it rejects.

    This is what keeps expert parallelism from multiplying the bytes
    the storage moves: a rank that owns a fraction of the experts reads
    only its own. The dense weights are not expert weights, so every
    rank keeps them.
    """
    import threading

    import torch
    from safetensors.torch import save_file

    from vllm.model_executor.model_loader import threaded_buffer_reader as tbr
    from vllm.model_executor.model_loader import weight_utils as wu

    path = str(tmp_path / "weights.safetensors")
    weights = {
        f"model.layers.0.mlp.experts.{i}.weight": torch.arange(64) for i in range(4)
    }
    weights["model.embed_tokens.weight"] = torch.arange(32)
    save_file(weights, path)

    layout = _safetensors_layout(path)
    reads: list[tuple[int, int]] = []
    fd_path: dict[int, str] = {}
    lock = threading.Lock()
    real_open = open
    real_pread = wu.os.pread
    monkeypatch.delattr(wu.os, "preadv", raising=False)

    def track_open(path, *args, **kwargs):
        handle = real_open(path, *args, **kwargs)
        with lock:
            fd_path[handle.fileno()] = path
        return handle

    def record_pread(fd, count, offset):
        data = real_pread(fd, count, offset)
        with lock:
            reads.append((offset, len(data)))
        return data

    monkeypatch.setattr(tbr, "open", track_open, raising=False)
    monkeypatch.setattr(wu.os, "pread", record_pread)

    result = dict(
        wu.safetensors_weights_iterator(
            [path],
            False,
            "streaming",
            local_expert_ids={1},
        )
    )
    assert set(result) == {
        "model.layers.0.mlp.experts.1.weight",
        "model.embed_tokens.weight",
    }
    assert torch.equal(result["model.layers.0.mlp.experts.1.weight"], torch.arange(64))

    # The bytes of the experts this rank does not own are never read.
    for name in (
        "model.layers.0.mlp.experts.0.weight",
        "model.layers.0.mlp.experts.2.weight",
        "model.layers.0.mlp.experts.3.weight",
    ):
        start, size = layout[name]
        for offset, count in reads:
            assert not (offset < start + size and start < offset + count), (
                f"read the bytes of {name}"
            )
