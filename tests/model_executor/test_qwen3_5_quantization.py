# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock, patch


def test_qwen3_5_lm_head_receives_quant_config():
    from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLMBase

    mock_quant_config = Mock()

    mock_hf_config = Mock()
    mock_hf_config.tie_word_embeddings = False
    mock_hf_config.vocab_size = 128
    mock_hf_config.hidden_size = 64

    mock_vllm_config = Mock()
    mock_vllm_config.model_config.hf_text_config = mock_hf_config
    mock_vllm_config.cache_config.mamba_cache_mode = "align"
    mock_vllm_config.scheduler_config = Mock()
    mock_vllm_config.quant_config = mock_quant_config
    mock_vllm_config.lora_config = None

    mock_pp_group = Mock()
    mock_pp_group.is_last_rank = True

    with (
        patch("vllm.model_executor.models.qwen3_5.Qwen3_5Model") as MockModel,
        patch("vllm.model_executor.models.qwen3_5.ParallelLMHead") as MockLMHead,
        patch("vllm.model_executor.models.qwen3_5.LogitsProcessor"),
        patch(
            "vllm.model_executor.models.qwen3_5.get_pp_group",
            return_value=mock_pp_group,
        ),
    ):
        MockModel.return_value.make_empty_intermediate_tensors = Mock()

        Qwen3_5ForCausalLMBase(vllm_config=mock_vllm_config)

        MockLMHead.assert_called_once()
        call_kwargs = MockLMHead.call_args.kwargs
        assert call_kwargs["quant_config"] is mock_quant_config


def test_qwen3_5_mtp_lm_head_receives_quant_config():
    from vllm.config import CompilationMode
    from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MTP

    mock_quant_config = Mock()

    mock_hf_config = Mock()
    mock_hf_config.tie_word_embeddings = False
    mock_hf_config.vocab_size = 128
    mock_hf_config.hidden_size = 64

    mock_vllm_config = Mock()
    mock_vllm_config.model_config.hf_text_config = mock_hf_config
    mock_vllm_config.cache_config.mamba_cache_mode = "align"
    mock_vllm_config.compilation_config.mode = CompilationMode.NONE
    mock_vllm_config.quant_config = mock_quant_config

    mock_pp_group = Mock()
    mock_pp_group.is_last_rank = True

    with (
        patch("vllm.model_executor.models.qwen3_5_mtp.Qwen3_5MultiTokenPredictor"),
        patch("vllm.model_executor.models.qwen3_5_mtp.ParallelLMHead") as MockLMHead,
        patch("vllm.model_executor.models.qwen3_5_mtp.LogitsProcessor"),
        patch(
            "vllm.model_executor.models.qwen3_5_mtp.get_pp_group",
            return_value=mock_pp_group,
        ),
    ):
        Qwen3_5MTP(vllm_config=mock_vllm_config)

        MockLMHead.assert_called_once()
        call_kwargs = MockLMHead.call_args.kwargs
        assert call_kwargs["quant_config"] is mock_quant_config


def _make_per_tensor_scale_param(num_shards: int):
    """Build a real ``PerTensorScaleParameter`` without a distributed env.

    ``BasevLLMParameter.__init__`` queries the TP rank/size, so patch those to
    the single-rank values while constructing the parameter.
    """
    import torch

    from vllm.model_executor.parameter import PerTensorScaleParameter

    with (
        patch(
            "vllm.model_executor.parameter.get_tensor_model_parallel_rank",
            return_value=0,
        ),
        patch(
            "vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
            return_value=1,
        ),
    ):
        return PerTensorScaleParameter(
            data=torch.zeros(num_shards, dtype=torch.float32),
            weight_loader=lambda *a, **k: None,
        )


def test_per_tensor_scale_tuple_shard_slices_per_shard():
    """A fused disk weight (e.g. the Gated-Delta-Net ``in_proj_qkv`` in
    Qwen3.5 / Qwen3-Next) may ship one per-tensor scale per fused shard, i.e. a
    ``[q, k, v]`` vector loaded via the ``(0, 1, 2)`` tuple shard id. The
    ``MergedColumnParallelLinear`` scale loader must slice it per shard rather
    than broadcasting the whole vector (which previously tripped
    ``assert loaded_weight.shape[0] == 1``).
    """
    import torch

    from vllm.model_executor.layers.linear import MergedColumnParallelLinear

    # 4 logical scales: q, k, v, z (Qwen3.5 in_proj_qkvz layout).
    param = _make_per_tensor_scale_param(4)
    loaded = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

    fake_self = Mock()
    MergedColumnParallelLinear.weight_loader_v2(
        fake_self, param, loaded, loaded_shard_id=(0, 1, 2)
    )

    # q/k/v receive their own scale; z (shard 3) is left untouched.
    assert param.data.tolist() == [1.0, 2.0, 3.0, 0.0]


def test_per_tensor_scale_tuple_shard_broadcasts_single_scalar():
    """Regression: a single shared scalar scale (e.g. Phi-3's on-disk-fused
    ``gate_up_proj``) still broadcasts to every listed shard.
    """
    import torch

    from vllm.model_executor.layers.linear import MergedColumnParallelLinear

    param = _make_per_tensor_scale_param(2)
    loaded = torch.tensor(5.0, dtype=torch.float32)  # 0-D scalar

    fake_self = Mock()
    MergedColumnParallelLinear.weight_loader_v2(
        fake_self, param, loaded, loaded_shard_id=(0, 1)
    )

    assert param.data.tolist() == [5.0, 5.0]
