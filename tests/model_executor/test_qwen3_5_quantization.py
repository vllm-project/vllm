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


def test_qwen3_5_mtp_registered_class_exposes_hf_to_vllm_mapper():
    """The registered (top-level) MTP draft class must expose ``hf_to_vllm_mapper``.

    ``configure_quant_config`` resolves the mapper via
    ``getattr(model_class, "hf_to_vllm_mapper", None)`` on the *registered* class
    (see vllm/model_executor/model_loader/utils.py). The target model class
    exposes one (inherited from Qwen3VLForConditionalGeneration), so its
    checkpoint per-layer quant lists (e.g. INC/AutoRound ``block_name_to_quantize``)
    get mapped from HF names to vLLM runtime names. If the draft class only
    carries the mapper on its inner predictor submodule, ``getattr`` returns
    ``None`` and the draft's quant lists are left in HF-name space -- so every
    MTP layer silently resolves Unquantized. Guard the symmetry here.
    """
    from vllm.model_executor.models.qwen3_5_mtp import (
        Qwen3_5MoeMTP,
        Qwen3_5MTP,
        Qwen3_5MultiTokenPredictor,
    )

    for cls in (Qwen3_5MTP, Qwen3_5MoeMTP):
        mapper = getattr(cls, "hf_to_vllm_mapper", None)
        assert mapper is not None, (
            f"{cls.__name__} must expose hf_to_vllm_mapper so that "
            "configure_quant_config maps its per-layer quantization config"
        )
        assert mapper is Qwen3_5MultiTokenPredictor.hf_to_vllm_mapper


def test_configure_quant_config_maps_mtp_draft_layer_names():
    """``configure_quant_config`` applies the mapper for the MTP draft class.

    This is the behavioral change: because ``Qwen3_5MTP`` now exposes
    ``hf_to_vllm_mapper``, ``configure_quant_config`` calls
    ``apply_vllm_mapper`` on the quant config (mapping its ``block_name_to_quantize``
    from HF to vLLM names), exactly as it does for the target model. A class
    without a mapper leaves the quant config untouched.
    """
    from vllm.model_executor.model_loader.utils import configure_quant_config
    from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MTP

    quant_config = Mock()
    configure_quant_config(quant_config, Qwen3_5MTP)
    quant_config.apply_vllm_mapper.assert_called_once()

    # Contrast: a class with no mapper must not trigger mapping (regression guard
    # documenting the exact pre-fix behavior that silently dropped draft quant).
    class _NoMapper:
        pass

    quant_config_no_mapper = Mock()
    configure_quant_config(quant_config_no_mapper, _NoMapper)
    quant_config_no_mapper.apply_vllm_mapper.assert_not_called()


def test_inc_resolver_quantizes_mapped_draft_prefixed_layer():
    """End-to-end (pure-python) resolution: a draft-prefixed layer resolves to a
    quantized config only when the ``block_name_to_quantize`` prefix is in vLLM
    runtime-name space.

    INC/AutoRound marks quantized layers by prefix (``layer_name.startswith(name)``
    in INCConfigParser). At runtime the MTP layers are named ``mtp.layers.*``
    (vLLM space). The checkpoint ships the prefix in HF space (``model.mtp.layers``);
    only after ``hf_to_vllm_mapper`` rewrites it to ``mtp.layers`` does the draft
    layer match and resolve quantized. This test pins that resolution behavior,
    which is the mechanism the fix restores for the draft path.
    """
    from types import SimpleNamespace

    from vllm.model_executor.layers.quantization.inc.config_parser import (
        INCConfigParser,
    )

    def _parser(block_names):
        cfg = SimpleNamespace(
            block_name_to_quantize=block_names,
            extra_config=None,
            weight_bits=4,
            group_size=128,
            sym=True,
            packing_format="gptq",
            backend="gptq",
            data_type="int",
            packed_modules_mapping={},
        )
        return INCConfigParser(cfg)

    layer = object()  # not a ParallelLMHead / FusedMoE
    draft_layer_name = "mtp.layers.0.self_attn.qkv_proj"

    # vLLM-name prefix (post-mapper): draft layer is recognized as quantized.
    mapped = _parser(["mtp.layers"]).resolve(layer, draft_layer_name)
    assert mapped.quantized is True
    assert mapped.bits == 4

    # HF-name prefix (mapper not applied -- the bug): draft layer is missed and
    # silently resolves Unquantized.
    unmapped = _parser(["model.mtp.layers"]).resolve(layer, draft_layer_name)
    assert unmapped.quantized is False
    assert unmapped.bits == 16
