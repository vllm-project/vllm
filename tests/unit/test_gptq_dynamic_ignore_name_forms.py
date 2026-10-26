import re

from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQConfig
from vllm.model_executor.layers.quantization.utils.gptq_utils import (
    get_dynamic_override,
    _dynamic_name_variants,
)


def _cfg(dynamic):
    return AutoGPTQConfig.from_config({
        "bits": 4,
        "group_size": 128,
        "sym": True,
        "desc_act": False,
        "dynamic": dynamic,
    })


CHECKPOINT_PREFIX = "model.language_model."
MAPPED_PREFIX = "model."
MODULE_PREFIX = "language_model.model."
TAIL = "layers.43.self_attn.qkv_proj"

checkpoint_name = CHECKPOINT_PREFIX + TAIL
mapped_name = MAPPED_PREFIX + TAIL
module_name = MODULE_PREFIX + TAIL


def test_name_variants_cover_all_three_forms():
    for name in (checkpoint_name, mapped_name, module_name):
        variants = set(_dynamic_name_variants(name))
        assert checkpoint_name in variants, name
        assert mapped_name in variants, name
        assert module_name in variants, name


def test_checkpoint_form_pattern_matches_module_tree_name():
    cfg = _cfg({"-:" + re.escape(checkpoint_name): {}})
    assert get_dynamic_override(cfg, layer_name=module_name) is False


def test_module_tree_form_pattern_matches_checkpoint_name():
    cfg = _cfg({"-:" + re.escape(module_name): {}})
    assert get_dynamic_override(cfg, layer_name=checkpoint_name) is False


def test_suffix_pattern_still_matches_all_forms():
    cfg = _cfg({"-:.*layers\\.43\\.self_attn\\.qkv_proj$": {}})
    for name in (checkpoint_name, mapped_name, module_name):
        assert get_dynamic_override(cfg, layer_name=name) is False


def test_re_match_anchor_pattern_matches_via_variants():
    cfg = _cfg({"-:layers\\.43\\.self_attn\\.qkv_proj$": {}})
    for name in (checkpoint_name, mapped_name, module_name):
        assert get_dynamic_override(cfg, layer_name=name) is False


def test_unmatched_layer_keeps_default():
    cfg = _cfg({"-:.*layers\\.42\\.self_attn\\.qkv_proj$": {}})
    assert get_dynamic_override(cfg, layer_name=module_name) is None
    assert get_dynamic_override(cfg, layer_name=module_name, default_value=True) is True


def test_positive_override_still_matches_first_form():
    cfg = _cfg({"+:.*layers\\.43\\.self_attn\\.qkv_proj$": {"bits": 8}})
    assert get_dynamic_override(cfg, layer_name=module_name, key="bits") == 8
    assert get_dynamic_override(cfg, layer_name=checkpoint_name, key="bits") == 8
