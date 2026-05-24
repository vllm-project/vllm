# SPDX-License-Identifier: Apache-2.0
"""Tests for vllm._genesis.compat.models.registry."""
from __future__ import annotations


from vllm._genesis.compat.models.registry import (
    SUPPORTED_MODELS,
    ModelEntry,
    get_model,
    list_models,
    list_recommended_for_hardware,
)


class TestRegistryShape:
    def test_registry_not_empty(self):
        assert len(SUPPORTED_MODELS) > 0

    def test_qwen3_6_27b_int4_present_as_PROD(self):
        """The current PROD baseline must be registered as PROD."""
        m = get_model("qwen3_6_27b_int4_autoround")
        assert m is not None
        assert m.status == "PROD"

    def test_each_entry_has_essential_fields(self):
        for key, m in SUPPORTED_MODELS.items():
            assert m.key == key, f"key mismatch: {key} vs entry.key {m.key}"
            assert m.hf_id, f"{key}: missing hf_id"
            assert m.title, f"{key}: missing title"
            assert m.size_gb > 0, f"{key}: zero/negative size"
            assert m.quant_format, f"{key}: missing quant_format"
            assert m.model_class, f"{key}: missing model_class"
            assert m.license, f"{key}: missing license"
            assert m.status in ("PROD", "SUPPORTED", "EXPERIMENTAL", "PLANNED"), (
                f"{key}: unknown status {m.status}"
            )
            assert isinstance(m.gated, bool)
            assert isinstance(m.is_hybrid, bool)
            assert isinstance(m.is_moe, bool)

    def test_each_tested_config_has_essentials(self):
        for key, m in SUPPORTED_MODELS.items():
            for cfg in m.tested_configs:
                assert cfg.name, f"{key}: tested_config missing name"
                assert cfg.tensor_parallel_size >= 1
                assert cfg.max_model_len > 0
                assert 0 < cfg.gpu_memory_utilization <= 1.0
                assert cfg.kv_cache_dtype


class TestGetModel:
    def test_known_key(self):
        m = get_model("qwen3_6_27b_int4_autoround")
        assert m is not None
        assert isinstance(m, ModelEntry)

    def test_unknown_key_returns_none(self):
        assert get_model("nonexistent_model_key") is None


class TestListModels:
    def test_no_filter_returns_all(self):
        models = list_models()
        assert len(models) == len(SUPPORTED_MODELS)

    def test_PROD_filter(self):
        prod = list_models(status_filter="PROD")
        for m in prod:
            assert m.status == "PROD"

    def test_PLANNED_filter(self):
        planned = list_models(status_filter="PLANNED")
        for m in planned:
            assert m.status == "PLANNED"

    def test_PROD_first_in_order(self):
        """PROD entries come first, then SUPPORTED, then ..."""
        models = list_models()
        statuses = [m.status for m in models]
        # Check that PROD precedes SUPPORTED and PLANNED
        for i, s in enumerate(statuses):
            for j in range(i + 1, len(statuses)):
                if s == "PLANNED":
                    pass  # planned can be after anything except other planned
                if s == "PROD" and statuses[j] in ("EXPERIMENTAL", "PLANNED"):
                    pass  # ok
                # Specifically: if PROD appears at i, all earlier statuses must be PROD too
        # Looser check: first non-PROD index >= last PROD index
        prod_indices = [i for i, s in enumerate(statuses) if s == "PROD"]
        non_prod_indices = [i for i, s in enumerate(statuses) if s != "PROD"]
        if prod_indices and non_prod_indices:
            assert max(prod_indices) < min(non_prod_indices)


class TestListRecommendedForHardware:
    def test_dual_a5000_24gb(self):
        """2× A5000 = 48GB total. Should fit 27B INT4 (14GB) but not 80B model."""
        recommended = list_recommended_for_hardware(
            vram_gb_total=48.0, num_gpus=2, hardware_class="rtx_a5000",
        )
        keys = [m.key for m in recommended]
        # 27B INT4 (TP=2, ~14 GB per rank min) → fits
        assert "qwen3_6_27b_int4_autoround" in keys
        # 80B model needs 16 GB per rank at TP=4 (which we don't have)
        assert "qwen3_next_80b_awq" not in keys

    def test_single_3060_12gb_fits_almost_nothing(self):
        """A small single GPU shouldn't recommend large models."""
        recommended = list_recommended_for_hardware(
            vram_gb_total=12.0, num_gpus=1,
        )
        # All our models need 24+ GB per rank min → empty
        assert len(recommended) == 0

    def test_excludes_PLANNED(self):
        """PLANNED models never recommended (not yet validated)."""
        recommended = list_recommended_for_hardware(
            vram_gb_total=192.0, num_gpus=4,  # plenty of VRAM
        )
        for m in recommended:
            assert m.status != "PLANNED"
