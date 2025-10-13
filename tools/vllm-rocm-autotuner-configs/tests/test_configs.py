# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate all config files."""

import json
from pathlib import Path

import pytest


def load_all_configs(configs_dir):
    """Load all config files."""
    configs = []
    for config_file in configs_dir.glob("rocm_config_*.json"):
        with open(config_file) as f:
            data = json.load(f)
            configs.append((config_file.name, data))
    return configs


class TestConfigStructure:
    """Test config file structure."""

    @pytest.mark.parametrize(
        "config_name,config_data",
        load_all_configs(
            Path(__file__).parent.parent / "src" /
            "vllm_rocm_autotuner_configs" / "configs"),
    )
    def test_valid_json(self, config_name, config_data):
        """Test that configs are valid JSON."""
        assert isinstance(config_data, dict)
        assert "model_configs" in config_data

    @pytest.mark.parametrize(
        "config_name,config_data",
        load_all_configs(
            Path(__file__).parent.parent / "src" /
            "vllm_rocm_autotuner_configs" / "configs"),
    )
    def test_model_configs_structure(self, config_name, config_data):
        """Test model_configs section structure."""
        model_configs = config_data["model_configs"]

        for model_id, model_data in model_configs.items():
            # Must have recipes
            if "recipes" in model_data:
                recipes = model_data["recipes"]
                assert isinstance(recipes, list)
                assert len(recipes) > 0

                for recipe in recipes:
                    # Required fields
                    assert "name" in recipe
                    assert "rank" in recipe
                    assert isinstance(recipe["rank"], int)

                    # Must have at least one config type
                    assert "env_vars" in recipe or "cli_args" in recipe

    @pytest.mark.parametrize(
        "config_name,config_data",
        load_all_configs(
            Path(__file__).parent.parent / "src" /
            "vllm_rocm_autotuner_configs" / "configs"),
    )
    def test_signature_format(self, config_name, config_data):
        """Test signature format is correct."""
        for model_id, model_data in config_data["model_configs"].items():
            if "signature" in model_data:
                sig = model_data["signature"]
                parts = sig.split("_")

                assert len(parts) == 4, f"Signature must have 4 parts: {sig}"
                assert parts[1].endswith("L"), f"Part 2 must end with L: {sig}"
                assert parts[2].endswith("H"), f"Part 3 must end with H: {sig}"
                assert parts[3].endswith("A"), f"Part 4 must end with A: {sig}"

                # Extract numbers
                try:
                    int(parts[1][:-1])
                    int(parts[2][:-1])
                    int(parts[3][:-1])
                except ValueError:
                    pytest.fail(f"Invalid signature numbers: {sig}")

    @pytest.mark.parametrize(
        "config_name,config_data",
        load_all_configs(
            Path(__file__).parent.parent / "src" /
            "vllm_rocm_autotuner_configs" / "configs"),
    )
    def test_recipes_ranked(self, config_name, config_data):
        """Test that recipes are properly ranked."""
        for model_id, model_data in config_data["model_configs"].items():
            recipes = model_data.get("recipes", [])

            if len(recipes) > 1:
                ranks = [r["rank"] for r in recipes]
                # Ranks should be unique
                assert len(ranks) == len(
                    set(ranks)), f"{model_id}: duplicate ranks"
                # Should have rank 1
                assert 1 in ranks, f"{model_id}: missing rank 1 recipe"

    @pytest.mark.parametrize(
        "config_name,config_data",
        load_all_configs(
            Path(__file__).parent.parent / "src" /
            "vllm_rocm_autotuner_configs" / "configs"),
    )
    def test_env_vars_valid(self, config_name, config_data):
        """Test that env vars are valid."""
        for model_id, model_data in config_data["model_configs"].items():
            for recipe in model_data.get("recipes", []):
                env_vars = recipe.get("env_vars", {})

                for key, value in env_vars.items():
                    # Keys should be uppercase
                    assert key.isupper() or key.startswith("VLLM_"), (
                        f"{model_id}/{recipe['name']}: "
                        f"env var {key} should be uppercase")

                    # Values should be strings or convertible
                    assert isinstance(
                        value,
                        (str, int, float)), (f"{model_id}/{recipe['name']}: "
                                             f"env var {key} has invalid type")

    @pytest.mark.parametrize(
        "config_name,config_data",
        load_all_configs(
            Path(__file__).parent.parent / "src" /
            "vllm_rocm_autotuner_configs" / "configs"),
    )
    def test_cli_args_valid(self, config_name, config_data):
        """Test that CLI args are valid."""
        for model_id, model_data in config_data["model_configs"].items():
            for recipe in model_data.get("recipes", []):
                cli_args = recipe.get("cli_args", {})

                for key, value in cli_args.items():
                    # Keys should use hyphens, not underscores
                    assert "_" not in key or key.startswith("_"), (
                        f"{model_id}/{recipe['name']}: "
                        f"CLI arg {key} should use hyphens")

                    # Values should be valid types
                    assert isinstance(
                        value, (str, int, float, bool,
                                dict)), (f"{model_id}/{recipe['name']}: "
                                         f"CLI arg {key} has invalid type")
