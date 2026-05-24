# SPDX-License-Identifier: Apache-2.0
"""Tests for `genesis recipe adopt URL` — pull a recipe from a URL,
validate it, save it locally.

Completes the recipe sharing loop: Sander saves his v794 PROD recipe
+ pushes the JSON to a public gist or repo, a community user runs
`genesis recipe adopt <url>`, and they get an identical launch
configuration in one command.

Security model
──────────────
- HTTPS only by default (HTTP refused unless --allow-http given)
- File size capped (default 100KB — recipes are small)
- Schema validated BEFORE saving (no garbage hits disk)
- Recipe-name validation reuses the same regex as `recipe save` so
  path traversal is impossible
"""
from __future__ import annotations

import json

import pytest


_VALID_RECIPE = {
    "genesis_recipe_version": "1.0",
    "name": "adopted-recipe",
    "description": "Pulled from URL",
    "envs": {"GENESIS_ENABLE_PN14_TQ_DECODE_OOB_CLAMP": "1"},
    "vllm_serve": {"model": "/models/X", "tensor_parallel_size": 2},
    "container": {"image": "vllm/vllm-openai:nightly", "name": "test"},
}


@pytest.fixture
def tmp_recipes_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("GENESIS_RECIPES_DIR", str(tmp_path / "recipes"))
    return tmp_path / "recipes"


@pytest.fixture
def fake_url_fetcher(monkeypatch):
    """Mock the HTTP fetch — returns whatever the test sets in the dict."""
    state = {"body": json.dumps(_VALID_RECIPE), "size": None}
    def fake_fetch(url, max_bytes):
        if state.get("error"):
            raise RuntimeError(state["error"])
        body = state["body"]
        if state.get("size") and len(body) > state["size"]:
            raise RuntimeError(f"recipe exceeded {state['size']} byte limit")
        return body
    from vllm._genesis.compat import recipes
    monkeypatch.setattr(recipes, "_fetch_url_body", fake_fetch)
    return state


# ─── URL validation ─────────────────────────────────────────────────────


class TestURLValidation:
    def test_https_url_accepted(self, tmp_recipes_dir, fake_url_fetcher):
        from vllm._genesis.compat.recipes import adopt_recipe
        result = adopt_recipe("https://gist.example.com/foo.json", "imported")
        assert result is not None
        assert result["name"] == "imported"

    def test_http_url_refused_by_default(self, tmp_recipes_dir, fake_url_fetcher):
        from vllm._genesis.compat.recipes import adopt_recipe
        with pytest.raises(ValueError) as ex:
            adopt_recipe("http://example.com/recipe.json", "imported")
        assert "https" in str(ex.value).lower() or "http" in str(ex.value).lower()

    def test_http_with_allow_flag(self, tmp_recipes_dir, fake_url_fetcher):
        from vllm._genesis.compat.recipes import adopt_recipe
        result = adopt_recipe("http://example.com/recipe.json", "imported",
                              allow_http=True)
        assert result is not None

    def test_non_url_string_refused(self, tmp_recipes_dir):
        from vllm._genesis.compat.recipes import adopt_recipe
        with pytest.raises(ValueError):
            adopt_recipe("not a url", "imported")
        with pytest.raises(ValueError):
            adopt_recipe("ftp://server/recipe", "imported")
        with pytest.raises(ValueError):
            adopt_recipe("", "imported")


# ─── Body validation ────────────────────────────────────────────────────


class TestBodyValidation:
    def test_invalid_json_refused(
        self, tmp_recipes_dir, fake_url_fetcher,
    ):
        fake_url_fetcher["body"] = "not valid json {{"
        from vllm._genesis.compat.recipes import adopt_recipe
        with pytest.raises(ValueError):
            adopt_recipe("https://example.com/recipe.json", "imported")

    def test_recipe_must_validate_schema(
        self, tmp_recipes_dir, fake_url_fetcher,
    ):
        # Empty dict is invalid (missing required fields)
        fake_url_fetcher["body"] = json.dumps({"name": ""})
        from vllm._genesis.compat.recipes import adopt_recipe
        with pytest.raises(ValueError) as ex:
            adopt_recipe("https://example.com/recipe.json", "imported")
        assert "validation" in str(ex.value).lower() or \
               "schema" in str(ex.value).lower() or \
               "name" in str(ex.value).lower()

    def test_oversized_body_refused(
        self, tmp_recipes_dir, fake_url_fetcher,
    ):
        # Cap the fetch to 50 bytes, body is larger
        fake_url_fetcher["body"] = json.dumps(_VALID_RECIPE)
        fake_url_fetcher["size"] = 50
        from vllm._genesis.compat.recipes import adopt_recipe
        with pytest.raises(RuntimeError):
            adopt_recipe("https://example.com/recipe.json", "imported")


# ─── Adoption persists to local recipes dir ─────────────────────────────


class TestAdoption:
    def test_adopted_saved_with_target_name(
        self, tmp_recipes_dir, fake_url_fetcher,
    ):
        from vllm._genesis.compat.recipes import adopt_recipe, list_names
        adopt_recipe("https://example.com/recipe.json", "my-imported")
        assert "my-imported" in list_names()

    def test_adopt_overwrites_existing(
        self, tmp_recipes_dir, fake_url_fetcher,
    ):
        from vllm._genesis.compat.recipes import (
            adopt_recipe, save, load,
        )
        save("dup-name", _VALID_RECIPE)
        # Adopting with the same name should overwrite
        fake_url_fetcher["body"] = json.dumps(
            {**_VALID_RECIPE, "description": "from URL"}
        )
        adopt_recipe("https://example.com/recipe.json", "dup-name")
        loaded = load("dup-name")
        assert loaded["description"] == "from URL"

    def test_adopt_records_origin_url(
        self, tmp_recipes_dir, fake_url_fetcher,
    ):
        """Adopted recipe should record the URL it was pulled from
        in a `_adopted_from` field for provenance."""
        from vllm._genesis.compat.recipes import adopt_recipe, load
        adopt_recipe("https://example.com/recipe.json", "tracked")
        rec = load("tracked")
        assert "_adopted_from" in rec
        assert "example.com" in rec["_adopted_from"]


# ─── CLI ─────────────────────────────────────────────────────────────────


class TestCLI:
    def test_adopt_subcommand_routes(
        self, tmp_recipes_dir, fake_url_fetcher, capsys,
    ):
        from vllm._genesis.compat.recipes import main
        rc = main(["adopt", "https://example.com/recipe.json",
                    "from-cli"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "from-cli" in captured.out

    def test_adopt_invalid_url_returns_nonzero(
        self, tmp_recipes_dir, capsys,
    ):
        from vllm._genesis.compat.recipes import main
        rc = main(["adopt", "ftp://nope/recipe", "x"])
        assert rc != 0

    def test_adopt_propagates_validation_failure(
        self, tmp_recipes_dir, fake_url_fetcher, capsys,
    ):
        fake_url_fetcher["body"] = "{malformed"
        from vllm._genesis.compat.recipes import main
        rc = main(["adopt", "https://example.com/r.json", "x"])
        assert rc != 0


# ─── Integration with `recipe show` after adopt ─────────────────────────


class TestPostAdoptInspect:
    def test_show_after_adopt(self, tmp_recipes_dir, fake_url_fetcher, capsys):
        from vllm._genesis.compat.recipes import main
        main(["adopt", "https://example.com/recipe.json", "post-adopt"])
        capsys.readouterr()  # clear
        rc = main(["show", "post-adopt"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "post-adopt" in captured.out
