# SPDX-License-Identifier: Apache-2.0
"""Tests for vllm._genesis.compat.recipes — `genesis recipe` system.

A recipe captures everything needed to reproduce a Genesis launch:
  - hardware target (model_key + tp)
  - container settings (image, name, ports, mounts)
  - env variables (Genesis env flags + system env)
  - vllm serve command-line args
  - expected metrics (for regression detection)
  - human-readable notes / quirks

Recipes are stored as JSON at ~/.genesis/recipes/<name>.json (default),
override via GENESIS_RECIPES_DIR env. Format is intentionally JSON
(not YAML) to stay dependency-free — Genesis ships zero runtime
dependencies beyond what vllm itself requires.

CLI:
  genesis recipe save <name> --from-container <docker-name>
  genesis recipe list
  genesis recipe show <name>
  genesis recipe load <name>             # writes a launch shell script
  genesis recipe delete <name>
"""
from __future__ import annotations

import json

import pytest


# ─── Synthetic recipe + fixtures ────────────────────────────────────────


_VALID_RECIPE = {
    "genesis_recipe_version": "1.0",
    "name": "test-prod",
    "description": "Test recipe for unit tests",
    "created_at": "2026-04-30T00:00:00Z",
    "created_by": "test",
    "target": {
        "hardware_class": "rtx_a5000_x2",
        "model_key": "qwen3_6_27b_int4_autoround",
        "vllm_pin": "0.20.1rc1.dev16+g7a1eb8ac2",
    },
    "container": {
        "image": "vllm/vllm-openai:nightly",
        "name": "test-container",
        "ports": [8000],
        "shm_size": "8g",
        "memory": "64g",
    },
    "envs": {
        "GENESIS_ENABLE_PN14_TQ_DECODE_OOB_CLAMP": "1",
        "GENESIS_ENABLE_PN12_FFN_INTERMEDIATE_POOL": "1",
    },
    "vllm_serve": {
        "model": "/models/Qwen3.6-27B-int4-AutoRound",
        "tensor_parallel_size": 2,
        "max_model_len": 280000,
        "kv_cache_dtype": "turboquant_k8v4",
    },
    "expected_metrics": {
        "wall_tps": 103.3,
        "cv": 0.049,
    },
    "notes": "Test note.",
}


@pytest.fixture
def tmp_recipes_dir(tmp_path, monkeypatch):
    """Redirect recipe storage to a temp dir for hermetic tests."""
    monkeypatch.setenv("GENESIS_RECIPES_DIR", str(tmp_path / "recipes"))
    return tmp_path / "recipes"


# ─── Storage / round-trip ───────────────────────────────────────────────


class TestStorage:
    def test_save_creates_file(self, tmp_recipes_dir):
        from vllm._genesis.compat.recipes import save
        save("test-prod", _VALID_RECIPE)
        assert (tmp_recipes_dir / "test-prod.json").is_file()

    def test_save_then_load_roundtrip(self, tmp_recipes_dir):
        from vllm._genesis.compat.recipes import save, load
        save("rt-test", _VALID_RECIPE)
        loaded = load("rt-test")
        # save() normalizes name to the file-name parameter — a recipe
        # saved as "rt-test" has name="rt-test" inside, regardless of
        # what name field was in the input dict.
        assert loaded["name"] == "rt-test"
        # Other fields preserved
        assert loaded["envs"] == _VALID_RECIPE["envs"]
        assert loaded["target"] == _VALID_RECIPE["target"]

    def test_load_unknown_returns_None(self, tmp_recipes_dir):
        from vllm._genesis.compat.recipes import load
        result = load("does-not-exist")
        assert result is None

    def test_list_names_empty(self, tmp_recipes_dir):
        from vllm._genesis.compat.recipes import list_names
        assert list_names() == []

    def test_list_names_after_save(self, tmp_recipes_dir):
        from vllm._genesis.compat.recipes import save, list_names
        save("a", _VALID_RECIPE)
        save("b", _VALID_RECIPE)
        save("c", _VALID_RECIPE)
        assert sorted(list_names()) == ["a", "b", "c"]

    def test_delete_removes_file(self, tmp_recipes_dir):
        from vllm._genesis.compat.recipes import save, delete, list_names
        save("doomed", _VALID_RECIPE)
        assert "doomed" in list_names()
        assert delete("doomed") is True
        assert "doomed" not in list_names()

    def test_delete_unknown_returns_false(self, tmp_recipes_dir):
        from vllm._genesis.compat.recipes import delete
        assert delete("nonexistent") is False

    def test_recipe_name_validation_rejects_path_traversal(
        self, tmp_recipes_dir,
    ):
        """Recipe names must not contain path separators / .. — would
        let an attacker write outside the recipes dir."""
        from vllm._genesis.compat.recipes import save
        for bad in ("../etc/passwd", "/abs/path", "name/with/slash",
                    "name\\with\\backslash"):
            with pytest.raises((ValueError, OSError)):
                save(bad, _VALID_RECIPE)


# ─── Recipe shape validation ────────────────────────────────────────────


class TestValidate:
    def test_valid_recipe_passes(self):
        from vllm._genesis.compat.recipes import validate_recipe
        issues = validate_recipe(_VALID_RECIPE)
        assert issues == []

    def test_missing_name_fails(self):
        from vllm._genesis.compat.recipes import validate_recipe
        bad = {k: v for k, v in _VALID_RECIPE.items() if k != "name"}
        issues = validate_recipe(bad)
        assert any("name" in i.lower() for i in issues)

    def test_unknown_genesis_env_warns(self):
        from vllm._genesis.compat.recipes import validate_recipe
        bad = dict(_VALID_RECIPE)
        bad["envs"] = {"GENESIS_ENABLE_FAKE_PATCH_99999": "1", **bad["envs"]}
        issues = validate_recipe(bad)
        # Must surface the unknown env (warning or error)
        assert any("FAKE_PATCH" in i for i in issues)

    def test_validate_returns_list(self):
        from vllm._genesis.compat.recipes import validate_recipe
        out = validate_recipe(_VALID_RECIPE)
        assert isinstance(out, list)


# ─── Container introspection ────────────────────────────────────────────


class TestFromContainer:
    def test_from_container_calls_docker_inspect(self, monkeypatch, tmp_recipes_dir):
        """from_container() should shell out to `docker inspect <name>`
        and parse the JSON. Mock it for hermetic test."""
        fake_inspect = json.dumps([{
            "Name": "/test-vllm",
            "Config": {
                "Image": "vllm/vllm-openai:nightly",
                "Env": [
                    "GENESIS_ENABLE_PN14_TQ_DECODE_OOB_CLAMP=1",
                    "VLLM_NO_USAGE_STATS=1",
                ],
                "Cmd": ["-c", "exec vllm serve --model /models/X "
                              "--tensor-parallel-size 2 "
                              "--max-model-len 280000 "
                              "--kv-cache-dtype turboquant_k8v4"],
            },
            "HostConfig": {
                "Binds": ["/nfs/genesis/models:/models:ro"],
                "ShmSize": 8589934592,
                "Memory": 68719476736,
                "PortBindings": {"8000/tcp": [{"HostPort": "8000"}]},
            },
        }])
        from vllm._genesis.compat import recipes
        monkeypatch.setattr(recipes, "_docker_inspect", lambda n: fake_inspect)
        rec = recipes.from_container("test-vllm")
        assert rec is not None
        assert "envs" in rec
        assert rec["envs"].get("GENESIS_ENABLE_PN14_TQ_DECODE_OOB_CLAMP") == "1"

    def test_from_container_extracts_vllm_serve_args(self, monkeypatch, tmp_recipes_dir):
        fake_inspect = json.dumps([{
            "Name": "/test-vllm",
            "Config": {
                "Image": "vllm/vllm-openai:nightly",
                "Env": [],
                "Cmd": ["-c", "exec vllm serve "
                              "--model /models/X "
                              "--tensor-parallel-size 2 "
                              "--max-model-len 280000"],
            },
            "HostConfig": {"Binds": [], "ShmSize": 0, "Memory": 0,
                            "PortBindings": {}},
        }])
        from vllm._genesis.compat import recipes
        monkeypatch.setattr(recipes, "_docker_inspect", lambda n: fake_inspect)
        rec = recipes.from_container("test-vllm")
        # Either nested vllm_serve dict OR raw command preserved
        assert (
            rec["vllm_serve"].get("model") == "/models/X"
            or "/models/X" in str(rec.get("vllm_command", ""))
        )

    def test_from_container_unknown_returns_None(self, monkeypatch, tmp_recipes_dir):
        from vllm._genesis.compat import recipes
        # Simulate docker inspect returning empty / failing
        monkeypatch.setattr(recipes, "_docker_inspect",
                            lambda n: (_ for _ in ()).throw(
                                RuntimeError("no such container")))
        rec = recipes.from_container("does-not-exist")
        assert rec is None


# ─── Launch script generation ───────────────────────────────────────────


class TestToLaunchScript:
    def test_generate_bash_string(self, tmp_recipes_dir):
        from vllm._genesis.compat.recipes import to_launch_script
        script = to_launch_script(_VALID_RECIPE)
        assert isinstance(script, str)
        assert script.startswith("#!/")
        assert "docker run" in script
        assert "test-container" in script
        assert "vllm/vllm-openai:nightly" in script

    def test_includes_genesis_envs(self, tmp_recipes_dir):
        from vllm._genesis.compat.recipes import to_launch_script
        script = to_launch_script(_VALID_RECIPE)
        assert "GENESIS_ENABLE_PN14_TQ_DECODE_OOB_CLAMP=1" in script
        assert "GENESIS_ENABLE_PN12_FFN_INTERMEDIATE_POOL=1" in script

    def test_includes_vllm_serve_args(self, tmp_recipes_dir):
        from vllm._genesis.compat.recipes import to_launch_script
        script = to_launch_script(_VALID_RECIPE)
        assert "--tensor-parallel-size 2" in script
        assert "--max-model-len 280000" in script
        assert "--kv-cache-dtype turboquant_k8v4" in script

    def test_includes_metadata_comment(self, tmp_recipes_dir):
        """Generated script should have a comment with recipe name +
        creation date so operators can trace where it came from."""
        from vllm._genesis.compat.recipes import to_launch_script
        script = to_launch_script(_VALID_RECIPE)
        assert "test-prod" in script
        # creation timestamp included
        assert "2026-04-30" in script or _VALID_RECIPE["created_at"] in script


# ─── CLI ─────────────────────────────────────────────────────────────────


class TestCLI:
    def test_list_subcommand_empty(self, tmp_recipes_dir, capsys):
        from vllm._genesis.compat.recipes import main
        rc = main(["list"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "no recipes" in captured.out.lower() or "0 recipe" in captured.out.lower()

    def test_save_then_list(self, tmp_recipes_dir, capsys):
        from vllm._genesis.compat.recipes import save, main
        save("alpha", _VALID_RECIPE)
        save("beta", _VALID_RECIPE)
        main(["list"])
        captured = capsys.readouterr()
        assert "alpha" in captured.out
        assert "beta" in captured.out

    def test_show_subcommand(self, tmp_recipes_dir, capsys):
        from vllm._genesis.compat.recipes import save, main
        save("inspect-me", _VALID_RECIPE)
        rc = main(["show", "inspect-me"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "inspect-me" in captured.out
        assert "qwen3_6_27b_int4_autoround" in captured.out

    def test_show_unknown_returns_nonzero(self, tmp_recipes_dir, capsys):
        from vllm._genesis.compat.recipes import main
        rc = main(["show", "missing"])
        assert rc != 0

    def test_delete_subcommand(self, tmp_recipes_dir, capsys):
        from vllm._genesis.compat.recipes import save, list_names, main
        save("doomed", _VALID_RECIPE)
        rc = main(["delete", "doomed"])
        assert rc == 0
        assert "doomed" not in list_names()

    def test_load_writes_launch_script(self, tmp_path, tmp_recipes_dir):
        from vllm._genesis.compat.recipes import save, main
        save("with-launch", _VALID_RECIPE)
        out_script = tmp_path / "launch.sh"
        rc = main(["load", "with-launch", "--out", str(out_script)])
        assert rc == 0
        assert out_script.is_file()
        content = out_script.read_text()
        assert "docker run" in content
        assert "GENESIS_ENABLE_PN14_TQ_DECODE_OOB_CLAMP=1" in content


# ─── Diff (community A/B compare) ───────────────────────────────────────


class TestDiffRecipes:
    """`diff_recipes(a, b)` returns a structured comparison so an
    operator can A/B two configs and see exactly what differs."""

    def test_diff_identical_recipes_is_empty(self):
        from vllm._genesis.compat.recipes import diff_recipes
        d = diff_recipes(_VALID_RECIPE, dict(_VALID_RECIPE))
        assert d["added"] == {}
        assert d["removed"] == {}
        assert d["changed"] == {}

    def test_diff_added_keys(self):
        from vllm._genesis.compat.recipes import diff_recipes
        a = {"envs": {"X": "1"}}
        b = {"envs": {"X": "1", "Y": "2"}}
        d = diff_recipes(a, b)
        assert d["added"] == {"envs.Y": "2"}
        assert d["removed"] == {}
        assert d["changed"] == {}

    def test_diff_removed_keys(self):
        from vllm._genesis.compat.recipes import diff_recipes
        a = {"envs": {"X": "1", "Y": "2"}}
        b = {"envs": {"X": "1"}}
        d = diff_recipes(a, b)
        assert d["removed"] == {"envs.Y": "2"}
        assert d["added"] == {}

    def test_diff_changed_value(self):
        from vllm._genesis.compat.recipes import diff_recipes
        a = {"envs": {"X": "1"}}
        b = {"envs": {"X": "2"}}
        d = diff_recipes(a, b)
        assert d["changed"] == {"envs.X": ("1", "2")}
        assert d["added"] == {}
        assert d["removed"] == {}

    def test_diff_nested_dict(self):
        from vllm._genesis.compat.recipes import diff_recipes
        a = {"vllm_serve": {"max_model_len": 280000, "tensor_parallel_size": 2}}
        b = {"vllm_serve": {"max_model_len": 320000, "tensor_parallel_size": 2}}
        d = diff_recipes(a, b)
        assert d["changed"] == {"vllm_serve.max_model_len": (280000, 320000)}

    def test_diff_ignores_provenance_fields(self):
        """`created_at`, `created_by`, `_adopted_from`, `_adopted_at`
        are operator-irrelevant — recipes saved at different times by
        different people but otherwise identical should diff clean."""
        from vllm._genesis.compat.recipes import diff_recipes
        a = dict(_VALID_RECIPE)
        a["created_at"] = "2026-01-01T00:00:00Z"
        a["created_by"] = "alice"
        a["_adopted_from"] = "https://gist.github.com/alice/1.json"
        b = dict(_VALID_RECIPE)
        b["created_at"] = "2026-04-30T00:00:00Z"
        b["created_by"] = "bob"
        b["_adopted_from"] = "https://gist.github.com/bob/2.json"
        d = diff_recipes(a, b)
        assert d["changed"] == {}, (
            f"provenance-only delta should diff clean; got {d}"
        )

    def test_diff_lists_are_compared_as_values(self):
        """Lists (e.g. ports) compare as opaque values — order matters."""
        from vllm._genesis.compat.recipes import diff_recipes
        a = {"container": {"ports": [8000]}}
        b = {"container": {"ports": [8000, 8001]}}
        d = diff_recipes(a, b)
        assert "container.ports" in d["changed"]


class TestDiffCLI:
    def test_diff_subcommand_routes(self, tmp_recipes_dir, capsys):
        from vllm._genesis.compat.recipes import save, main
        save("alpha", _VALID_RECIPE)
        modified = dict(_VALID_RECIPE)
        modified["vllm_serve"] = dict(modified["vllm_serve"])
        modified["vllm_serve"]["max_model_len"] = 320000
        save("beta", modified)
        rc = main(["diff", "alpha", "beta"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "max_model_len" in captured.out
        assert "280000" in captured.out
        assert "320000" in captured.out

    def test_diff_unknown_recipe_returns_nonzero(
        self, tmp_recipes_dir, capsys
    ):
        from vllm._genesis.compat.recipes import save, main
        save("real", _VALID_RECIPE)
        rc = main(["diff", "real", "missing"])
        assert rc != 0

    def test_diff_identical_recipes_says_so(
        self, tmp_recipes_dir, capsys
    ):
        from vllm._genesis.compat.recipes import save, main
        save("twin-a", _VALID_RECIPE)
        save("twin-b", dict(_VALID_RECIPE))
        rc = main(["diff", "twin-a", "twin-b"])
        assert rc == 0
        captured = capsys.readouterr()
        assert "identical" in captured.out.lower() or "no differences" in captured.out.lower()

    def test_diff_json_output(
        self, tmp_recipes_dir, capsys
    ):
        from vllm._genesis.compat.recipes import save, main
        save("x", _VALID_RECIPE)
        modified = dict(_VALID_RECIPE)
        modified["envs"] = {**modified["envs"], "GENESIS_NEW": "1"}
        save("y", modified)
        rc = main(["diff", "x", "y", "--json"])
        assert rc == 0
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert "added" in parsed
        assert "removed" in parsed
        assert "changed" in parsed
