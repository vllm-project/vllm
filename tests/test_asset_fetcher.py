# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from tests import utils as test_utils

ASSET = b"complete"
ASSET_SHA256 = hashlib.sha256(ASSET).hexdigest()


def test_fetch_repairs_and_reuses_asset(monkeypatch, tmp_path):
    path = tmp_path / "asset.bin"
    path.write_bytes(b"stale")
    calls = 0

    def download(url, destination, timeout):
        nonlocal calls
        calls += 1
        assert timeout == 300
        destination.write_bytes(ASSET)

    monkeypatch.setattr(test_utils.global_http_connection, "download_file", download)
    assets = test_utils.TestAssetFetcher(tmp_path)

    assert assets.fetch("https://example.com/a", path.name, ASSET_SHA256) == path
    assert path.read_bytes() == ASSET
    assert path.stat().st_mode & 0o777 == 0o644
    assert assets.fetch("https://example.com/a", path.name, ASSET_SHA256) == path
    assert calls == 1


def test_fetch_honors_hf_offline_mode(monkeypatch, tmp_path):
    path = tmp_path / "asset.bin"
    corrupt_path = tmp_path / "corrupt.bin"
    path.write_bytes(ASSET)
    corrupt_path.write_bytes(b"stale")
    monkeypatch.setattr(test_utils.hf_constants, "HF_HUB_OFFLINE", True)
    monkeypatch.setattr(
        test_utils.global_http_connection,
        "download_file",
        lambda *args, **kwargs: pytest.fail("a valid asset must not download"),
    )
    assets = test_utils.TestAssetFetcher(tmp_path)

    assert assets.fetch("https://example.com/a", path.name, ASSET_SHA256) == path
    with pytest.raises(FileNotFoundError, match="HF_HUB_OFFLINE"):
        assets.fetch("https://example.com/b", corrupt_path.name, ASSET_SHA256)
    assert corrupt_path.read_bytes() == b"stale"


def test_fetch_rejects_bad_download_without_replacing_target(monkeypatch, tmp_path):
    path = tmp_path / "asset.bin"
    path.write_bytes(b"stale")

    def download(url, destination, timeout):
        destination.write_bytes(b"unexpected")

    monkeypatch.setattr(test_utils.global_http_connection, "download_file", download)

    with pytest.raises(ValueError, match="failed validation"):
        test_utils.TestAssetFetcher(tmp_path).fetch(
            "https://example.com/a", path.name, ASSET_SHA256
        )

    assert path.read_bytes() == b"stale"
    assert list(tmp_path.iterdir()) == [path]


def test_fetch_cleans_up_interrupted_download(monkeypatch, tmp_path):
    def download(url, destination, timeout):
        destination.write_bytes(b"partial")
        raise OSError("download interrupted")

    monkeypatch.setattr(test_utils.global_http_connection, "download_file", download)

    with pytest.raises(OSError, match="download interrupted"):
        test_utils.TestAssetFetcher(tmp_path).fetch(
            "https://example.com/a", "asset.bin", ASSET_SHA256
        )

    assert list(tmp_path.iterdir()) == []


def test_fetch_handles_concurrent_writers(monkeypatch, tmp_path):
    workers = 8
    barrier = threading.Barrier(workers)

    def download(url, destination, timeout):
        barrier.wait(timeout=5)
        destination.write_bytes(ASSET)

    monkeypatch.setattr(test_utils.global_http_connection, "download_file", download)
    assets = test_utils.TestAssetFetcher(tmp_path)
    path = tmp_path / "asset.bin"

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(
            executor.map(
                lambda _: assets.fetch(
                    "https://example.com/a", path.name, ASSET_SHA256
                ),
                range(workers),
            )
        )

    assert results == [path] * workers
    assert path.read_bytes() == ASSET
    assert sorted(tmp_path.iterdir()) == [path]


def test_for_suite_uses_hf_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HOME", str(tmp_path))

    assert test_utils.TestAssetFetcher.for_suite("eval/data").directory == (
        tmp_path / "vllm-test-assets" / "eval/data"
    )


def test_for_suite_uses_default_hf_home(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setattr(test_utils.hf_constants, "HF_HOME", str(tmp_path))

    assert test_utils.TestAssetFetcher.for_suite("eval/data").directory == (
        tmp_path / "vllm-test-assets" / "eval/data"
    )


@pytest.mark.parametrize("name", ["", "../escape", "/tmp/escape"])
def test_for_suite_rejects_unsafe_paths(name):
    with pytest.raises(ValueError, match="Unsafe test asset path"):
        test_utils.TestAssetFetcher.for_suite(name)


@pytest.mark.parametrize("filename", ["../asset.bin", "nested/asset.bin"])
def test_fetch_rejects_unsafe_filenames(filename, tmp_path):
    with pytest.raises(ValueError, match="Unsafe test asset path"):
        test_utils.TestAssetFetcher(tmp_path).fetch(
            "https://example.com/a", filename, ASSET_SHA256
        )


def test_gsm8k_eval_supports_direct_script_invocation(tmp_path):
    script = (Path(__file__).parent / "evals" / "gsm8k" / "gsm8k_eval.py").resolve()
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [sys.executable, "-I", str(script), "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "GSM8K evaluation for vLLM serve" in result.stdout


def test_dummy_model_path_filters_weights_from_shared_snapshot(monkeypatch, tmp_path):
    from tests import conftest as test_conftest

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text(
        json.dumps({"architectures": ["Original"]}), encoding="utf-8"
    )
    (snapshot / "model.safetensors").write_bytes(b"large weight placeholder")

    class FakeHfApi:
        def snapshot_download(self, **kwargs):
            assert kwargs == {
                "repo_id": "org/model",
                "ignore_patterns": ["*.safetensors"],
            }
            return str(snapshot)

    monkeypatch.setattr(test_conftest, "hf_api", lambda: FakeHfApi())

    model_path = Path(
        test_conftest._create_dummy_model_path(
            tmp_path / "models",
            "dummy",
            "org/model",
            "DummyArchitecture",
            ["*.safetensors"],
        )
    )

    assert json.loads((model_path / "config.json").read_text(encoding="utf-8"))[
        "architectures"
    ] == ["DummyArchitecture"]
    assert not (model_path / "model.safetensors").exists()
