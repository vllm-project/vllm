# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import io
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from tests import cache_utils


def test_download_url_to_file_repairs_empty_file_and_reuses_hit(monkeypatch, tmp_path):
    path = tmp_path / "asset.bin"
    path.touch()
    calls = 0

    def urlopen(url, timeout):
        nonlocal calls
        calls += 1
        return io.BytesIO(b"complete")

    monkeypatch.setattr(cache_utils.urllib.request, "urlopen", urlopen)

    assert cache_utils.download_url_to_file("https://example.com/a", path) == path
    assert path.read_bytes() == b"complete"
    assert cache_utils.download_url_to_file("https://example.com/a", path) == path
    assert calls == 1


def test_download_url_to_file_repairs_hash_mismatch(monkeypatch, tmp_path):
    path = tmp_path / "asset.bin"
    path.write_bytes(b"stale")
    expected_sha256 = "3a6eb0790f39ac87c94f3856b2dd2c5d110e6811602261a9a923d3bb23adc8b7"

    monkeypatch.setattr(
        cache_utils.urllib.request,
        "urlopen",
        lambda url, timeout: io.BytesIO(b"data"),
    )

    cache_utils.download_url_to_file(
        "https://example.com/a", path, expected_sha256=expected_sha256
    )

    assert path.read_bytes() == b"data"
    assert path.stat().st_mode & 0o777 == 0o644


def test_download_url_to_file_reuses_valid_hash(monkeypatch, tmp_path):
    path = tmp_path / "asset.bin"
    path.write_bytes(b"data")
    expected_sha256 = "3A6EB0790F39AC87C94F3856B2DD2C5D110E6811602261A9A923D3BB23ADC8B7"

    monkeypatch.setattr(
        cache_utils.urllib.request,
        "urlopen",
        lambda url, timeout: pytest.fail("a valid cache hit must not download"),
    )

    assert (
        cache_utils.download_url_to_file(
            "https://example.com/a", path, expected_sha256=expected_sha256
        )
        == path
    )


def test_download_url_to_file_rejects_hash_mismatch_without_replacing_target(
    monkeypatch, tmp_path
):
    path = tmp_path / "asset.bin"
    path.write_bytes(b"stale")
    monkeypatch.setattr(
        cache_utils.urllib.request,
        "urlopen",
        lambda url, timeout: io.BytesIO(b"unexpected"),
    )

    with pytest.raises(ValueError, match="failed cache validation"):
        cache_utils.download_url_to_file(
            "https://example.com/a", path, expected_sha256="0" * 64
        )

    assert path.read_bytes() == b"stale"


@pytest.mark.parametrize("expected_sha256", ["short", "g" * 64])
def test_download_url_to_file_rejects_invalid_hash(expected_sha256, tmp_path):
    with pytest.raises(ValueError, match="exactly 64 hexadecimal digits"):
        cache_utils.download_url_to_file(
            "https://example.com/a",
            tmp_path / "asset.bin",
            expected_sha256=expected_sha256,
        )


def test_download_url_to_file_honors_cache_only_overrides(monkeypatch, tmp_path):
    path = tmp_path / "asset.bin"
    monkeypatch.setenv(cache_utils.TEST_CACHE_ONLY_ENV, "yes")

    with pytest.raises(FileNotFoundError, match=cache_utils.TEST_CACHE_ONLY_ENV):
        cache_utils.download_url_to_file("https://example.com/a", path)

    monkeypatch.setattr(
        cache_utils.urllib.request,
        "urlopen",
        lambda url, timeout: io.BytesIO(b"complete"),
    )
    cache_utils.download_url_to_file("https://example.com/a", path, local_only=False)

    with pytest.raises(FileNotFoundError):
        cache_utils.download_url_to_file(
            "https://example.com/b", tmp_path / "missing.bin", local_only=True
        )


def test_download_url_to_file_does_not_publish_partial_download(monkeypatch, tmp_path):
    path = tmp_path / "asset.bin"

    class FailingResponse(io.BytesIO):
        def read(self, size=-1):
            data = super().read(size)
            if not data:
                raise OSError("download interrupted")
            return data

    monkeypatch.setattr(
        cache_utils.urllib.request,
        "urlopen",
        lambda url, timeout: FailingResponse(b"partial"),
    )

    with pytest.raises(OSError, match="download interrupted"):
        cache_utils.download_url_to_file("https://example.com/a", path)

    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


def test_download_url_to_file_handles_concurrent_writers(monkeypatch, tmp_path):
    path = tmp_path / "asset.bin"
    workers = 8
    barrier = threading.Barrier(workers)

    def urlopen(url, timeout):
        barrier.wait(timeout=5)
        return io.BytesIO(b"complete")

    monkeypatch.setattr(cache_utils.urllib.request, "urlopen", urlopen)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(
            executor.map(
                lambda _: cache_utils.download_url_to_file(
                    "https://example.com/a", path
                ),
                range(workers),
            )
        )

    assert results == [path] * workers
    assert path.read_bytes() == b"complete"
    assert sorted(tmp_path.iterdir()) == [path]


def test_download_to_vllm_test_cache_uses_distinct_url_keys(monkeypatch, tmp_path):
    monkeypatch.setenv(cache_utils.TEST_CACHE_ENV, str(tmp_path))
    monkeypatch.setattr(
        cache_utils.urllib.request,
        "urlopen",
        lambda url, timeout: io.BytesIO(url.encode()),
    )

    first = cache_utils.download_to_vllm_test_cache(
        "https://one.example/data.bin?revision=1", "suite"
    )
    second = cache_utils.download_to_vllm_test_cache(
        "https://two.example/data.bin?revision=1", "suite"
    )

    assert first.parent == second.parent == tmp_path / "suite"
    assert first.name != second.name
    assert first.name.endswith("-data.bin")
    assert second.name.endswith("-data.bin")


@pytest.mark.parametrize(
    ("namespace", "filename"),
    [
        ("../outside", None),
        ("/outside", None),
        ("suite", "../outside"),
        ("suite", "/outside"),
        ("suite", "nested/file"),
    ],
)
def test_download_to_vllm_test_cache_rejects_escaping_paths(
    monkeypatch, tmp_path, namespace, filename
):
    monkeypatch.setenv(cache_utils.TEST_CACHE_ENV, str(tmp_path))

    with pytest.raises(ValueError, match="safe relative cache path"):
        cache_utils.download_to_vllm_test_cache(
            "https://example.com/a", namespace, filename=filename
        )


def test_download_url_to_file_rejects_directories_and_replaces_symlinks(
    monkeypatch, tmp_path
):
    directory = tmp_path / "directory"
    directory.mkdir()
    with pytest.raises(IsADirectoryError):
        cache_utils.download_url_to_file("https://example.com/a", directory)

    source = tmp_path / "source.bin"
    source.write_bytes(b"untrusted")
    path = tmp_path / "asset.bin"
    path.symlink_to(source)
    with pytest.raises(FileNotFoundError):
        cache_utils.download_url_to_file("https://example.com/a", path, local_only=True)

    monkeypatch.setattr(
        cache_utils.urllib.request,
        "urlopen",
        lambda url, timeout: io.BytesIO(b"complete"),
    )
    cache_utils.download_url_to_file("https://example.com/a", path)

    assert not path.is_symlink()
    assert path.read_bytes() == b"complete"
    assert source.read_bytes() == b"untrusted"


def test_gsm8k_eval_supports_direct_script_invocation(tmp_path):
    script = Path(__file__).resolve().parent / "evals" / "gsm8k" / "gsm8k_eval.py"
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
