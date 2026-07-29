# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import gc
import tempfile
import weakref
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import httpx
import huggingface_hub
import pytest
from huggingface_hub import _CACHED_NO_EXIST

from vllm.transformers_utils.repo_utils import (
    any_pattern_in_repo_files,
    get_hf_file_to_dict,
    is_mistral_model_repo,
    list_filtered_repo_files,
)


def test_hf_hub_cached_fallback_does_not_retain_caller(monkeypatch, tmp_path):
    class Owner:
        pass

    repo_id = "test-org/test-model"
    filename = "config.json"
    commit_hash = "0" * 40
    storage = tmp_path / "models--test-org--test-model"
    pointer = storage / "snapshots" / commit_hash / filename
    pointer.parent.mkdir(parents=True)
    pointer.write_text("{}")
    refs = storage / "refs"
    refs.mkdir()
    (refs / "main").write_text(commit_hash)

    def fail_metadata_request(*args, **kwargs):
        try:
            raise OSError("simulated connection reset")
        except OSError as error:
            raise httpx.ConnectError("simulated transient HEAD failure") from error

    monkeypatch.setattr(
        huggingface_hub.file_download,
        "get_hf_file_metadata",
        fail_metadata_request,
    )

    def download_from_caller(owner):
        return huggingface_hub.hf_hub_download(
            repo_id,
            filename,
            cache_dir=tmp_path,
        )

    gc_was_enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        owner = Owner()
        owner_ref = weakref.ref(owner)
        assert Path(download_from_caller(owner)) == pointer
        del owner
        assert owner_ref() is None
    finally:
        if gc_was_enabled:
            gc.enable()
        gc.collect()


def test_hf_hub_uncached_metadata_failure_still_raises(monkeypatch, tmp_path):
    def fail_metadata_request(*args, **kwargs):
        raise httpx.ConnectError("simulated transient HEAD failure")

    monkeypatch.setattr(
        huggingface_hub.file_download,
        "get_hf_file_metadata",
        fail_metadata_request,
    )

    with pytest.raises(huggingface_hub.errors.LocalEntryNotFoundError) as exc_info:
        huggingface_hub.hf_hub_download(
            "test-org/uncached-model",
            "config.json",
            cache_dir=tmp_path,
        )
    assert isinstance(exc_info.value.__cause__, httpx.ConnectError)


@pytest.mark.parametrize(
    "allow_patterns,expected_relative_files",
    [
        (
            ["*.json", "correct*.txt"],
            ["json_file.json", "subfolder/correct.txt", "correct_2.txt"],
        ),
    ],
)
def test_list_filtered_repo_files(
    allow_patterns: list[str], expected_relative_files: list[str]
):
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Prep folder and files
        path_tmp_dir = Path(tmp_dir)
        subfolder = path_tmp_dir / "subfolder"
        subfolder.mkdir()
        (path_tmp_dir / "json_file.json").touch()
        (path_tmp_dir / "correct_2.txt").touch()
        (path_tmp_dir / "incorrect.txt").touch()
        (path_tmp_dir / "incorrect.jpeg").touch()
        (subfolder / "correct.txt").touch()
        (subfolder / "incorrect_sub.txt").touch()

        def _glob_path() -> list[str]:
            return [
                str(file.relative_to(path_tmp_dir))
                for file in path_tmp_dir.glob("**/*")
                if file.is_file()
            ]

        # Patch list_repo_files called by fn
        with patch(
            "vllm.transformers_utils.repo_utils.list_repo_files",
            MagicMock(return_value=_glob_path()),
        ) as mock_list_repo_files:
            out_files = sorted(
                list_filtered_repo_files(
                    tmp_dir, allow_patterns, "revision", "model", "token"
                )
            )
        assert out_files == sorted(expected_relative_files)
        assert mock_list_repo_files.call_count == 1
        assert mock_list_repo_files.call_args_list[0] == call(
            repo_id=tmp_dir,
            revision="revision",
            repo_type="model",
            token="token",
        )


@pytest.mark.parametrize(
    ("allow_patterns", "expected_bool"),
    [
        (["*.json", "correct*.txt"], True),
        (
            ["*.jpeg"],
            True,
        ),
        (
            ["not_found.jpeg"],
            False,
        ),
    ],
)
def test_one_filtered_repo_files(allow_patterns: list[str], expected_bool: bool):
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Prep folder and files
        path_tmp_dir = Path(tmp_dir)
        subfolder = path_tmp_dir / "subfolder"
        subfolder.mkdir()
        (path_tmp_dir / "incorrect.jpeg").touch()
        (subfolder / "correct.txt").touch()

        def _glob_path() -> list[str]:
            return [
                str(file.relative_to(path_tmp_dir))
                for file in path_tmp_dir.glob("**/*")
                if file.is_file()
            ]

        # Patch list_repo_files called by fn
        with patch(
            "vllm.transformers_utils.repo_utils.list_repo_files",
            MagicMock(return_value=_glob_path()),
        ) as mock_list_repo_files:
            assert (
                any_pattern_in_repo_files(
                    tmp_dir, allow_patterns, "revision", "model", "token"
                )
            ) is expected_bool
        assert mock_list_repo_files.call_count == 1
        assert mock_list_repo_files.call_args_list[0] == call(
            repo_id=tmp_dir,
            revision="revision",
            repo_type="model",
            token="token",
        )


@pytest.mark.parametrize(
    ("cache_result", "should_download"),
    [
        # HF Hub recorded a prior 404: don't re-probe the Hub.
        (_CACHED_NO_EXIST, False),
        # File not in cache and existence unknown: preserve download behavior.
        (None, True),
    ],
)
def test_get_hf_file_to_dict_honors_no_exist_marker(
    cache_result: object, should_download: bool
):
    with (
        patch(
            "vllm.transformers_utils.repo_utils.try_to_load_from_cache",
            MagicMock(return_value=cache_result),
        ),
        patch(
            "vllm.transformers_utils.repo_utils._try_download_from_hf_hub",
            MagicMock(return_value=None),
        ) as mock_download,
    ):
        result = get_hf_file_to_dict("processor_config.json", "some/repo")
    assert result is None
    assert mock_download.call_count == int(should_download)


@pytest.mark.parametrize(
    ("files", "expected_bool"),
    [
        (["consolidated.safetensors", "incorrect.txt"], True),
        (["consolidated-1.safetensors", "incorrect.txt"], True),
        (
            ["consolidated-1.json"],
            False,
        ),
    ],
)
def test_is_mistral_model_repo(files: list[str], expected_bool: bool):
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Prep folder and files
        path_tmp_dir = Path(tmp_dir)
        for file in files:
            (path_tmp_dir / file).touch()

        def _glob_path() -> list[str]:
            return [
                str(file.relative_to(path_tmp_dir))
                for file in path_tmp_dir.glob("**/*")
                if file.is_file()
            ]

        # Patch list_repo_files called by fn
        with patch(
            "vllm.transformers_utils.repo_utils.list_repo_files",
            MagicMock(return_value=_glob_path()),
        ) as mock_list_repo_files:
            assert (
                is_mistral_model_repo(tmp_dir, "revision", "model", "token")
                is expected_bool
            )
        assert mock_list_repo_files.call_count == 1
        assert mock_list_repo_files.call_args_list[0] == call(
            repo_id=tmp_dir,
            revision="revision",
            repo_type="model",
            token="token",
        )
