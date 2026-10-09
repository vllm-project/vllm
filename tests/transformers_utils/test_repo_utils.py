# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import tempfile
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
from huggingface_hub import _CACHED_NO_EXIST

from vllm.transformers_utils.repo_utils import (
    any_pattern_in_repo_files,
    get_hf_file_to_dict,
    get_non_weight_snapshot_path,
    is_mistral_model_repo,
    list_filtered_repo_files,
    with_retry,
)


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


def test_with_retry_does_not_retry_fatal_errors():
    """A definitive answer must not be retried, which costs a delay per attempt."""
    func = MagicMock(side_effect=FileNotFoundError("no such file"))

    with pytest.raises(FileNotFoundError):
        with_retry(func, "Error", fatal_errors=(FileNotFoundError,))

    func.assert_called_once()


@pytest.mark.parametrize(
    ("side_effect", "expected"),
    [("/snapshots/abc", "/snapshots/abc"), (OSError("offline"), "some/repo")],
)
def test_get_non_weight_snapshot_path(side_effect: object, expected: str):
    """Hub repos resolve to a local snapshot, falling back to the repo ID."""
    get_non_weight_snapshot_path.cache_clear()
    mock_snapshot = MagicMock(side_effect=[side_effect])
    with patch("vllm.transformers_utils.repo_utils.hf_api") as mock_api:
        mock_api.return_value.snapshot_download = mock_snapshot
        assert get_non_weight_snapshot_path("some/repo") == expected
    assert "*.safetensors" in mock_snapshot.call_args.kwargs["ignore_patterns"]


def test_get_non_weight_snapshot_path_local(tmp_path: Path):
    get_non_weight_snapshot_path.cache_clear()
    with patch("vllm.transformers_utils.repo_utils.hf_api") as mock_api:
        assert get_non_weight_snapshot_path(str(tmp_path)) == str(tmp_path)
    mock_api.assert_not_called()
