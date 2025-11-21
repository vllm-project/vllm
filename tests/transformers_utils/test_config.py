# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import tempfile
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from vllm.transformers_utils.config import list_filtered_repo_files


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
        (path_tmp_dir / "uncorrect.txt").touch()
        (path_tmp_dir / "uncorrect.jpeg").touch()
        (subfolder / "correct.txt").touch()
        (subfolder / "uncorrect_sub.txt").touch()

        def _glob_path() -> list[str]:
            return [
                str(file.relative_to(path_tmp_dir))
                for file in path_tmp_dir.glob("**/*")
                if file.is_file()
            ]

        # Patch list_repo_files called by fn
        with patch(
            "vllm.transformers_utils.config.list_repo_files",
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
