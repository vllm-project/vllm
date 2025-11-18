# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import tempfile
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from vllm.transformers_utils.utils import (
    _list_local_hf_repo_files,
    is_cloud_storage,
    is_gcs,
    is_s3,
    list_files_from_hf_or_path,
)


def test_is_gcs():
    assert is_gcs("gs://model-path")
    assert not is_gcs("s3://model-path/path-to-model")
    assert not is_gcs("/unix/local/path")
    assert not is_gcs("nfs://nfs-fqdn.local")


def test_is_s3():
    assert is_s3("s3://model-path/path-to-model")
    assert not is_s3("gs://model-path")
    assert not is_s3("/unix/local/path")
    assert not is_s3("nfs://nfs-fqdn.local")


def test_is_cloud_storage():
    assert is_cloud_storage("gs://model-path")
    assert is_cloud_storage("s3://model-path/path-to-model")
    assert not is_cloud_storage("/unix/local/path")
    assert not is_cloud_storage("nfs://nfs-fqdn.local")


@pytest.mark.parametrize("revision", ["test", "empty", None])
def test_list_local_hf_repo_files(revision: str | None):
    repo_id = "test-org/test-local-hf"
    # tmp fake hf hub cache
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Prep hf constants overrides
        hf_hub_cache = tmp_dir
        repo_id_separator = "-"
        default_revision = "main"

        path_tmp_dir = Path(hf_hub_cache)
        repo_path = path_tmp_dir / repo_id_separator.join(
            ["models", *repo_id.split("/")]
        )
        repo_path.mkdir()

        main_revision_file = repo_path / "refs" / default_revision
        main_revision_file.parent.mkdir(parents=True)
        main_revision_file.open(mode="w", encoding="utf-8").write("main")

        main_revision_dir = repo_path / "snapshots" / "main"
        main_revision_dir.mkdir(parents=True)

        for file in ["main-file-1.txt", "main-file-2.json"]:
            (main_revision_dir / file).touch()

        for dir in ["main-dir-1", "main-dir-2.json"]:
            (main_revision_dir / dir).mkdir()

        if revision == "test":
            revision_dir = repo_path / "snapshots" / revision
            revision_dir.mkdir()
            for file in ["revision-file-1.txt", "revision-file-2.json"]:
                (revision_dir / file).touch()

            for dir in ["revision-dir-1", "revision-dir-2.json"]:
                (revision_dir / dir).mkdir()

        with (
            patch("huggingface_hub.constants.HF_HUB_CACHE", hf_hub_cache),
            patch("huggingface_hub.constants.DEFAULT_REVISION", default_revision),
            patch("huggingface_hub.constants.REPO_ID_SEPARATOR", repo_id_separator),
        ):
            out_files = sorted(_list_local_hf_repo_files(repo_id, revision))

    if revision is None:
        expected_out_files = sorted(
            str(main_revision_dir / filename)
            for filename in ["main-file-1.txt", "main-file-2.json"]
        )
    elif revision == "test":
        expected_out_files = sorted(
            str(revision_dir / filename)
            for filename in ["revision-file-1.txt", "revision-file-2.json"]
        )
    else:
        expected_out_files = []

    assert out_files == expected_out_files


@pytest.mark.parametrize(
    "allow_patterns,expected_relative_files",
    [
        (
            ["*.json", "correct*.txt"],
            ["json_file.json", "subfolder/correct.txt", "correct_2.txt"],
        ),
    ],
)
def test_list_files_from_hf_or_path(
    allow_patterns: list[str], expected_relative_files: list[str]
):
    # Local dir
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

        expected_out_files = sorted(
            str(path_tmp_dir / file) for file in expected_relative_files
        )

        def _glob_path() -> list[str]:
            return [str(file) for file in path_tmp_dir.glob("**/*") if file.is_file()]

        # test directory
        out_files = sorted(list_files_from_hf_or_path(tmp_dir, allow_patterns, None))
        assert out_files == expected_out_files

        # test local hf repo
        # mock _list_local_hf_repo_files tested above
        with (
            patch("huggingface_hub.constants.HF_HUB_OFFLINE", 1),
            patch(
                "vllm.transformers_utils.utils._list_local_hf_repo_files",
                MagicMock(return_value=_glob_path()),
            ) as mock_offline,
        ):
            out_files = sorted(
                list_files_from_hf_or_path(
                    "dummy-org/dummy-repo", allow_patterns, "offline"
                )
            )
        assert out_files == expected_out_files
        assert mock_offline.call_count == 1
        assert mock_offline.call_args_list[0] == call(
            repo_id="dummy-org/dummy-repo", revision="offline"
        )

        # test remote hf repo
        # mock HfAPI.list_repo_files
        with patch(
            "huggingface_hub.HfApi.list_repo_files",
            MagicMock(return_value=_glob_path()),
        ) as mock_online:
            out_files = sorted(
                list_files_from_hf_or_path(
                    "dummy-org/dummy-repo", allow_patterns, "online"
                )
            )
        assert out_files == expected_out_files
        assert mock_online.call_count == 1
        assert mock_online.call_args_list[0] == call(
            repo_id="dummy-org/dummy-repo", revision="online"
        )
