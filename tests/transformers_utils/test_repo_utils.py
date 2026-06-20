# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import tempfile
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
import requests
from huggingface_hub.utils import RepositoryNotFoundError

from vllm.transformers_utils.repo_utils import (
    any_pattern_in_repo_files,
    is_mistral_model_repo,
    is_transient_hf_error,
    list_filtered_repo_files,
    maybe_resolve_latest_hf_revision,
    retry_with_kwargs,
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


def test_retry_with_kwargs_retries_with_kwargs():
    calls: list[dict[str, object]] = []

    def flaky_call(**kwargs):
        calls.append(kwargs.copy())
        if len(calls) == 1:
            raise RuntimeError("transient failure")
        return kwargs["local_files_only"]

    call_with_retry = retry_with_kwargs(flaky_call, local_files_only=True)
    assert call_with_retry(model="cached-model") is True

    assert calls == [
        {"model": "cached-model"},
        {"model": "cached-model", "local_files_only": True},
    ]


def test_retry_with_kwargs_does_not_retry_when_predicate_rejects():
    calls = 0

    def failing_call(**kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("failure")

    call_with_retry = retry_with_kwargs(
        failing_call,
        retry_on_exception=lambda e: False,
        local_files_only=True,
    )
    with pytest.raises(RuntimeError, match="failure"):
        call_with_retry()

    assert calls == 1


def test_retry_with_kwargs_retries_with_missing_none_kwarg():
    calls: list[dict[str, object | None]] = []

    def flaky_call(**kwargs):
        calls.append(kwargs.copy())
        if len(calls) == 1:
            raise RuntimeError("transient failure")
        return kwargs["revision"]

    call_with_retry = retry_with_kwargs(flaky_call, revision=None)
    assert call_with_retry(model="cached-model") is None

    assert calls == [
        {"model": "cached-model"},
        {"model": "cached-model", "revision": None},
    ]


def test_is_transient_hf_error_rejects_wrapped_hub_access_errors():
    response = requests.Response()
    response.status_code = 403

    try:
        raise requests.HTTPError(response=response)
    except requests.HTTPError as e:
        try:
            raise RepositoryNotFoundError("private repo", response=response) from e
        except RepositoryNotFoundError as exc:
            assert not is_transient_hf_error(exc)


def test_is_transient_hf_error_accepts_server_errors():
    response = requests.Response()
    response.status_code = 503

    assert is_transient_hf_error(requests.HTTPError(response=response))


def test_maybe_resolve_latest_hf_revision_preserves_explicit_revision(
    monkeypatch: pytest.MonkeyPatch,
):
    hf_api_mock = MagicMock()
    monkeypatch.setattr(
        "vllm.transformers_utils.repo_utils.hf_api",
        hf_api_mock,
    )

    assert (
        maybe_resolve_latest_hf_revision("org/model", "pinned-revision")
        == "pinned-revision"
    )
    hf_api_mock.assert_not_called()


def test_maybe_resolve_latest_hf_revision_can_force_dataset_lookup(
    monkeypatch: pytest.MonkeyPatch,
):
    class Info:
        sha = "latest-dataset-sha"

    class FakeHfApi:
        def dataset_info(self, repo_id, token=None):
            assert repo_id == "org/dataset"
            assert token is None
            return Info()

    monkeypatch.setattr(
        "vllm.transformers_utils.repo_utils.envs.VLLM_CI_ENSURE_LATEST_HF_REVISION",
        False,
        raising=False,
    )
    monkeypatch.setattr(
        "vllm.transformers_utils.repo_utils.envs.VLLM_USE_MODELSCOPE",
        False,
        raising=False,
    )
    monkeypatch.setattr(
        "vllm.transformers_utils.repo_utils.huggingface_hub.constants.HF_HUB_OFFLINE",
        False,
    )
    monkeypatch.setattr(
        "vllm.transformers_utils.repo_utils.hf_api",
        lambda: FakeHfApi(),
    )

    assert (
        maybe_resolve_latest_hf_revision(
            "org/dataset",
            None,
            repo_type="dataset",
            ensure_latest=True,
        )
        == "latest-dataset-sha"
    )
