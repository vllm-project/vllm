# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import pytest

from vllm.model_executor.model_loader import weight_utils
from vllm.model_executor.model_loader.weight_utils import (
    record_checkpoint_files,
    release_checkpoint_page_cache,
)


@pytest.fixture
def released(monkeypatch):
    """Inodes passed to posix_fadvise(DONTNEED), in call order."""
    inodes: list[int] = []

    def fadvise(fd, offset, length, advice):
        assert (offset, length, advice) == (0, 0, 4)
        inodes.append(os.fstat(fd).st_ino)

    monkeypatch.setattr(weight_utils, "_checkpoint_files", {})
    monkeypatch.setattr(os, "posix_fadvise", fadvise, raising=False)
    monkeypatch.setattr(os, "POSIX_FADV_DONTNEED", 4, raising=False)
    return inodes


def test_release_once_after_target_and_drafter(tmp_path, released):
    files = []
    for i in range(2):
        path = tmp_path / f"model-{i}.safetensors"
        path.write_bytes(b"weights")
        files.append(str(path))

    # The target model and an MTP drafter read the same checkpoint files.
    record_checkpoint_files(files)
    record_checkpoint_files(files)
    assert released == []

    release_checkpoint_page_cache()
    assert released == [os.stat(f).st_ino for f in files]

    # Released files are forgotten.
    release_checkpoint_page_cache()
    assert len(released) == 2


def test_release_skips_missing_files(tmp_path, released):
    present = tmp_path / "model-0.safetensors"
    present.write_bytes(b"weights")
    record_checkpoint_files([str(tmp_path / "missing.safetensors"), str(present)])

    release_checkpoint_page_cache()
    assert released == [os.stat(present).st_ino]
