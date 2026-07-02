# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

from typing_extensions import override

from vllm.v1.kv_offload.base import LoadStoreSpec


@dataclass
class FileSystemLoadStoreSpec(LoadStoreSpec):
    """Worker-visible filesystem transfer spec.

    file_paths are the committed block files. temp_file_paths are present only
    for stores; workers write temp files and the scheduler-side tier commits
    them after every worker has reported completion.
    """

    file_paths: list[str]
    block_size: int
    temp_file_paths: list[str] | None = None
    num_ranks: int = 1

    @staticmethod
    @override
    def medium() -> str:
        return "file_system"
