# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, NamedTuple, TypeVar

_T = TypeVar("_T")


class MultiModalPlaceholderMap:
    """
    Relates multi-modal embeddings to their corresponding placeholders.

    Note: This is only used in V0.
    """

    class IndexMap(NamedTuple):
        src: list[int]
        dest: list[int]

    src_ranges: list[range]
    """
    The indices of the multi-modal embeddings that will replace the
    corresponding placeholder embeddings pointed to by ``dest_ranges``.
    """

    src_len: int
    """
    The total number of flattened multi-modal embeddings.
    """

    dest_ranges: list[range]
    """
    The indices of the placeholder embeddings that will be replaced by the
    multimodal embeddings.
    """

    dest_len: int
    """
    The total number of embeddings in the destination tensor.
    """

    def __init__(self):
        self.src_ranges = []
        self.src_len = 0
        self.dest_ranges = []
        self.dest_len = 0

    def extend(self, other: "MultiModalPlaceholderMap"):
        """
        Adds the placeholders from another ``MultiModalPlaceholderMap`` to this
        instance based on the source and destination tensors being
        concatenated.
        """

        self.src_ranges.extend(
            range(self.src_len + r.start, self.src_len + r.stop)
            for r in other.src_ranges)
        self.src_len += other.src_len
        self.dest_ranges.extend(
            range(self.dest_len + r.start, self.dest_len + r.stop)
            for r in other.dest_ranges)
        self.dest_len += other.dest_len

    def index_map(self) -> "IndexMap":
        """
        Finalizes the placeholder map into lists of indices that can be used to
        index the source and destination tensors.
        """

        src_indices = [i for r in self.src_ranges for i in r]
        dest_indices = [i for r in self.dest_ranges for i in r]

        if len(src_indices) != len(dest_indices):
            raise ValueError(
                f"The number of source ({len(src_indices)}) and destination "
                f"indices ({len(dest_indices)}) must be the same.")

        return self.IndexMap(src=src_indices, dest=dest_indices)


class MediaIO(ABC, Generic[_T]):

    @abstractmethod
    def load_bytes(self, data: bytes) -> _T:
        raise NotImplementedError

    @abstractmethod
    def load_base64(self, media_type: str, data: str) -> _T:
        """
        List of media types:
        https://www.iana.org/assignments/media-types/media-types.xhtml
        """
        raise NotImplementedError

    @abstractmethod
    def load_file(self, filepath: Path) -> _T:
        raise NotImplementedError
