# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import NamedTuple

import torch
import torch.nn.functional as F


def check_embeddings_close(
    *,
    embeddings_0_lst: Sequence[list[float]],
    embeddings_1_lst: Sequence[list[float]],
    name_0: str,
    name_1: str,
    tol: float = 1e-3,
) -> None:
    assert len(embeddings_0_lst) == len(embeddings_1_lst)

    for prompt_idx, (embeddings_0, embeddings_1) in enumerate(
            zip(embeddings_0_lst, embeddings_1_lst)):
        assert len(embeddings_0) == len(embeddings_1), (
            f"Length mismatch: {len(embeddings_0)} vs. {len(embeddings_1)}")

        sim = F.cosine_similarity(torch.tensor(embeddings_0),
                                  torch.tensor(embeddings_1),
                                  dim=0)

        fail_msg = (f"Test{prompt_idx}:"
                    f"\n{name_0}:\t{embeddings_0[:16]!r}"
                    f"\n{name_1}:\t{embeddings_1[:16]!r}")

        assert sim >= 1 - tol, fail_msg


def matryoshka_fy(tensor, dimensions):
    tensor = torch.tensor(tensor)
    tensor = tensor[..., :dimensions]
    tensor = F.normalize(tensor, p=2, dim=1)
    return tensor


class EmbedModelInfo(NamedTuple):
    name: str
    is_matryoshka: bool
    architecture: str = ""
    enable_test: bool = True
