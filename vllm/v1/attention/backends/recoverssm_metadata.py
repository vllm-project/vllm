# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import abc
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RecoverSSMPostprocessMetadata:
    """Metadata used during postprocessing for align-mode prefix caching."""

    num_spec_decodes: int
    request_indices: torch.Tensor | None
    block_table: torch.Tensor
    num_computed_tokens: torch.Tensor
    block_size: int


class RecoverSSMMetadata(abc.ABC):
    @abc.abstractmethod
    def commit_recoverssm_state(
        self, num_accepted_tokens: torch.Tensor
    ) -> RecoverSSMPostprocessMetadata | None:
        raise NotImplementedError


def recoverssm_spec_rows(
    is_prefilling_cpu: torch.Tensor,
    query_start_loc_cpu: torch.Tensor,
    num_decode_draft_tokens_cpu: torch.Tensor,
    num_reqs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Which batch rows take the RecoverSSM spec path, shared by the RecoverSSM metadata
    builders (GDN, PLE).

    Every active decode row does (a draft-less step is a window of one token), plus
    every row that has drafts. Only the first ``num_reqs`` rows are real; padding rows
    never take the spec path. Returns the draft-count vector to hand to the base builder
    (1 for spec rows, -1 otherwise, padding included) and the spec rows in batch order,
    which is also the order of the base builder's spec state indices."""
    query_lens = query_start_loc_cpu[: num_reqs + 1].diff()
    active_decode = (~is_prefilling_cpu[:num_reqs]) & (query_lens > 0)
    spec_mask = (num_decode_draft_tokens_cpu[:num_reqs] >= 0) | active_decode
    drafts = torch.full_like(num_decode_draft_tokens_cpu, -1)
    drafts[:num_reqs] = torch.where(spec_mask, 1, -1).to(drafts.dtype)
    return drafts, spec_mask.nonzero().flatten()


def recoverssm_request_indices(
    rows: torch.Tensor, num_spec_decodes: int, device: torch.device
) -> torch.Tensor | None:
    """None when the spec rows are exactly 0..n-1, else their batch rows (the commit
    plan's request mapping). Raises if the base builder counted a different number of
    spec rows than the classification above."""
    if rows.numel() != num_spec_decodes:
        raise ValueError(
            f"RecoverSSM: {rows.numel()} spec rows classified, but the builder "
            f"produced {num_spec_decodes}"
        )
    if num_spec_decodes == 0 or int(rows[-1]) == num_spec_decodes - 1:
        return None
    return rows.to(torch.int32).to(device, non_blocking=True)
