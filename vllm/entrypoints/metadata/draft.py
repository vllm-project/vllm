# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.entrypoints.metadata.base import (BriefMetadata, DetailMetadata,
                                            Metadata, PoolerConfigMetadata)

if TYPE_CHECKING:
    pass


class DraftBrief(BriefMetadata):
    pass


class DraftDetail(DetailMetadata):
    pass


@dataclass
class DraftMetadata(Metadata):
    brief: DraftBrief
    detail: DraftDetail
    pooler_config: PoolerConfigMetadata
