# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.entrypoints.metadata.base import (BriefMetadata, DetailMetadata,
                                            Metadata, PoolerConfigMetadata)

if TYPE_CHECKING:
    pass


class ClassifyBrief(BriefMetadata):
    pass


class ClassifyDetail(DetailMetadata):
    pass


@dataclass
class ClassifyMetadata(Metadata):
    brief: ClassifyBrief
    detail: ClassifyDetail
    pooler_config: PoolerConfigMetadata
