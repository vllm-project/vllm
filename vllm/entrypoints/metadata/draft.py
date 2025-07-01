# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.entrypoints.metadata.base import BriefMetadata, Metadata

if TYPE_CHECKING:
    pass


class DraftBrief(BriefMetadata):
    pass


@dataclass
class DraftMetadata(Metadata):
    brief: DraftBrief
