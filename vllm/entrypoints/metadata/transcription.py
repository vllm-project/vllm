# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

from vllm.entrypoints.metadata.base import (BriefMetadata, DetailMetadata,
                                            Metadata)


class TranscriptionBrief(BriefMetadata):
    pass


class TranscriptionDetail(DetailMetadata):
    pass


@dataclass
class TranscriptionMetadata(Metadata):
    brief: TranscriptionBrief
    detail: TranscriptionDetail
