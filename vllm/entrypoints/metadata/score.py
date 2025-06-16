# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

from vllm.entrypoints.metadata.base import (BriefMetadata, DetailMetadata,
                                            Metadata, PoolerConfigMetadata)


class ScoreBrief(BriefMetadata):
    pass


class ScoreDetail(DetailMetadata):
    pass


@dataclass
class ScoreMetadata(Metadata):
    brief: ScoreBrief
    detail: ScoreDetail
    pooler_config: PoolerConfigMetadata
