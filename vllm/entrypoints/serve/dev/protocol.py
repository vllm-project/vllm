# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pydantic import BaseModel


class ReconfigureRequest(BaseModel):
    max_num_seqs: int | None = None
    max_num_batched_tokens: int | None = None
