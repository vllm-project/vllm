# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any, Optional


class DisaggregatedRequestManagerFactory:

    @classmethod
    def build_request_manager(cls, kv_transfer_params: Optional[dict[str,
                                                                     Any]]):
        # Lazily create the request manager based on the kv_transfer_params
        #  (eg decode first..)
        pass
