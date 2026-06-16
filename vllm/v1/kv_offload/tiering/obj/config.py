# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Connection configuration for the object store secondary tier."""

from dataclasses import dataclass


@dataclass
class ObjStoreConfig:
    """Connection parameters for an object store backend."""

    bucket: str
    endpoint_override: str
    access_key: str
    secret_key: str
    scheme: str = "http"
    ca_bundle: str = ""

    def to_nixl_params(self) -> dict[str, str]:
        """Build the NIXL backend params dict."""
        params: dict[str, str] = {
            "bucket": self.bucket,
            "endpoint_override": self.endpoint_override,
            "scheme": self.scheme,
            "access_key": self.access_key,
            "secret_key": self.secret_key,
        }
        if self.ca_bundle:
            params["ca_bundle"] = self.ca_bundle
        return params
