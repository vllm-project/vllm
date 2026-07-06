# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Connection configuration for the object store secondary tier."""

from dataclasses import dataclass


@dataclass
class ObjStoreConfig:
    """Connection parameters for an object store backend.

    When ``access_key`` and ``secret_key`` are left empty the NIXL OBJ
    plugin falls back to the AWS SDK default credential provider chain
    (IAM roles, environment variables, credential files, etc.), which
    enables workload-identity based auth on Kubernetes.
    """

    bucket: str
    endpoint_override: str
    access_key: str = ""
    secret_key: str = ""
    session_token: str = ""
    region: str = ""
    scheme: str = "http"
    ca_bundle: str = ""

    def __repr__(self) -> str:
        masked = {f: "***" if getattr(self, f) else ""
                  for f in ("access_key", "secret_key", "session_token")}
        return (f"ObjStoreConfig(bucket={self.bucket!r}, "
                f"endpoint_override={self.endpoint_override!r}, "
                f"access_key={masked['access_key']!r}, "
                f"secret_key={masked['secret_key']!r}, "
                f"session_token={masked['session_token']!r}, "
                f"region={self.region!r}, scheme={self.scheme!r}, "
                f"ca_bundle={self.ca_bundle!r})")

    def to_nixl_params(self) -> dict[str, str]:
        """Build the NIXL backend params dict.

        Credential and optional fields are only included when non-empty
        so that the AWS SDK default credential chain can activate.
        """
        params: dict[str, str] = {
            "bucket": self.bucket,
            "endpoint_override": self.endpoint_override,
            "scheme": self.scheme,
        }
        # Omit empty optional fields so the NIXL OBJ plugin's underlying
        # AWS SDK can fall back to its default credential provider chain
        # (IAM roles, env vars, credential files, etc.).
        # https://github.com/ai-dynamo/nixl/blob/main/src/plugins/obj/README.md
        for key in ("access_key", "secret_key", "session_token",
                     "region", "ca_bundle"):
            value = getattr(self, key)
            if value:
                params[key] = value
        return params
