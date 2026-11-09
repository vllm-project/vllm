# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Connection configuration for the object store secondary tier."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

_RESERVED_BACKEND_PARAM_KEYS = frozenset(
    {
        "bucket",
        "endpoint_override",
        "access_key",
        "secret_key",
        "session_token",
        "region",
        "scheme",
        "ca_bundle",
        "num_threads",
    }
)


def _to_nixl_param(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower()
    return str(value)


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
    access_key: str = field(default="", repr=False)
    secret_key: str = field(default="", repr=False)
    session_token: str = field(default="", repr=False)
    region: str = ""
    scheme: str = "http"
    ca_bundle: str = ""
    backend_params: Mapping[str, Any] = field(default_factory=dict, repr=False)

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
        for key in ("access_key", "secret_key", "session_token", "region", "ca_bundle"):
            value = getattr(self, key)
            if value:
                params[key] = value

        duplicate_keys = _RESERVED_BACKEND_PARAM_KEYS.intersection(self.backend_params)
        if duplicate_keys:
            raise ValueError(
                "backend_params cannot override reserved object store "
                f"parameters: {sorted(duplicate_keys)}"
            )

        params.update(
            {key: _to_nixl_param(value) for key, value in self.backend_params.items()}
        )
        return params
