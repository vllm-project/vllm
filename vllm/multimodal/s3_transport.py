# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Frontend transport for multimodal tensors in S3-compatible object stores."""

import contextlib
from typing import Any
from urllib.parse import urlsplit

import torch
from pydantic import Field

import vllm.envs as envs
from vllm.config.mm_tensor_refs import MMTensorRefsConfig
from vllm.exceptions import RetryableRequestError
from vllm.logger import init_logger
from vllm.multimodal.tensor_transport import (
    BaseMMTensorReference,
    MMTensorTransportError,
    decode_mm_tensor_bundle,
    verify_mm_tensor_digest,
)
from vllm.utils.import_utils import PlaceholderModule

try:
    import boto3
    import botocore.config
    import botocore.exceptions
except ImportError:
    boto3 = PlaceholderModule("boto3")  # type: ignore[assignment]
    botocore = PlaceholderModule("botocore")  # type: ignore[assignment]

logger = init_logger(__name__)


class S3MMReference(BaseMMTensorReference):
    """Reference to one processed multimodal tensor bundle in S3."""

    uri: str = Field(min_length=1, max_length=2048)
    """An `s3://bucket/key` URI under the configured `s3_allowed_uri_prefix`."""


def _parse_s3_uri(uri: str, *, allow_empty_key: bool = False) -> tuple[str, str]:
    parsed = urlsplit(uri)
    key = parsed.path.removeprefix("/")
    if (
        parsed.scheme.lower() != "s3"
        or not parsed.netloc
        or (not key and not allow_empty_key)
        or parsed.query
        or parsed.fragment
        or any(part in (".", "..") for part in key.split("/"))
    ):
        raise ValueError(f"Invalid S3 URI {uri!r}; expected s3://bucket/key")
    return parsed.netloc, key


class S3MMReader:
    """Load client-provided multimodal tensors from an S3-compatible store.

    Endpoint and credentials come from the server environment: `S3_ENDPOINT_URL`
    if set, otherwise the standard boto3 configuration. Every caller of the
    endpoint can reference every object the server can read under
    `config.s3_allowed_uri_prefix`.
    """

    def __init__(self, config: MMTensorRefsConfig, s3_client: Any | None = None):
        bucket, key = _parse_s3_uri(config.s3_allowed_uri_prefix, allow_empty_key=True)
        self._allowed_bucket, self._allowed_key = bucket, key.rstrip("/")
        self._max_object_bytes = config.max_object_bytes
        if s3_client is None:
            s3_client = boto3.session.Session().client(
                "s3",
                endpoint_url=envs.S3_ENDPOINT_URL,
                config=botocore.config.Config(
                    max_pool_connections=max(10, config.max_concurrent_reads)
                ),
            )
        self._s3_client = s3_client

    def validate(self, reference: S3MMReference) -> tuple[str, str]:
        """Validate a reference and return its server-approved bucket and key."""
        bucket, key = _parse_s3_uri(reference.uri)
        if bucket != self._allowed_bucket or (
            self._allowed_key and not key.startswith(f"{self._allowed_key}/")
        ):
            raise ValueError("S3 URI must be under the configured prefix")
        if reference.nbytes > self._max_object_bytes:
            raise ValueError(
                f"Multimodal tensor object declares {reference.nbytes} bytes, "
                f"exceeding the configured limit of {self._max_object_bytes} bytes"
            )
        return bucket, key

    def read(self, reference: S3MMReference) -> dict[str, torch.Tensor]:
        """Fetch, bound, verify, and decode one referenced tensor bundle.

        A missing, inaccessible or mismatched object raises the same error, so
        a caller can't learn about objects it doesn't already know.

        Raises:
            ValueError: If the reference is not allowed.
            MMTensorTransportError: If the object can't be loaded or decoded.
            RetryableRequestError: If the store failed or was unreachable.

        """
        bucket, key = self.validate(reference)

        def unavailable(detail: str) -> MMTensorTransportError:
            logger.info("Could not load %r: %s", reference.uri, detail)
            return MMTensorTransportError(
                f"Could not load multimodal tensor object {reference.uri!r}"
            )

        try:
            response = self._s3_client.get_object(Bucket=bucket, Key=key)
        except botocore.exceptions.ClientError as e:
            status = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if status is not None and status < 500 and status not in (408, 429):
                raise unavailable(f"HTTP {status}") from e
            raise RetryableRequestError(
                f"Could not load multimodal tensor object {reference.uri!r}"
            ) from e
        except botocore.exceptions.BotoCoreError as e:
            raise RetryableRequestError(
                f"Could not load multimodal tensor object {reference.uri!r}"
            ) from e

        body = response.get("Body")
        if body is None:
            raise unavailable("response has no body")

        try:
            with contextlib.closing(body):
                content_length = response.get("ContentLength")
                if content_length != reference.nbytes:
                    raise unavailable(
                        f"object has {content_length} bytes, "
                        f"reference declares {reference.nbytes}"
                    )
                payload = body.read(reference.nbytes + 1)
        except (botocore.exceptions.BotoCoreError, OSError) as e:
            raise RetryableRequestError(
                f"Could not read multimodal tensor object {reference.uri!r}"
            ) from e

        if len(payload) != reference.nbytes:
            raise unavailable(
                f"read {len(payload)} bytes, reference declares {reference.nbytes}"
            )
        if not verify_mm_tensor_digest(payload, reference.digest):
            raise unavailable("SHA-256 digest mismatch")
        return decode_mm_tensor_bundle(payload, location=repr(reference.uri))
