# SPDX-License-Identifier: Apache-2.0
"""Typed domain errors for cache-control operations.

Transport-agnostic: each names *what* failed, not an HTTP status. The HTTP layer
(``http_apis/error_handlers.py``) maps each subclass to a status code; an RPC /
CLI / test caller can ``except`` the specific type. The subclasses are
intentionally behaviorless -- the type itself is the signal -- so the message is
passed in at the raise site.
"""


class CacheControlError(Exception):
    """Base for cache-control domain failures."""


class InvalidRequest(CacheControlError): ...  # malformed / unsupported / not-ready


class NotFound(CacheControlError): ...  # referenced adapter / job missing


class Unavailable(CacheControlError): ...  # backend can't serve the request
