# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Python wrapper for the Go OCI client library."""

import ctypes
import json
import os
from pathlib import Path
from typing import Optional


class ManifestResult(ctypes.Structure):
    """Result structure for PullManifest."""

    _fields_ = [
        ("manifest_json", ctypes.c_char_p),
        ("error", ctypes.c_char_p),
    ]


class ErrorResult(ctypes.Structure):
    """Result structure for error-only operations."""

    _fields_ = [
        ("error", ctypes.c_char_p),
    ]


class OciGoClient:
    """Python wrapper for the Go OCI client library.

    This client uses the go-containerregistry library to pull OCI images
    with proper authentication support via Docker config.
    """

    def __init__(self):
        """Initialize the OCI Go client by loading the shared library."""
        # Find the library
        lib_dir = Path(__file__).parent / "oci_go"
        lib_path = lib_dir / "liboci.so"

        # Try .so first, then .dylib (macOS), then .a (static)
        if not lib_path.exists():
            lib_path = lib_dir / "liboci.dylib"
        if not lib_path.exists():
            # For static library, we need to use cdll differently
            lib_path = lib_dir / "liboci.a"
            if not lib_path.exists():
                raise RuntimeError(
                    f"OCI Go library not found. Expected at {lib_dir}/liboci.{{so,dylib,a}}"
                )

        # Load the library
        self.lib = ctypes.CDLL(str(lib_path))

        # Set up function signatures
        self._setup_functions()

    def _setup_functions(self):
        """Configure ctypes function signatures."""
        # PullManifest
        self.lib.PullManifest.argtypes = [ctypes.c_char_p]
        self.lib.PullManifest.restype = ManifestResult

        # PullBlob
        self.lib.PullBlob.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
        ]
        self.lib.PullBlob.restype = ErrorResult

        # TestAuthentication
        self.lib.TestAuthentication.argtypes = [ctypes.c_char_p]
        self.lib.TestAuthentication.restype = ErrorResult

    def pull_manifest(self, image_ref: str) -> dict:
        """Pull OCI manifest for the given image reference.

        Args:
            image_ref: OCI image reference (e.g., "docker.io/user/image:tag")

        Returns:
            Parsed manifest as a dictionary

        Raises:
            RuntimeError: If pulling the manifest fails
        """
        result = self.lib.PullManifest(image_ref.encode("utf-8"))

        if result.error:
            error_msg = result.error.decode("utf-8")
            raise RuntimeError(f"Failed to pull manifest: {error_msg}")

        if not result.manifest_json:
            raise RuntimeError("No manifest returned")

        manifest_str = result.manifest_json.decode("utf-8")
        return json.loads(manifest_str)

    def pull_blob(self, image_ref: str, digest: str, output_path: str) -> None:
        """Pull a blob (layer) from the OCI registry.

        Args:
            image_ref: OCI image reference
            digest: Blob digest (e.g., "sha256:abc123...")
            output_path: Path where the blob should be saved

        Raises:
            RuntimeError: If pulling the blob fails
        """
        result = self.lib.PullBlob(
            image_ref.encode("utf-8"),
            digest.encode("utf-8"),
            output_path.encode("utf-8"),
        )

        if result.error:
            error_msg = result.error.decode("utf-8")
            raise RuntimeError(f"Failed to pull blob: {error_msg}")

    def test_authentication(self, image_ref: str) -> Optional[str]:
        """Test if authentication is working for the given image reference.

        Args:
            image_ref: OCI image reference to test

        Returns:
            None if authentication succeeds, error message otherwise
        """
        result = self.lib.TestAuthentication(image_ref.encode("utf-8"))

        if result.error:
            error_msg = result.error.decode("utf-8")
            return error_msg
        return None
