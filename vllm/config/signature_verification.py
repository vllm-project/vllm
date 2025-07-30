# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from collections.abc import Iterable
from dataclasses import field
from pathlib import Path
from typing import Literal, Optional

import model_signing
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from vllm.config.utils import config
from vllm.logger import init_logger

logger = init_logger(__name__)

VerificationMethod = Literal["sigstore", "certificate", "key"]

# yapf: enable


@config
@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class SignatureVerificationConfig:
    """Signature verification configuration."""

    verification_method: Optional[VerificationMethod] = None
    """The signature verification method to use."""
    signature: str = "model.sig"
    """Path to the signature file. If only a filename is given, then a file
    with this name is expected to hold the signature in the model's
    directory."""
    ignore_paths: list[str] = field(default_factory=list)
    """File paths to ignore when verifying. Absolute file paths will be used
    as-is, others will be made absolute by prepending the model's
    directory."""
    ignore_git_paths: bool = True
    """Ignore git-related files when verifying."""
    identity: str = ""
    """The email identity that signed with the 'sigstore' method."""
    identity_provider: str = ""
    """The email identity provider that was used when signing with the
    'sigstore' method."""
    use_staging: bool = False
    """Use the Sigstore staging server."""
    certificate_chain: list[str] = field(default_factory=list)
    """A PEM file containing the certificate chain needed for verifying with
    method 'certificate'."""
    log_fingerprints: bool = False
    """Whether to log certificate fingerprints when using the 'certificate'
    method."""
    public_key: Optional[str] = None
    """A PEM file containing the public key required for signature
    verification when using the 'key' method."""
    _verification_done: bool = False
    """Set to True once signature verification was done."""

    def signature_verification_requested(self) -> bool:
        """Check whether signature verification was requested."""
        return self.verification_method is not None

    def signature_verification_needed(self) -> bool:
        """Check whether signature verification is requested but has
        not been done yet."""
        return self.signature_verification_requested() and \
               not self._verification_done

    def _verify_sigstore(
        self,
        model_path: Path,
        signature: Path,
        ignore_paths: Iterable[Path],
        ignore_git_paths: bool,
        identity: str,
        identity_provider: str,
        use_staging: bool,
    ) -> None:
        """Verify using Sigstore"""
        try:
            model_signing.verifying.Config().use_sigstore_verifier(
                identity=identity,
                oidc_issuer=identity_provider,
                use_staging=use_staging,
            ).set_hashing_config(
                model_signing.hashing.Config().set_ignored_paths(
                    paths=list(ignore_paths) + [signature],
                    ignore_git_paths=ignore_git_paths,
                )).verify(model_path, signature)
        except Exception as err:
            logger.error("Verification failed with error: %s", err)
            raise ValueError("Sigstore verification failed on %s",
                             model_path) from err

    def _verify_certificate(
        self,
        model_path: Path,
        signature: Path,
        ignore_paths: Iterable[Path],
        ignore_git_paths: bool,
        certificate_chain: Iterable[Path],
        log_fingerprints: bool,
    ) -> None:
        """Verify using a certificate chain"""
        try:
            model_signing.verifying.Config().use_certificate_verifier(
                certificate_chain=certificate_chain,
                log_fingerprints=log_fingerprints,
            ).set_hashing_config(
                model_signing.hashing.Config().set_ignored_paths(
                    paths=list(ignore_paths) + [signature],
                    ignore_git_paths=ignore_git_paths,
                )).verify(model_path, signature)
        except Exception as err:
            logger.error("Verification failed with error: %s", err)
            raise ValueError(
                "Signature verification with certificate failed "
                "on %s", model_path) from err

    def _verify_private_key(
        self,
        model_path: Path,
        signature: Path,
        ignore_paths: Iterable[Path],
        ignore_git_paths: bool,
        public_key: Path,
    ) -> None:
        """Verify using a public key (paired with a private one)."""
        try:
            model_signing.verifying.Config().use_elliptic_key_verifier(
                public_key=public_key, ).set_hashing_config(
                    model_signing.hashing.Config().set_ignored_paths(
                        paths=list(ignore_paths) + [signature],
                        ignore_git_paths=ignore_git_paths,
                    )).verify(model_path, signature)
        except Exception as err:
            logger.error("Verification failed with error: %s", err)
            raise ValueError(
                "Signature verification with public key failed "
                "on %s", model_path) from err

    def verify_signature(self, model: str) -> None:
        """Verify the signature of a model."""
        model_path = Path(model)

        def make_abs(model_path: Path, file: str) -> Path:
            if os.path.isabs(file):
                return Path(file)
            return model_path / file

        ignore_paths = [make_abs(model_path, f) for f in self.ignore_paths]
        signature = make_abs(model_path, self.signature)

        if self.verification_method == "sigstore":
            self._verify_sigstore(model_path, signature, ignore_paths,
                                  self.ignore_git_paths, self.identity,
                                  self.identity_provider, self.use_staging)
        elif self.verification_method == "certificate":
            cert_chain_paths = [
                make_abs(model_path, f) for f in self.certificate_chain
            ]

            self._verify_certificate(model_path, signature, ignore_paths,
                                     self.ignore_git_paths, cert_chain_paths,
                                     self.log_fingerprints)
        elif self.verification_method == "key":
            if not self.public_key:
                raise ValueError("Missing public key")

            self._verify_private_key(model_path, signature, ignore_paths,
                                     self.ignore_git_paths,
                                     make_abs(model_path, self.public_key))
        else:
            raise NotImplementedError(
                "Unsupported signature verification method "
                f"'{self.verification_method}'")
        logger.info("Signature verification succeeded on %s", model)
        self._verification_done = True
