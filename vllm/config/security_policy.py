# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import logging
import os
from collections.abc import Iterable
from dataclasses import field
from pathlib import Path
from typing import Any, Literal, Optional

import jsonschema
import model_signing
import regex as re
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from typing_extensions import Self

from vllm.config.utils import config
from vllm.logger import init_logger

logger = init_logger(__name__)

# yapf: enable

SECURITY_POLICY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["policy"],
    "properties": {
        "policy": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "signatures": {
                    "description":
                    "A dictionary describing the signers of AI models",
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "loras": {
                            "$ref": "#/definitions/signed-object"
                        },
                        "models": {
                            "$ref": "#/definitions/signed-object"
                        },
                        "signers": {
                            "description":
                            "A dictionary of signer descriptions",
                            "type": "object",
                            "patternProperties": {
                                "": {
                                    "description":
                                    "A dictionary describing a signer",
                                    "type":
                                    "object",
                                    "required": ["verification_method"],
                                    "additionalProperties":
                                    False,
                                    "properties": {
                                        "verification_method": {
                                            "description":
                                            "The signature verification method",
                                            "type":
                                            "string",
                                            "enum": [
                                                "sigstore", "certificate",
                                                "key", "skip"
                                            ]
                                        },
                                        "identity": {
                                            "type": "string",
                                            "format": "email"
                                        },
                                        "identity_provider": {
                                            "type": "string"
                                        },
                                        "certificate_chain": {
                                            "type": "array",
                                            "items": {
                                                "type": "string"
                                            }
                                        },
                                        "public_key": {
                                            "type": "string"
                                        }
                                    },
                                    "allOf": [{
                                        "if": {
                                            "properties": {
                                                "verification_method": {
                                                    "const": "sigstore"
                                                }
                                            }
                                        },
                                        "then": {
                                            "required":
                                            ["identity", "identity_provider"]
                                        }
                                    }, {
                                        "if": {
                                            "properties": {
                                                "verification_method": {
                                                    "const": "certificate"
                                                }
                                            }
                                        },
                                        "then": {
                                            "required": ["certificate_chain"]
                                        }
                                    }, {
                                        "if": {
                                            "properties": {
                                                "verification_method": {
                                                    "const": "key"
                                                }
                                            }
                                        },
                                        "then": {
                                            "required": ["public_key"]
                                        }
                                    }, {
                                        "if": {
                                            "properties": {
                                                "verification_method": {
                                                    "const": "skip"
                                                }
                                            }
                                        },
                                        "then": {
                                            "required": []
                                        }
                                    }]
                                }
                            }
                        }
                    }
                }
            }
        }
    },
    "definitions": {
        "signed-object": {
            "description": "A dictionary of signed objects with path as key",
            "type": "object",
            "patternProperties": {
                "": {
                    "description": "",
                    "type": "object",
                    "required": ["signer"],
                    "additionalProperties": False,
                    "properties": {
                        "signer": {
                            "type": "string"
                        },
                        "signature": {
                            "type": "string"
                        },
                        "ignore_paths": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            }
                        },
                        "ignore_git_paths": {
                            "type": "boolean"
                        },
                        "use_staging": {
                            "type": "boolean"
                        },
                        "log_fingerprints": {
                            "type": "boolean"
                        },
                    }
                }
            }
        }
    }
}


class SignatureVerificationError(Exception):
    """A signature verification error occurred"""
    pass


VerificationMethod = Literal["sigstore", "certificate", "key", "skip"]


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
            raise SignatureVerificationError(
                f"Sigstore verification failed on {model_path}") from err

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
        if log_fingerprints:
            logging.getLogger("model_signing").setLevel(logging.INFO)
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
            raise SignatureVerificationError("Signature verification with "
                                             "certificate failed "
                                             f"on {model_path}") from err
        finally:
            logging.getLogger("model_signing").setLevel(logging.WARNING)

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
            raise SignatureVerificationError(
                "Signature verification with public key failed "
                f"on {model_path}") from err

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
        elif self.verification_method != "skip":
            raise NotImplementedError(
                "Unsupported signature verification method "
                f"'{self.verification_method}'")

        if self.verification_method == "skip":
            logger.info("Skipped signature verification on %s", model)
        else:
            logger.info("Signature verification succeeded on %s", model)


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class SecurityPolicy:

    policy_json: Optional[dict] = None

    def __init__(self, policy_json) -> None:
        self.policy_json = policy_json
        self.init()

    def init(self) -> None:
        self.regex_map = {}
        self.validate()
        self.models_verified = set()

    def __post_init__(self) -> None:
        self.init()

    def validate(self: Self) -> None:
        """Validate the policy with scheme and correctness of regular
        expressions and check consistency."""
        jsonschema.validate(instance=self.policy_json,
                            schema=SECURITY_POLICY_SCHEMA)

        # Get the ids of all signers
        signers = self.policy_json.get("policy", {})\
            .get("signatures", {}).get("signers", {}).keys()

        self.compile_regexs("models", signers)
        self.compile_regexs("loras", signers)

    def compile_regexs(self, object_types: str, signers: list[str]) -> None:
        """Compile the regular expressions and while walking all the entries
        check for consistency with availability of all signers"""

        param_map = self.policy_json.get("policy", {})\
            .get("signatures", {}).get(object_types, {})

        self.regex_map[object_types] = []
        for key, value in param_map.items():
            try:
                if key.startswith("regex:"):
                    regex = re.compile(key[6:])
                    self.regex_map[object_types].append((regex, value))
            except Exception as e:
                raise ValueError(f"Invalid regular expression: {key}") from e

            signer = value["signer"]
            if signer not in signers:
                raise ValueError(
                    f"Signer {signer} cannot be found in 'signers' map.")

    @classmethod
    def from_file(cls: Self, security_policy: str) -> Self:
        """Instantiate a SecurityPolicy from a file"""
        with open(security_policy) as file:
            policy_json = json.load(file)

        return SecurityPolicy(policy_json)

    def __get_model_params(self, param_map: dict, object_type: str,
                           path: str) -> None:

        # Find the parameters by the exact path
        model_params = param_map.get(path)
        if not model_params and len(path) > 0 and path[-1] != '/':
            # One more try with '/' appended to path
            # Paths in policy should all end in '/'
            model_params = param_map.get(path + '/')
        if not model_params:
            # Find the parameters by assuming that the path
            # in the policy is a regular expression
            for regex, value in self.regex_map[object_type]:
                if regex.match(path):
                    model_params = value
                    break
        return model_params

    def getSignatureVerificationConfig(
            self: Self, object_type: str,
            path: str) -> SignatureVerificationConfig:
        """ Get the SignatureVerificationConfig for the given path."""

        # Get the map for all models for example
        param_map = self.policy_json.get("policy", {})\
            .get("signatures", {}).get(object_type)

        model_params = self.__get_model_params(param_map, object_type, path)
        if not model_params:
            raise ValueError(
                f"No signature verification parameters found for '{path}'")

        # Get the signer parameter
        signer = model_params.get("signer")
        if not signer:
            raise ValueError(f"Could not find signer for '{path}'")

        signer_params = self.policy_json.get("policy", {})\
            .get("signatures", {}).get("signers", {})\
            .get(signer)
        if not signer_params:
            raise ValueError(
                f"Parameters for signer {signer} could not be found")

        return SignatureVerificationConfig(
            verification_method=signer_params.get("verification_method"),
            signature=model_params.get("signature", "model.sig"),
            ignore_paths=model_params.get("ignore_paths", []),
            ignore_git_paths=model_params.get("ignore_git_paths", True),
            identity=signer_params.get("identity", ""),
            identity_provider=signer_params.get("identity_provider", ""),
            use_staging=signer_params.get("use_staging", False),
            certificate_chain=signer_params.get("certificate_chain", []),
            log_fingerprints=signer_params.get("log_fingerprints", False),
            public_key=signer_params.get("public_key"),
        )

    def model_signature_verification_requested(self: Self) -> bool:
        """Model signature verification is requested if models are specified
        in the policy."""
        return self.policy_json.get("policy", {})\
            .get("signatures", {})\
            .get("models") is not None

    def lora_signature_verification_requested(self: Self) -> bool:
        """LoRA signature verification is requested if loras are specified
        in the policy."""
        return self.policy_json.get("policy", {})\
            .get("signatures", {})\
            .get("loras") is not None

    def model_need_verification(self: Self, model_path: str) -> bool:
        """Check whether a model with the given path  still needs to be
        verified."""
        if model_path in self.models_verified:
            return False
        try:
            svc = self.getSignatureVerificationConfig("models", model_path)
            return svc.verification_method != "skip"
        except Exception:
            return False

    def verify_model_signature(self: Self, model_path: str) -> None:
        """Verify the signature on a model given its path."""
        svc = self.getSignatureVerificationConfig("models", model_path)
        svc.verify_signature(model_path)

        self.models_verified.add(model_path)

    def verify_lora_signature(self: Self, model_path: str) -> None:
        """Verify the signature on a LoRA given its path."""
        svc = self.getSignatureVerificationConfig("loras", model_path)
        svc.verify_signature(model_path)
