# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import Optional

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from vllm.config.security_policy import SecurityPolicy
from vllm.config.utils import config


@config
@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class SecurityConfig:
    """vLLM Security Configuration"""
    security_policy: Optional[str] = None
    """Security policy file"""
    _security_policy: Optional[SecurityPolicy] = None
    """SecurityPolicy object create from security_policy"""

    def getSecurityPolicy(self) -> Optional[SecurityPolicy]:
        """Get the SecurityPolicy created from the security policy file,
        if available"""
        if self.security_policy and not self._security_policy:
            self._security_policy = SecurityPolicy.from_file(
                self.security_policy)
        return self._security_policy

    def model_signature_verification_needed(self, model_path: str) -> bool:
        """Check whether signature verification was requested and was not
        done, yet. Returns False in case no signature verification was
        requested or the signature verification is already done. Returns
        True if signature verification was request but not done yet."""
        security_policy = self.getSecurityPolicy()
        if not security_policy:
            return False
        if not security_policy.model_signature_verification_requested():
            return False

        return security_policy.model_need_verification(model_path)

    def maybe_verify_model_signature(self, model_path: str) -> None:
        """If there's a security policy that requires model signature
           verification and the model exists at the given path,
           then verify its signature."""
        security_policy = self.getSecurityPolicy()
        if security_policy and \
           security_policy.model_signature_verification_requested() and \
           os.path.exists(model_path) and \
           os.path.isdir(model_path):
            security_policy.verify_model_signature(model_path)
