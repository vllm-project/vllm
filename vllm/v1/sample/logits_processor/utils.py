# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Optional

import regex as re
import torch

from vllm.config import VllmConfig

# <package name>.<entrypoint name> compiled regular expression
package_name_regex = r'[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?'
function_name_or_wildcard_regex = r'[A-Za-z_][A-Za-z0-9_]*|\*'
logitsprocs_package_pattern = re.compile(
    rf'^({package_name_regex})\.({function_name_or_wildcard_regex})$')


def extract_package_and_function(s: str) -> Optional[tuple[str, str]]:
    """Return (package name,entrypoint name) (or `None` if no regex match)"""
    match = logitsprocs_package_pattern.fullmatch(s)
    if match:
        return match.group(1), match.group(2)
    return None


@dataclass
class LogitProcessorCtorArgs:
    vllm_config: VllmConfig
    device: torch.device
    is_pin_memory: bool
