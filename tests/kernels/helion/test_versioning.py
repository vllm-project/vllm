# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CI policy tests for Helion kernel versioning.

These tests enforce lifecycle rules for versioned kernels:
- Max 2 versions per kernel
- Signature compatibility across versions
- Core platform coverage for newest version (when >1 version)
"""

import inspect

import pytest

from vllm.utils.import_utils import has_helion

if not has_helion():
    pytest.skip(
        "Helion is not installed. Install with: pip install vllm[helion]",
        allow_module_level=True,
    )

from vllm.kernels.helion.config_manager import ConfigManager
from vllm.kernels.helion.platforms import CORE_PLATFORMS
from vllm.kernels.helion.register import _REGISTERED_KERNELS


class TestVersioningPolicy:
    """CI policy tests for kernel versioning lifecycle."""

    def test_registry_is_not_empty(self):
        """Ensure ops were actually registered (guard against missing imports)."""
        assert _REGISTERED_KERNELS, (
            "No Helion kernels registered. Ensure vllm.kernels.helion.ops "
            "is imported before running versioning policy tests."
        )

    def test_max_two_versions(self):
        """No kernel may have more than 2 registered versions."""
        violations = [
            f"  {name}: has {len(versions)} versions ({sorted(versions.keys())})"
            for name, versions in _REGISTERED_KERNELS.items()
            if len(versions) > 2
        ]
        assert not violations, (
            "POLICY: Each kernel may have at most 2 registered versions "
            "(current + previous). A new version cannot be added until the "
            "oldest version is removed.\n"
            "Violations:\n" + "\n".join(violations)
        )

    def test_newest_version_has_core_coverage(self):
        """When >1 version exists, newest must cover all CORE_PLATFORMS."""
        config_manager = ConfigManager()
        violations = []

        for name, versions in _REGISTERED_KERNELS.items():
            if len(versions) < 2:
                continue

            newest_ver = max(versions)
            newest_wrapper = versions[newest_ver]
            config_set = config_manager.load_config_set(newest_wrapper.versioned_name)
            covered_platforms = frozenset(
                p for p in config_set.get_platforms() if config_set.get_config_keys(p)
            )

            missing = CORE_PLATFORMS - covered_platforms
            if missing:
                violations.append(
                    f"  {name}: v{newest_ver} ({newest_wrapper.versioned_name}) "
                    f"is missing configs for {sorted(missing)}. "
                    f"Run 'python scripts/autotune_helion_kernels.py "
                    f"--kernel {name}' on the missing platforms."
                )

        assert not violations, (
            "POLICY: When multiple versions of a kernel coexist, the newest "
            f"version must have configs for all CORE_PLATFORMS "
            f"{sorted(CORE_PLATFORMS)} before it can be registered alongside "
            f"an older version.\n"
            "Violations:\n" + "\n".join(violations)
        )

    def test_signature_compatibility(self):
        """All versions of a kernel must have identical parameter names/annotations."""
        violations = []

        for name, versions in _REGISTERED_KERNELS.items():
            if len(versions) < 2:
                continue

            sigs = {}
            for ver, wrapper in sorted(versions.items()):
                sig = inspect.signature(wrapper.raw_kernel_func)
                param_info = [
                    (p_name, p.annotation) for p_name, p in sig.parameters.items()
                ]
                sigs[ver] = param_info

            ver_list = sorted(sigs)
            reference_ver = ver_list[0]
            reference_sig = sigs[reference_ver]

            for ver in ver_list[1:]:
                if sigs[ver] != reference_sig:
                    violations.append(
                        f"  {name}: v{ver} has different signature than "
                        f"v{reference_ver}\n"
                        f"    v{reference_ver}: {reference_sig}\n"
                        f"    v{ver}: {sigs[ver]}"
                    )

        assert not violations, (
            "POLICY: All versions of a kernel must have identical function "
            "signatures (parameter names and type annotations), because the "
            "caller is version-unaware and passes the same arguments "
            "regardless of which version is resolved.\n"
            "Violations:\n" + "\n".join(violations)
        )
