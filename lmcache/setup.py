# SPDX-License-Identifier: Apache-2.0
"""LMCache setup script — policy-driven extension build.

Uses the strategy pattern so that each platform lives in its own file
under ``setup_extensions/build_profiles/``.  Adding a new platform requires
zero changes to this file.
"""

# Standard
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).parent
# Ensure the project root is importable when ``setup.py`` runs inside a
# PEP 517 build subprocess (where CWD is not added to ``sys.path``).
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Third Party
from setuptools import find_packages, setup  # noqa: E402

# First Party
from setup_extensions import BuildPolicy, BuildProfile  # noqa: E402


def _read_requirements(path: Path) -> list[str]:
    if not path.exists():
        return []
    reqs: list[str] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            reqs.append(line)
    return reqs


if __name__ == "__main__":
    policy = BuildPolicy()
    profile = policy.resolve_profile()
    ext_modules, cmdclass, req_file = policy.collect_extensions(profile)

    install_requires = _read_requirements(ROOT_DIR / "requirements" / "common.txt")
    extras_require: dict[str, list[str]] = {}
    if (
        not BuildProfile.is_gpu_ext_disabled()
        and req_file is not None
        and profile is not None
    ):
        install_requires += _read_requirements(ROOT_DIR / "requirements" / req_file)
        for extra_name, extra_req_file in profile.extras_requirements().items():
            extras_require[extra_name] = _read_requirements(
                ROOT_DIR / "requirements" / extra_req_file
            )

    setup(
        packages=find_packages(exclude=("csrc",)),
        ext_modules=ext_modules,
        cmdclass=cmdclass,
        include_package_data=True,
        install_requires=install_requires,
        extras_require=extras_require,
    )
