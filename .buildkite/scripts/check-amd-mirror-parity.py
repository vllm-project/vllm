# SPDX-License-Identifier: Apache-2.0
"""Check that the AMD steps match the AMD mirror declarations.

Each suite's `mirror.amd` block in `.buildkite/test_areas/*.yaml` declares the label
tier the suite carries in the AMD pipeline and the device class (`device`) it runs on.
`.buildkite/test-amd.yaml` carries the steps, their `mirror_hardwares` selector tags
and their `agent_pool`. When the two files drift, the suite runs on a device class
nobody picked and nothing fails to say so (#57956).

Enforced invariant: a suite whose mirror declares a tier must have a step of the same
name at exactly that tier, carrying the tier's selector tags and the mirror's device as
its agent pool. A same-named step sitting at another tier, a step on the wrong pool and
a step carrying another tier's tags are all violations. Suites legitimately run at
several tiers, so steps are keyed by (tier, name), not by name alone.

Reported but not enforced: mirrors with no step of that name at all, and mirrors whose
declared device is a pool no step uses. Both need a placement call on the AMD side
rather than a mechanical fix.

Usage:
    python3 .buildkite/scripts/check-amd-mirror-parity.py [repo-root]
"""

import glob
import os
import re
import sys

import yaml

LANE_SETS = {
    "mi355": ["amdexperimental", "amdproduction", "amdgfx950nightly", "amdmi355"],
    "mi300": ["amdexperimental", "amdproduction", "amdgfx942nightly", "amdmi300"],
    "mi250": ["amdexperimental", "amdproduction", "amdgfx90anightly", "amdmi250"],
}

TIER_LABEL = re.compile(r"^:amd: \((MI\d+(?: [A-Z]+)?)\) (.+)$")


def tier_and_name(label):
    """Split a `:amd: (TIER) Name` label into its tier and its suite name."""
    match = TIER_LABEL.match(label.strip())
    if not match:
        return None, None
    tier, name = match.groups()
    # Steps are sharded as `Name Shard %N`; mirrors declare the unsharded name.
    return tier, re.sub(r"\s+Shard %N$", "", name.strip())


def load_mirrors(root):
    """(tier, name) -> (device, test_areas file) for every AMD mirror declaration."""
    mirrors = {}
    for path in sorted(glob.glob(os.path.join(root, ".buildkite/test_areas/*.yaml"))):
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        for step in data.get("steps") or []:
            if not isinstance(step, dict):
                continue
            amd = (step.get("mirror") or {}).get("amd") or {}
            tier, name = tier_and_name(amd.get("label", ""))
            if tier:
                mirrors[(tier, name)] = (amd.get("device"), os.path.basename(path))
    return mirrors


def load_steps(root):
    """(tier, name) -> (lanes, agent_pool) for every AMD step in test-amd.yaml."""
    with open(os.path.join(root, ".buildkite/test-amd.yaml")) as f:
        data = yaml.safe_load(f) or []
    steps = data if isinstance(data, list) else data.get("steps") or []
    by_key = {}
    for step in steps:
        if not isinstance(step, dict):
            continue
        tier, name = tier_and_name(step.get("label", ""))
        if tier:
            lanes = step.get("mirror_hardwares") or []
            by_key[(tier, name)] = (lanes, step.get("agent_pool"))
    return by_key


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    mirrors = load_mirrors(root)
    steps = load_steps(root)
    step_tiers = {}
    for tier, name in steps:
        step_tiers.setdefault(name, set()).add(tier)
    pools_in_use = {pool for _, pool in steps.values() if pool}

    violations, reported = [], []
    for (tier, name), (device, file) in mirrors.items():
        where = f"{file}: ({tier}) {name}"
        if (tier, name) in steps:
            lanes, pool = steps[(tier, name)]
            want = LANE_SETS[tier.split()[0].lower()]
            if set(lanes) != set(want):
                violations.append(
                    f"lane tags differ   {where}  has {lanes}  expected {want}"
                )
            if device and pool != device:
                violations.append(
                    f"agent pool differs {where}  has {pool}  expected {device}"
                )
        elif device in pools_in_use and step_tiers.get(name):
            others = ", ".join(sorted(step_tiers[name]))
            violations.append(f"tier differs       {where}  vs  step at {others}")
        else:
            reported.append(f"no AMD step        {where}")

    for row in violations:
        print(f"FAIL {row}")
    for row in sorted(reported):
        print(f"     {row}")
    print(
        f"mirrors={len(mirrors)}  steps={len(steps)}  "
        f"violations={len(violations)}  reported={len(reported)}"
    )
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
