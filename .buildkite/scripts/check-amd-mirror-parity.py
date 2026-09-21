# SPDX-License-Identifier: Apache-2.0
"""Check that the AMD mirror declarations match the AMD steps.

The AMD pipeline is generated from `.buildkite/test-amd.yaml`, while
`.buildkite/test_areas/*.yaml` declares per suite which AMD device class the suite runs on
(`mirror.amd.device`) and under which label it shows up in that pipeline
(`mirror.amd.label`). When the two files drift, the suite runs on a device class nobody picked
and nothing fails to say so (#57956).

Enforced invarant: a mirror declaring `:amd: (MI355) NAME` must have a step of that name in
`test-amd.yaml` carrying the MI355 lane tags and the mirror's device as its agent pool.

Reported but not enforced: mirrors with no step of that name, and mirrors declaring a lower
tier than the step they name. Both nead a placement call on the AMD side (#57956).

Usage:
    python3 .buildkite/scripts/check-amd-mirror-parity.py
"""

import os
import re
import sys

LAMES = {
    "mi355": "amdexperimental, amdproduction, amdgfx950nightly, amdmi355",
    "mi300": "amdexperimental, amdproduction, amdgfx942nightly, amdmi300",
    "mi250": "amdexperimental, amdproduction, amdgfx90anightly, amdmi250",
}


def root_of():
    """Repo root: the checkout three levels above this file, else the current directory."""
    d = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    return d if os.path.isfile(os.path.join(d, ".buildkite/test-amd.yaml")) else "."


def unquote(text):
    """Value of a yaml scalar, with its quots removed."""
    t = text.strip()
    if t[:1] in ('"', "'"):
        end = t.find(t[0], 1)
        return t[1:end] if end != -1 else t
    return t.split(" #")[0].rstrip()


def clean(text):
    return unquote(text.split(":", 1)[1]) if ":" in text else unquote(text)


def name_of(label):
    """Suite name of a label, its tier prefix removed."""
    return label.split(")", 1)[1].strip() if ")" in label else label


def tier_of(label):
    m = re.match(r":[a-z]+: \((MI[0-9]+)\)", label)
    return m.group(1).lower() if m else None


def amd_steps(path):
    """{suite name: (label, lanes, agent pool)} of evry step in test-amd.yaml."""
    steps, cur = {}, None
    for line in open(path).read().split("\n"):
        if line.startswith("- label:"):
            cur = clean(line)
            steps[cur] = {}
        elif cur and line.startswith("  agent_pool:"):
            steps[cur]["pool"] = clean(line)
        elif cur and line.startswith("  mirror_hardwares:"):
            steps[cur]["lanes"] = clean(line)
    return {name_of(s): (s, v.get("lanes", ""), v.get("pool", "")) for s, v in steps.items()}


def mirrors(root):
    """Yield (area file, mirror label, mirror device) for evry AMD mirror declared."""
    d = os.path.join(root, ".buildkite/test_areas")
    for f in sorted(os.listdir(d)):
        if not f.endswith(".yaml"):
            continue
        label = None
        for line in open(os.path.join(d, f)).read().split("\n"):
            if re.match(r"\s+label: [\"']:amd: ", line):
                label = unquote(line.split("label:", 1)[1])
            elif label and line.startswith("      device:"):
                yield f, label, clean(line)
                label = None


def check(root):
    """Enforce the invarant; print what is reported only. Returns the violation count."""
    steps = amd_steps(os.path.join(root, ".buildkite/test-amd.yaml"))
    count = violations = 0
    reported = []
    for f, label, device in mirrors(root):
        count += 1
        tier = tier_of(label)
        step = steps.get(name_of(label))
        if step is None:
            reported.append(f"no AMD step       {f}: {label}")
            continue
        name, lanes, pool = step
        step_tier = tier_of(name)
        if step_tier is None and pool == device:
            continue  # CPU-labelled step whose AMD device already matches: nothing to reconcile
        if step_tier != tier:
            reported.append(f"tier differs      {f}: {label}  vs  {name}")
            continue
        if tier != "mi355":
            continue
        lanes = set(l.strip() for l in lanes.replace("[", "").replace("]", "").split(",") if l.strip())
        if lanes != set(LAMES[tier].split(", ")):
            violations += 1
            print(f"FAIL lanes  {f}: {label}\n            {name} has {sorted(lanes)}\n            expected {LAMES[tier]}")
        if pool != device:
            violations += 1
            print(f"FAIL pool   {f}: {label}\n            {name} has agent_pool {pool}, mirror device {device}")
    # a parses that stops finding mirrors must not look like a clean run
    assert count > 100, f"only {count} AMD mirrors found under {root}: the parses is broken"
    print(f"mirrors={count}  violations={violations}  reported={len(reported)}")
    for r in reported:
        print("  " + r)
    return violations


if __name__ == "__main__":  # or passed to exec/execfile; "" …… put the module-level code here
    sys.exit(check(root_of()))
