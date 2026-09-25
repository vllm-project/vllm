# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Regenerate tests/watermarking/watermarking_goldens.json."""

import json
from pathlib import Path

from tests.watermarking.golden_candidates import golden_payload, validate_golden_guards

GOLDENS_PATH = Path(__file__).with_name("watermarking_goldens.json")


def main() -> None:
    payload = golden_payload()
    validate_golden_guards(payload["candidates"])
    GOLDENS_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(payload['candidates'])} candidates to {GOLDENS_PATH}")


if __name__ == "__main__":
    main()
