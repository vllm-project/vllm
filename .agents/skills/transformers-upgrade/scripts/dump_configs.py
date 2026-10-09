# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dump the config vLLM resolves for each checkpoint, one JSON line per repo.

Run once per vLLM tree (before and after a change), then compare the two
outputs with diff_dumps.py:

    PYTHONPATH=<vllm tree> .venv/bin/python dump_configs.py out.jsonl <repo>...

Pass --trust-remote-code to resolve checkpoints the way users who set it do.
"""

import json
import sys

from vllm.transformers_utils.config import get_config

out, *args = sys.argv[1:]
trust_remote_code = "--trust-remote-code" in args
with open(out, "w") as f:
    for repo in (arg for arg in args if arg != "--trust-remote-code"):
        try:
            config = get_config(repo, trust_remote_code=trust_remote_code)
            cls = f"{type(config).__module__}.{type(config).__name__}"
            record = {"repo": repo, "cls": cls, "cfg": config.to_dict()}
        except Exception as e:
            error = f"{type(e).__name__}: {e}".splitlines()[0]
            record = {"repo": repo, "error": error}
        f.write(json.dumps(record, default=str, sort_keys=True) + "\n")
        f.flush()
