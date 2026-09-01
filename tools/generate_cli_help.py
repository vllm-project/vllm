# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
OUTPUT = ROOT / "vllm_cli" / "_help.json"
ENV = {
    "COLUMNS": "80",
    "LINES": "24",
    "LC_ALL": "C",
    "NO_COLOR": "1",
    "PYTHONHASHSEED": "0",
    "TERM": "dumb",
    "VLLM_PLUGINS": "",
    "VLLM_LOGGING_LEVEL": "CRITICAL",
}
INSPECT = """
import argparse
import io
import json
import sys
import vllm.utils.argparse_utils as argparse_utils
import vllm.version
vllm.version.__version__ = "dev"
from vllm.entrypoints.cli.main import main
original = argparse_utils.FlexibleArgumentParser
search = original._search_keyword
class RecordingParser(original):
    parser = None
    def parse_args(self, *args, **kwargs):
        RecordingParser.parser = self
        raise RuntimeError("recorded")
argparse_utils.FlexibleArgumentParser = RecordingParser
sys.argv = ["vllm", "serve", "--help=all"]
try:
    main()
except RuntimeError as error:
    if str(error) != "recorded":
        raise
finally:
    argparse_utils.FlexibleArgumentParser = original
    original._search_keyword = search
subparsers = next(action for action in RecordingParser.parser._actions
                  if isinstance(action, argparse._SubParsersAction))
serve = subparsers.choices["serve"]
groups = {group.title.lower().replace("_", "-").lstrip("-")
          for group in serve._action_groups if group.title and group._group_actions}
options = {}
for group in serve._action_groups:
    for action in group._group_actions:
        aliases = tuple(action.option_strings)
        for alias in aliases:
            key = alias.lower().replace("_", "-").lstrip("-")
            if key in options and options[key] != aliases:
                raise ValueError(f"option lookup collision: {key}")
            options[key] = aliases
if groups & set(options):
    raise ValueError(f"group/option lookup collision: {sorted(groups & set(options))}")
def render(args):
    search, epilog, stdout = original._search_keyword, serve.epilog, sys.stdout
    output = io.StringIO()
    sys.stdout = output
    try:
        original.parse_args(RecordingParser.parser, args)
    except SystemExit as error:
        assert error.code == 0
    finally:
        sys.stdout = stdout
        original._search_keyword = search
        serve.epilog = epilog
    return output.getvalue()
keys = sorted(groups | set(options))
print(json.dumps({"help": {"top": render(["--help"]),
                           "serve": render(["serve", "--help"]),
                           "all": render(["serve", "--help=all"])},
                  "queries": {key: render(["serve", f"--help={key}"])
                              for key in keys}}))
"""


def _run(code: str) -> str:
    environment = {k: v for k, v in os.environ.items() if not k.startswith("VLLM_")}
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(ROOT), environment.get("PYTHONPATH", ""))
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        env=environment | ENV,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(result.stderr)
    return "\n".join(line.rstrip() for line in result.stdout.splitlines()) + "\n"


def main() -> None:
    if sys.argv[1:] not in ([], ["--check"]):
        raise SystemExit("usage: generate_cli_help.py [--check]")
    content = json.dumps(json.loads(_run(INSPECT)), indent=2, sort_keys=True) + "\n"
    if sys.argv[1:] == ["--check"]:
        if not OUTPUT.is_file() or OUTPUT.read_text() != content:
            raise SystemExit(
                "CLI help snapshot is stale; run tools/generate_cli_help.py"
            )
    else:
        OUTPUT.write_text(content)


if __name__ == "__main__":
    main()
