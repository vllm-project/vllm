# SPDX-License-Identifier: Apache-2.0
import logging
import re
from pathlib import Path

from tabulate import tabulate

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

ROOT_DIR = Path(__file__).parent
OLD_DIR = Path("docs/source")
NEW_DIR = Path("docs")
ADMONITIONS = {
    "note", "abstract", "info", "tip", "success", "question", "warning",
    "failure", "danger", "bug", "example", "quote"
}


def find_fence_blocks(lines: list[str]) -> list[dict]:
    blocks = []
    pattern = re.compile("^([^:`]*)([:`]{3,}){(.*)} ?(.*)?$")

    for i, line in enumerate(lines):
        if match := pattern.match(line):
            indent, fence, block_type, block_args = match.groups()
            for j, l in enumerate(lines[i:]):
                if re.match(f"^.{{{len(indent)}}}{fence}$", l):
                    break
            blocks.append({
                "indent": indent,
                "fence": fence,
                "type": block_type,
                "args": block_args,
                "start": i,
                "end": i + j,
                "handled": False,
            })
    return blocks


def parse_fence_block(lines: list[str],
                      indent: str) -> tuple[list[str], list[str]]:
    option_pattern = re.compile(f"^.{{{len(indent)}}}:(.*): (.*)$")
    option_matches = [option_pattern.match(l) for l in lines]
    content = [b for o, b in zip(option_matches, lines) if o is None]
    attrs = [
        f"{m.group(1)}=\"{m.group(2)}\"" for m in option_matches
        if m is not None
    ]
    return content, attrs


def replace_links(line: str) -> str:
    # Replace MyST anchors with MkDocs anchors
    line = re.sub(r"^\((.*)\)=$", r"[](){ #\1 }", line)
    # Fix references to these anchors
    line = re.sub(r"\[(.*?)\]\(#(.*?)\)", r"[\1](\2)", line)
    line = re.sub(r"<project:#(.*?)>", r"[\1](\1)", line)
    return line


def maybe_update_admonition(admonition: str) -> str:
    """Replace MyST admonitions with MkDocs admonitions."""
    admonition_mapping = {
        "attention": "warning",
        "caution": "warning",
        "error": "failure",
        "hint": "tip",
        "important": "warning",
        "seealso": "info",
    }
    return admonition_mapping.get(admonition, admonition)


def indent_lines(lines: list[str], indent: int) -> list[str]:
    return [(" " * indent) + line for line in lines]


handled_blocks = 0
unhandled_blocks = 0


def transpile_myst_to_md(old_path: Path) -> None:
    """
    Transpile MyST markdown files to standard markdown.
    """
    new_path = NEW_DIR / old_path.relative_to(OLD_DIR)
    new_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Transpiling to {new_path}")

    with open(old_path) as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        line = replace_links(line)

        # Move page title to front matter
        if ((match := re.match(r"^# (.*)$", line))
                and "```" not in "".join(lines[:i])):
            lines[0] = f"---\ntitle: {match.group(1)}\n---\n{lines[0]}"
            line = ""

        # Delete MyST options that don't have a MkDocs equivalent
        deletes = {":selected:", ":sync-group:"}
        if any(delete in line for delete in deletes):
            line = ""

        lines[i] = line

    # Get all fenced blocks and sort them so we process the inner blocks first
    blocks = find_fence_blocks(lines)
    blocks = sorted(blocks, key=lambda x: len(x["fence"]))

    # Process each block
    for block in blocks:
        indent = block["indent"]
        start = block["start"]
        end = block["end"]

        # Handle toctree
        if block["type"] == "toctree":
            lines[start] = "###### Contents\n"
            lines[start + 1:end] = ["" for _ in lines[start + 1:end]]
            lines[end] = ""
            block["handled"] = True

        block["type"] = maybe_update_admonition(block["type"])

        # Handle admonitions
        if block["type"] in ADMONITIONS:
            args = f" \"{block['args']}\"" if block["args"] else ""
            lines[start] = f"!!! {block['type']}{args}\n"
            lines[start + 1:end] = indent_lines(lines[start + 1:end], 4)
            lines[end] = ""
            block["handled"] = True

        # Handle raw HTML
        if block["type"] == "raw" and block["args"] == "html":
            lines[start] = ""
            lines[end] = ""
            block["handled"] = True

        # Handle images
        if block["type"] == "image":
            src = block["args"]
            content, attrs = parse_fence_block(lines[start + 1:end], indent)
            if attrs:
                logger.warning("Image attributes not handled: %s", attrs)
            lines[start] = f"{indent}![]({src})\n"
            lines[start + 1:end] = ["" for _ in lines[start + 1:end]]
            lines[end] = ""
        if block["type"] == "figure":
            src = block["args"]
            caption, attrs = parse_fence_block(lines[start + 1:end], indent)
            lines[start] = f'{indent}<figure markdown="span">\n'
            lines[start] += f"{indent}  ![]({src}){{ {' '.join(attrs)} }}\n"
            if caption:
                lines[
                    start] += f"{indent}  <figcaption>{caption[0]}</figcaption>\n"
            lines[start + 1:end] = ["" for _ in lines[start + 1:end]]
            lines[end] = f"{indent}</figure>\n"
            block["handled"] = True

        # Handle list table
        if block["type"] == "list-table":
            content, attrs = parse_fence_block(lines[start + 1:end], indent)
            row = []
            rows = []
            for c in content:
                if not c.strip():
                    continue
                if match := re.match(r"^[-*] [-*] (.*)$", c):
                    if row:
                        rows.append(row)
                    row = [match.group(1)]
                elif match := re.match(r"^  [-*] (.*)$", c):
                    row.append(match.group(1))
            table = tabulate(rows[1:], rows[0], tablefmt="github")
            lines[start] = "\n".join(
                indent_lines(table.splitlines(),
                             len(indent) + 2)) + "\n"
            if block["args"]:
                lines[
                    start] += f"{indent}  <figcaption>{block['args']}</figcaption>\n"
            lines[start + 1:end] = ["" for _ in lines[start + 1:end]]
            lines[end] = ""
            block["handled"] = True

        # Handle literal includes
        if block["type"] == "literalinclude":
            path = (old_path.parent / block["args"]).resolve()
            lines[
                start] = f'{indent}``` title="{path.relative_to(ROOT_DIR)}"\n'
            lines[start] += f'{indent}--8<-- "{path}"\n'
            # lines[start + 1:end] = ["" for _ in lines[start + 1:end]]
            lines[end] = f"{indent}```\n"
            block["handled"] = True
            logger.warning("Literal include only partially handled")

        # Handle includes
        if block["type"] == "include":
            content, attrs = parse_fence_block(lines[start + 1:end], indent)
            if not attrs:
                lines[start] = f'--8<-- "{block["args"]}"\n'
                lines[end] = ""
                block["handled"] = True

        # Handle tabs
        if block["type"] == "tab-item":
            content, attrs = parse_fence_block(lines[start + 1:end], indent)
            lines[start] = f"=== \"{block['args']}\"\n"
            lines[start] += "".join(indent_lines(content, 4))
            lines[start + 1:end] = ["" for _ in lines[start + 1:end]]
            lines[end] = ""
            block["handled"] = True
        if block["type"] == "tab-set":
            lines[start] = ""
            lines[end] = ""
            block["handled"] = True

        if not block["handled"]:
            logger.warning("Unhandled block: %s", block)

    global handled_blocks, unhandled_blocks
    handled_blocks += len([b for b in blocks if b["handled"]])
    unhandled_blocks += len([b for b in blocks if not b["handled"]])

    # Write the content to the new file
    content = "".join(lines)
    content = re.sub(r"\n{3,}", "\n\n", content)
    with open(new_path, "w") as f:
        f.write(content)


if __name__ == "__main__":
    for path in OLD_DIR.rglob("*"):
        if path.suffix == ".md":
            transpile_myst_to_md(path)
        elif path.suffix in {".png", ".jpg", ".ico"}:
            new_path = NEW_DIR / path.relative_to(OLD_DIR)
            new_path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "rb") as f:
                content = f.read()
            with open(new_path, "wb") as f:
                f.write(content)
        else:
            logger.warning("Skipping %s", path)
    logger.info(
        f"Handled {handled_blocks} blocks, unhandled {unhandled_blocks} blocks"
    )
