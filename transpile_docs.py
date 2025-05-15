# SPDX-License-Identifier: Apache-2.0
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from tabulate import tabulate

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

ROOT_DIR = Path(__file__).parent
OLD_DIR = Path("docs/source")
EXAMPLES_DIR = OLD_DIR / "getting_started/examples"
NEW_DIR = Path("docs")
ADMONITIONS = {
    "note", "abstract", "info", "tip", "success", "question", "warning",
    "failure", "danger", "bug", "example", "quote"
}


@dataclass
class Block:
    indent: str
    fence: str
    type: str
    args: str
    start: int
    end: int


def find_fence_blocks(lines: list[str]) -> list[Block]:
    blocks = []
    pattern = re.compile("^([^:`]*)([:`]{3,}){(.*)} ?(.*)?$")

    for i, line in enumerate(lines):
        if match := pattern.match(line):
            indent, fence, block_type, block_args = match.groups()
            for j, line in enumerate(lines[i:]):
                if re.match(f"^.{{{len(indent)}}}{fence}$", line):
                    break
            blocks.append(Block(
                indent=indent,
                fence=fence,
                type=block_type,
                args=block_args,
                start=i,
                end=i + j,
            ))
    return blocks


def parse_fence_block(lines: list[str],
                      indent: str) -> tuple[list[str], dict[str, str]]:
    option_pattern = re.compile(f"^.{{{len(indent)}}}:(.*): (.*)$")
    option_matches = [option_pattern.match(line) for line in lines]
    content = [b for o, b in zip(option_matches, lines) if o is None]
    attrs = {m.group(1): m.group(2) for m in option_matches if m is not None}
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


def to_name(heading: str) -> str:
    return heading.strip("# \n").replace(" ", "-").lower()


total_blocks = 0
unhandled_blocks = 0


def transpile_myst_to_md(old_path: Path) -> None:
    """
    Transpile MyST markdown files to standard markdown.
    """
    new_path = NEW_DIR / old_path.relative_to(OLD_DIR)
    if old_path.stem == "index":
        new_path = new_path.with_stem("README")
    new_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Transpiling from %s to %s", old_path, new_path)

    with open(old_path) as f:
        lines = f.readlines()

    snippets = {}
    template = OLD_DIR / "getting_started/installation/device.template.md"
    if old_path.parent.parent == template.parent:
        with open(template) as f:
            headings = [line for line in f.readlines() if line.strip()]
            snippets = {
                h: {
                    "start": to_name(h),
                    "end": to_name(headings[i-1]) if i else None
                } 
                for i, h 
                in enumerate(headings)
            }

    for i, line in enumerate(lines):
        # Replace MyST links with regular markdown links
        line = replace_links(line)

        # Move page title to front matter
        if ((match := re.match(r"^# (.*)$", line))
                and "```" not in "".join(lines[:i])):
            lines[0] = f"---\ntitle: {match.group(1)}\n---\n{lines[0]}"
            if i > 0:
                line = ""

        # Delete MyST options that don't have a MkDocs equivalent
        deletes = {":selected:", ":sync-group:"}
        if any(delete in line for delete in deletes):
            line = ""

        # Replace headings with snippet names
        if line in snippets:
            snippet = snippets[line]
            line = f"# --8<-- [start:{snippet['start']}]\n"
            if snippet["end"]:
                line = f"# --8<-- [end:{snippet['end']}]\n{line}"
        if snippets and i == len(lines) - 1:
            line += f"# --8<-- [end:{snippets[headings[-1]]['start']}]\n"

        lines[i] = line

    # Get all fenced blocks and sort them so we process the inner blocks first
    blocks = find_fence_blocks(lines)
    blocks = sorted(blocks, key=lambda x: len(x.fence))

    # Process each block
    for block in blocks:
        indent = block.indent
        start = block.start
        end = block.end

        block.type = maybe_update_admonition(block.type)

        # Handle toctree
        if block.type == "toctree":
            content, attrs = parse_fence_block(lines[start + 1:end], indent)
            caption = attrs.pop("caption", "")
            content = [c_stripped for c in content if (c_stripped := c.strip())]
            content = [f"- [{c.title()}](./{c}.md)\n" for c in content]
            lines[start] = f"{caption}:\n\n" if caption else ""
            lines[start] += "".join(content)
            lines[start + 1:end] = ["" for _ in lines[start + 1:end]]
            lines[end] = ""
            continue

        # Handle code blocks
        if block.type == "code-block":
            content, attrs = parse_fence_block(lines[start + 1:end], indent)
            caption = attrs.pop("caption", "")
            title = f' title="{caption}"' if caption else ""
            if attrs:
                logger.warning("Code block attributes not handled: %s", attrs)
            lines[start] = f"{indent}```{block.args}{title}\n"
            lines[start] += "".join(content)
            lines[start + 1:end] = ["" for _ in lines[start + 1:end]]
            lines[end] = f"{indent}```\n"
            continue

        # Handle math blocks
        if block.type == "math":
            content, _ = parse_fence_block(lines[start + 1:end], indent)
            math = [c for c in content if c.strip()]
            lines[start] = f"{indent}$$\n"
            lines[start] += "".join(math)
            lines[start + 1:end] = ["" for _ in lines[start + 1:end]]
            lines[end] = f"{indent}$$\n"
            continue

        # Handle contents
        if block.type == "contents":
            lines[start] = f"{indent}[TOC]\n"
            lines[start + 1:end] = ["" for _ in lines[start + 1:end]]
            lines[end] = ""
            continue

        # Handle admonitions
        if block.type in ADMONITIONS:
            args = f" \"{block.args}\"" if block.args else ""
            lines[start] = f"!!! {block.type}{args}\n"
            lines[start + 1:end] = indent_lines(lines[start + 1:end], 4)
            lines[end] = ""
            continue

        # Handle raw HTML
        if block.type == "raw" and block.args == "html":
            lines[start] = ""
            lines[end] = ""
            continue

        # Handle images
        if block.type == "image":
            src = block.args
            _, attrs = parse_fence_block(lines[start + 1:end], indent)
            alt = attrs.pop("alt", "")
            if attrs:
                logger.warning("Image attributes not handled: %s", attrs)
            lines[start] = f"{indent}![{alt}]({src})\n"
            lines[start + 1:end] = ["" for _ in lines[start + 1:end]]
            lines[end] = ""
            continue
        if block.type == "figure":
            src = block.args
            content, attrs = parse_fence_block(lines[start + 1:end], indent)
            lines[start] = f'{indent}<figure markdown="span">\n'
            attr_list = " ".join(f'{k}="{v}"' for k, v in attrs.items())
            lines[start] += f"{indent}  ![]({src}){{ {attr_list} }}\n"
            if content:
                figcaption = f"<figcaption>{content[0]}</figcaption>"
                lines[start] += f"{indent}  {figcaption}\n"
            lines[start + 1:end] = ["" for _ in lines[start + 1:end]]
            lines[end] = f"{indent}</figure>\n"
            continue

        # Handle list table
        if block.type == "list-table":
            content, _ = parse_fence_block(lines[start + 1:end], indent)
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
                             len(indent))) + "\n"
            if block.args:
                lines[
                    start] += f"{indent}  <figcaption>{block.args}</figcaption>\n"
            lines[start + 1:end] = ["" for _ in lines[start + 1:end]]
            lines[end] = ""
            continue

        # Handle literal includes
        if block.type == "literalinclude":
            # All the literal includes we use reference library files
            path = (old_path.parent / block.args).resolve()
            _, attrs = parse_fence_block(lines[start + 1:end], indent)
            language = attrs.pop("language", "")
            name = attrs.pop("start-after", "").replace('begin-', '')
            name = f":{name}" if name else ""
            attrs.pop("end-before", None)
            if attrs:
                logger.warning("Literal include attributes not handled: %s", attrs)
            title = "" if name else f' title="{path.relative_to(ROOT_DIR)}"'
            lines[start] = f"{indent}```{language}{title}\n"
            lines[start] += f'{indent}--8<-- "{path}{name}"\n'
            lines[start + 1:end] = ["" for _ in lines[start + 1:end]]
            lines[end] = f"{indent}```\n"
            continue

        # Handle includes
        if block.type == "include":
            # All the includes we use reference documentation files
            path = (new_path.parent / block.args).resolve()
            _, attrs = parse_fence_block(lines[start + 1:end], indent)
            name = to_name(attrs.pop("start-after", "").strip('"'))
            name = f":{name}" if name else ""
            attrs.pop("end-before", None)
            if attrs:
                logger.warning("Include attributes not handled: %s", attrs)
            lines[start] = f'{indent}--8<-- "{path}{name}"\n'
            lines[start + 1:end] = ["" for _ in lines[start + 1:end]]
            lines[end] = ""
            continue

        # Handle tabs
        if block.type == "tab-item":
            content, _ = parse_fence_block(lines[start + 1:end], indent)
            lines[start] = f"=== \"{block.args}\"\n"
            lines[start] += "".join(indent_lines(content, 4))
            lines[start + 1:end] = ["" for _ in lines[start + 1:end]]
            lines[end] = ""
            continue
        if block.type == "tab-set":
            lines[start] = ""
            lines[end] = ""
            continue

        global unhandled_blocks
        unhandled_blocks += 1
        logger.warning("Unhandled block: %s", block)

    global total_blocks
    total_blocks += len(blocks)

    # Write the content to the new file
    content = "".join(lines)
    content = re.sub(r"\n{3,}", "\n\n", content)
    with open(new_path, "w") as f:
        f.write(content)

def main():
    for path in OLD_DIR.rglob("*.*"):
        if path.is_relative_to(EXAMPLES_DIR):
            continue
        elif path.suffix == ".md":
            transpile_myst_to_md(path)
        elif path.suffix in {".png", ".jpg", ".ico"}:
            new_path = NEW_DIR / path.relative_to(OLD_DIR)
            new_path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "rb") as f:
                content = f.read()
            with open(new_path, "wb") as f:
                f.write(content)
        elif path.suffix in {".py", ".css", ".js"}:
            logger.info("Skipping %s", path)
        else:
            logger.warning("Skipping %s", path)
    logger.info(
        "%d blocks in total, %d unhandled", total_blocks, unhandled_blocks
    )


def on_startup(command: Literal["build", "gh-deploy", "serve"], dirty: bool) -> None:
    main()


if __name__ == "__main__":
    main()
