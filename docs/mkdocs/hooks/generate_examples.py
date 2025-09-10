# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import regex as re

logger = logging.getLogger("mkdocs")

ROOT_DIR = Path(__file__).parent.parent.parent.parent
ROOT_DIR_RELATIVE = '../../../../..'
EXAMPLE_DIR = ROOT_DIR / "examples"
EXAMPLE_DOC_DIR = ROOT_DIR / "docs/examples"


def fix_case(text: str) -> str:
    subs = {
        "api": "API",
        "cli": "CLI",
        "cpu": "CPU",
        "llm": "LLM",
        "mae": "MAE",
        "tpu": "TPU",
        "gguf": "GGUF",
        "lora": "LoRA",
        "rlhf": "RLHF",
        "vllm": "vLLM",
        "openai": "OpenAI",
        "lmcache": "LMCache",
        "multilora": "MultiLoRA",
        "mlpspeculator": "MLPSpeculator",
        r"fp\d+": lambda x: x.group(0).upper(),  # e.g. fp16, fp32
        r"int\d+": lambda x: x.group(0).upper(),  # e.g. int8, int16
    }
    for pattern, repl in subs.items():
        text = re.sub(rf'\b{pattern}\b', repl, text, flags=re.IGNORECASE)
    return text


@dataclass
class Example:
    """
    Example class for generating documentation content from a given path.

    Attributes:
        path (Path): The path to the main directory or file.
        category (str): The category of the document.
        main_file (Path): The main file in the directory.
        other_files (list[Path]): list of other files in the directory.
        title (str): The title of the document.

    Methods:
        __post_init__(): Initializes the main_file, other_files, and title attributes.
        determine_main_file() -> Path: Determines the main file in the given path.
        determine_other_files() -> list[Path]: Determines other files in the directory excluding the main file.
        determine_title() -> str: Determines the title of the document.
        generate() -> str: Generates the documentation content.
    """ # noqa: E501
    path: Path
    category: str = None
    main_file: Path = field(init=False)
    other_files: list[Path] = field(init=False)
    title: str = field(init=False)

    def __post_init__(self):
        self.main_file = self.determine_main_file()
        self.other_files = self.determine_other_files()
        self.title = self.determine_title()

    @property
    def is_code(self) -> bool:
        return self.main_file.suffix != ".md"

    def determine_main_file(self) -> Path:
        """
        Determines the main file in the given path.
        If the path is a file, it returns the path itself. Otherwise, it searches
        for Markdown files (*.md) in the directory and returns the first one found.
        Returns:
            Path: The main file path, either the original path if it's a file or the first
            Markdown file found in the directory.
        Raises:
            IndexError: If no Markdown files are found in the directory.
        """ # noqa: E501
        return self.path if self.path.is_file() else list(
            self.path.glob("*.md")).pop()

    def determine_other_files(self) -> list[Path]:
        """
        Determine other files in the directory excluding the main file.

        This method checks if the given path is a file. If it is, it returns an empty list.
        Otherwise, it recursively searches through the directory and returns a list of all
        files that are not the main file.

        Returns:
            list[Path]: A list of Path objects representing the other files in the directory.
        """ # noqa: E501
        if self.path.is_file():
            return []
        is_other_file = lambda file: file.is_file() and file != self.main_file
        return [file for file in self.path.rglob("*") if is_other_file(file)]

    def determine_title(self) -> str:
        if not self.is_code:
            # Specify encoding for building on Windows
            with open(self.main_file, encoding="utf-8") as f:
                first_line = f.readline().strip()
            match = re.match(r'^#\s+(?P<title>.+)$', first_line)
            if match:
                return match.group('title')
        return fix_case(self.path.stem.replace("_", " ").title())

    def fix_relative_links(self, content: str) -> str:
        """
        Fix relative links in markdown content by converting them to gh-file
        format.
        
        Args:
            content (str): The markdown content to process
            
        Returns:
            str: Content with relative links converted to gh-file format
        """
        # Regex to match markdown links [text](relative_path)
        # This matches links that don't start with http, https, ftp, or #
        link_pattern = r'\[([^\]]*)\]\((?!(?:https?|ftp)://|#)([^)]+)\)'

        def replace_link(match):
            link_text = match.group(1)
            relative_path = match.group(2)

            # Make relative to repo root
            gh_file = (self.main_file.parent / relative_path).resolve()
            gh_file = gh_file.relative_to(ROOT_DIR)

            return f'[{link_text}](gh-file:{gh_file})'

        return re.sub(link_pattern, replace_link, content)

    def generate(self) -> str:
        content = f"# {self.title}\n\n"
        content += f"Source <gh-file:{self.path.relative_to(ROOT_DIR)}>.\n\n"

        # Use long code fence to avoid issues with
        # included files containing code fences too
        code_fence = "``````"

        if self.is_code:
            content += (f"{code_fence}{self.main_file.suffix[1:]}\n"
                        f'--8<-- "{self.main_file}"\n'
                        f"{code_fence}\n")
        else:
            with open(self.main_file) as f:
                # Skip the title from md snippets as it's been included above
                main_content = f.readlines()[1:]
            content += self.fix_relative_links("".join(main_content))
        content += "\n"

        if not self.other_files:
            return content

        content += "## Example materials\n\n"
        for file in sorted(self.other_files):
            content += f'??? abstract "{file.relative_to(self.path)}"\n'
            if file.suffix != ".md":
                content += f"    {code_fence}{file.suffix[1:]}\n"
            content += f'    --8<-- "{file}"\n'
            if file.suffix != ".md":
                content += f"    {code_fence}\n"

        return content


def on_startup(command: Literal["build", "gh-deploy", "serve"], dirty: bool):
    logger.info("Generating example documentation")
    logger.debug("Root directory: %s", ROOT_DIR.resolve())
    logger.debug("Example directory: %s", EXAMPLE_DIR.resolve())
    logger.debug("Example document directory: %s", EXAMPLE_DOC_DIR.resolve())

    # Create the EXAMPLE_DOC_DIR if it doesn't exist
    if not EXAMPLE_DOC_DIR.exists():
        EXAMPLE_DOC_DIR.mkdir(parents=True)

    categories = sorted(p for p in EXAMPLE_DIR.iterdir() if p.is_dir())

    examples = []
    glob_patterns = ["*.py", "*.md", "*.sh"]
    # Find categorised examples
    for category in categories:
        globs = [category.glob(pattern) for pattern in glob_patterns]
        for path in itertools.chain(*globs):
            examples.append(Example(path, category.stem))
        # Find examples in subdirectories
        for path in category.glob("*/*.md"):
            examples.append(Example(path.parent, category.stem))

    # Generate the example documentation
    for example in sorted(examples, key=lambda e: e.path.stem):
        example_name = f"{example.path.stem}.md"
        doc_path = EXAMPLE_DOC_DIR / example.category / example_name
        if not doc_path.parent.exists():
            doc_path.parent.mkdir(parents=True)
        # Specify encoding for building on Windows
        with open(doc_path, "w+", encoding="utf-8") as f:
            f.write(example.generate())
        logger.debug("Example generated: %s", doc_path.relative_to(ROOT_DIR))
