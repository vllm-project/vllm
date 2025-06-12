# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import regex as re

ROOT_DIR = Path(__file__).parent.parent.parent.parent
ROOT_DIR_RELATIVE = '../../../../..'
EXAMPLE_DIR = ROOT_DIR / "examples"
EXAMPLE_DOC_DIR = ROOT_DIR / "docs/examples"
print(ROOT_DIR.resolve())
print(EXAMPLE_DIR.resolve())
print(EXAMPLE_DOC_DIR.resolve())


def fix_case(text: str) -> str:
    subs = {
        "api": "API",
        "cli": "CLI",
        "cpu": "CPU",
        "llm": "LLM",
        "mae": "MAE",
        "tpu": "TPU",
        "aqlm": "AQLM",
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
        return fix_case(self.path.stem.replace("_", " ").title())

    def generate(self) -> str:
        content = f"---\ntitle: {self.title}\n---\n\n"
        content += f"Source <gh-file:{self.path.relative_to(ROOT_DIR)}>.\n\n"

        # Use long code fence to avoid issues with
        # included files containing code fences too
        code_fence = "``````"
        is_code = self.main_file.suffix != ".md"
        if is_code:
            content += f"{code_fence}{self.main_file.suffix[1:]}\n"
        content += f'--8<-- "{self.main_file}"\n'
        if is_code:
            content += f"{code_fence}\n"
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
        print(doc_path)
        if not doc_path.parent.exists():
            doc_path.parent.mkdir(parents=True)
        with open(doc_path, "w+") as f:
            f.write(example.generate())
