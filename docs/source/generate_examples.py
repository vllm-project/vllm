# SPDX-License-Identifier: Apache-2.0

import itertools
import re
from dataclasses import dataclass, field
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
ROOT_DIR_RELATIVE = '../../../..'
EXAMPLE_DIR = ROOT_DIR / "examples"
EXAMPLE_DOC_DIR = ROOT_DIR / "docs/source/getting_started/examples"


def fix_case(text: str) -> str:
    subs = {
        "api": "API",
        "Cli": "CLI",
        "cpu": "CPU",
        "llm": "LLM",
        "tpu": "TPU",
        "aqlm": "AQLM",
        "gguf": "GGUF",
        "lora": "LoRA",
        "vllm": "vLLM",
        "openai": "OpenAI",
        "multilora": "MultiLoRA",
        "mlpspeculator": "MLPSpeculator",
        r"fp\d+": lambda x: x.group(0).upper(),  # e.g. fp16, fp32
        r"int\d+": lambda x: x.group(0).upper(),  # e.g. int8, int16
    }
    for pattern, repl in subs.items():
        text = re.sub(rf'\b{pattern}\b', repl, text, flags=re.IGNORECASE)
    return text


@dataclass
class Index:
    """
    Index class to generate a structured document index.

    Attributes:
        path (Path): The path save the index file to.
        title (str): The title of the index.
        description (str): A brief description of the index.
        caption (str): An optional caption for the table of contents.
        maxdepth (int): The maximum depth of the table of contents. Defaults to 1.
        documents (list[str]): A list of document paths to include in the index. Defaults to an empty list.

    Methods:
        generate() -> str:
            Generates the index content as a string in the specified format.
    """ # noqa: E501
    path: Path
    title: str
    description: str
    caption: str
    maxdepth: int = 1
    documents: list[str] = field(default_factory=list)

    def generate(self) -> str:
        content = f"# {self.title}\n\n{self.description}\n\n"
        content += ":::{toctree}\n"
        content += f":caption: {self.caption}\n:maxdepth: {self.maxdepth}\n"
        content += "\n".join(self.documents) + "\n:::\n"
        return content


@dataclass
class Example:
    """
    Example class for generating documentation content from a given path.

    Attributes:
        path (Path): The path to the main directory or file.
        category (str): The category of the document.
        main_file (Path): The main file in the directory.
        other_files (list[Path]): List of other files in the directory.
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
        # Convert the path to a relative path from __file__
        make_relative = lambda path: ROOT_DIR_RELATIVE / path.relative_to(
            ROOT_DIR)

        content = f"Source <gh-file:{self.path.relative_to(ROOT_DIR)}>.\n\n"
        include = "include" if self.main_file.suffix == ".md" else \
            "literalinclude"
        if include == "literalinclude":
            content += f"# {self.title}\n\n"
        content += f":::{{{include}}} {make_relative(self.main_file)}\n"
        if include == "literalinclude":
            content += f":language: {self.main_file.suffix[1:]}\n"
        content += ":::\n\n"

        if not self.other_files:
            return content

        content += "## Example materials\n\n"
        for file in sorted(self.other_files):
            include = "include" if file.suffix == ".md" else "literalinclude"
            content += f":::{{admonition}} {file.relative_to(self.path)}\n"
            content += ":class: dropdown\n\n"
            content += f":::{{{include}}} {make_relative(file)}\n:::\n"
            content += ":::\n\n"

        return content


def generate_examples():
    # Create the EXAMPLE_DOC_DIR if it doesn't exist
    if not EXAMPLE_DOC_DIR.exists():
        EXAMPLE_DOC_DIR.mkdir(parents=True)

    # Create empty indices
    examples_index = Index(
        path=EXAMPLE_DOC_DIR / "examples_index.md",
        title="Examples",
        description=
        "A collection of examples demonstrating usage of vLLM.\nAll documented examples are autogenerated using <gh-file:docs/source/generate_examples.py> from examples found in <gh-file:examples>.",  # noqa: E501
        caption="Examples",
        maxdepth=2)
    # Category indices stored in reverse order because they are inserted into
    # examples_index.documents at index 0 in order
    category_indices = {
        "other":
        Index(
            path=EXAMPLE_DOC_DIR / "examples_other_index.md",
            title="Other",
            description=
            "Other examples that don't strongly fit into the online or offline serving categories.",  # noqa: E501
            caption="Examples",
        ),
        "online_serving":
        Index(
            path=EXAMPLE_DOC_DIR / "examples_online_serving_index.md",
            title="Online Serving",
            description=
            "Online serving examples demonstrate how to use vLLM in an online setting, where the model is queried for predictions in real-time.",  # noqa: E501
            caption="Examples",
        ),
        "offline_inference":
        Index(
            path=EXAMPLE_DOC_DIR / "examples_offline_inference_index.md",
            title="Offline Inference",
            description=
            "Offline inference examples demonstrate how to use vLLM in an offline setting, where the model is queried for predictions in batches. We recommend starting with <project:basic.md>.",  # noqa: E501
            caption="Examples",
        ),
    }

    examples = []
    glob_patterns = ["*.py", "*.md", "*.sh"]
    # Find categorised examples
    for category in category_indices:
        category_dir = EXAMPLE_DIR / category
        globs = [category_dir.glob(pattern) for pattern in glob_patterns]
        for path in itertools.chain(*globs):
            examples.append(Example(path, category))
        # Find examples in subdirectories
        for path in category_dir.glob("*/*.md"):
            examples.append(Example(path.parent, category))
    # Find uncategorised examples
    globs = [EXAMPLE_DIR.glob(pattern) for pattern in glob_patterns]
    for path in itertools.chain(*globs):
        examples.append(Example(path))
    # Find examples in subdirectories
    for path in EXAMPLE_DIR.glob("*/*.md"):
        # Skip categorised examples
        if path.parent.name in category_indices:
            continue
        examples.append(Example(path.parent))

    # Generate the example documentation
    for example in sorted(examples, key=lambda e: e.path.stem):
        doc_path = EXAMPLE_DOC_DIR / f"{example.path.stem}.md"
        with open(doc_path, "w+") as f:
            f.write(example.generate())
        # Add the example to the appropriate index
        index = category_indices.get(example.category, examples_index)
        index.documents.append(example.path.stem)

    # Generate the index files
    for category_index in category_indices.values():
        if category_index.documents:
            examples_index.documents.insert(0, category_index.path.name)
            with open(category_index.path, "w+") as f:
                f.write(category_index.generate())

    with open(examples_index.path, "w+") as f:
        f.write(examples_index.generate())
