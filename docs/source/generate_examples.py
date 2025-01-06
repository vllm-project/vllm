import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
ROOT_DIR_RELATIVE = '../../../..'
EXAMPLE_DIR = ROOT_DIR / "examples"
EXAMPLE_DOC_DIR = ROOT_DIR / "docs/source/getting_started/examples"

CATEGORIES = {
    "Online Inference": {
        "keywords": ["openai"],
        "index_file_name":
        "examples_online_inference_index.md",
        "description":
        "Online inference examples demonstrate how to use vLLM in an online setting, where the model is queried for predictions in real-time.",  # noqa: E501
    },
    "Offline Inference": {
        "keywords": ["offline"],
        "index_file_name":
        "examples_offline_inference_index.md",
        "description":
        "Offline inference examples demonstrate how to use vLLM in an offline setting, where the model is queried for predictions in batches.",  # noqa: E501
    },
    "Other": {
        "keywords": [],
        "index_file_name":
        "examples_other_index.md",
        "description":
        "Other examples that don't strongly fit into the online or offline inference categories.",  # noqa: E501
    },
}


def is_example(path: Path) -> bool:
    return path.is_dir() or path.suffix in (".py", ".md")


def fix_case(text: str) -> str:
    subs = {
        "api": "API",
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
        title (str): The title of the index.
        description (str): A brief description of the index. Defaults to an empty string.
        maxdepth (int): The maximum depth of the table of contents. Defaults to 1.
        caption (str): An optional caption for the table of contents. Defaults to an empty string.
        documents (List[str]): A list of document paths to include in the index. Defaults to an empty list.

    Methods:
        generate() -> str:
            Generates the index content as a string in the specified format.
    """
    title: str
    description: str = field(default="")
    maxdepth: int = 1
    caption: str = field(default="")
    documents: List[str] = field(default_factory=list)

    def generate(self) -> str:
        content = f"# {self.title}\n\n"
        content += f"{self.description}\n\n"
        content += f"```{{toctree}}\n:maxdepth: {self.maxdepth}\n"
        if self.caption:
            content += f":caption: {self.caption}\n"
        content += "\n".join(self.documents) + "\n```\n"
        return content


@dataclass
class Example:
    """
    Example class for generating documentation content from a given path.

    Attributes:
        path (Path): The path to the main directory or file.
        main_file (Path): The main file in the directory.
        other_files (List[Path]): List of other files in the directory.
        title (str): The title of the document.
        url (str): The URL to the document on GitHub.
        category (str): The category of the document.

    Methods:
        __post_init__(): Initializes the main_file, other_files, title, url, and category attributes.
        determine_main_file() -> Path: Determines the main file in the given path.
        determine_other_files() -> List[Path]: Determines other files in the directory excluding the main file.
        determine_title() -> str: Determines the title of the document.
        determine_url() -> str: Determines the URL to the document on GitHub.
        determine_category() -> str: Determines the category of the document based on its title and content.
        generate() -> str: Generates the documentation content.
    """
    path: Path
    main_file: Path = field(init=False)
    other_files: List[Path] = field(init=False)
    title: str = field(init=False)
    url: str = field(init=False)
    category: str = field(init=False)

    def __post_init__(self):
        self.main_file = self.determine_main_file()
        self.other_files = self.determine_other_files()
        self.title = self.determine_title()
        self.url = self.determine_url()
        self.category = self.determine_category()

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
        """
        return self.path if self.path.is_file() else list(
            self.path.glob("*.md")).pop()

    def determine_other_files(self) -> List[Path]:
        """
        Determine other files in the directory excluding the main file.

        This method checks if the given path is a file. If it is, it returns an empty list.
        Otherwise, it recursively searches through the directory and returns a list of all
        files that are not the main file.

        Returns:
            List[Path]: A list of Path objects representing the other files in the directory.
        """
        if self.path.is_file():
            return []
        is_other_file = lambda file: file.is_file() and file != self.main_file
        return [file for file in self.path.rglob("*") if is_other_file(file)]

    def determine_title(self) -> str:
        return fix_case(self.path.stem.replace("_", " ").title())

    def determine_url(self) -> str:
        return f"https://github.com/vllm-project/vllm/blob/main/examples/{self.path.relative_to(EXAMPLE_DIR)}"

    def determine_category(self) -> str:
        """
        Determines the category of the document based on its title and content.

        The method reads the content of the main file associated with the document,
        converts both the title and content to lowercase, and checks for the presence
        of any keywords defined in the CATEGORIES dictionary. If a keyword is found
        in either the title or the content, the corresponding category is returned.

        Returns:
            str: The category of the document.
        """
        title = self.title.lower()
        with open(self.main_file, "r") as f:
            content = f.read().lower()
        final_category = "Other"
        for category, category_info in CATEGORIES.items():
            if any(indicator in content or indicator in title
                   for indicator in category_info["keywords"]):
                final_category = category
        return final_category

    def generate(self) -> str:
        # Convert the path to a relative path from __file__
        make_relative = lambda path: ROOT_DIR_RELATIVE / path.relative_to(
            ROOT_DIR)

        content = f"Source <{self.url}>.\n\n"
        if self.main_file.suffix == ".py":
            content += f"# {self.title}\n\n"
        include = "include" if self.main_file.suffix == ".md" else "literalinclude"
        content += f":::{{{include}}} {make_relative(self.main_file)}\n:::\n\n"
        if self.other_files:
            content += "## Example source\n\n"
            for file in self.other_files:
                include = "include" if file.suffix == ".md" else "literalinclude"
                content += f":::{{admonition}} {file.relative_to(self.path)}\n:class: dropdown\n\n"
                content += f":::{{{include}}} {make_relative(file)}\n:::\n"
                content += ":::\n\n"

        return content


def generate_examples():
    # Create the EXAMPLE_DOC_DIR if it doesn't exist
    if not EXAMPLE_DOC_DIR.exists():
        EXAMPLE_DOC_DIR.mkdir(parents=True)

    # Generate examples_index and examples_{category}_index files
    examples_index = Index(
        title="Examples",
        description=
        "A collection of examples demonstrating usage of vLLM.\n\nAll documented examples are autogenerated from examples found in [vllm/examples](https://github.com/vllm-project/vllm/tree/main/examples).",
        maxdepth=2,
        caption="Categories")
    category_indices = {
        category: Index(title=category,
                        description=category_info["description"])
        for category, category_info in CATEGORIES.items()
    }

    examples = [
        Example(path) for path in sorted(EXAMPLE_DIR.iterdir())
        if is_example(path)
    ]
    for example in examples:
        category_indices[example.category].documents.append(example.path.stem)
        doc_path = EXAMPLE_DOC_DIR / f"{example.path.stem}.md"
        with open(doc_path, "w+") as f:
            f.write(example.generate())

    # Generate the toctree for the example scripts
    for category, category_index in category_indices.items():
        category_index_name = CATEGORIES[category]["index_file_name"]
        category_index_path = EXAMPLE_DOC_DIR / category_index_name
        examples_index.documents.append(category_index_path.stem)
        with open(category_index_path, "w+") as f:
            f.write(category_index.generate())

    with open(EXAMPLE_DOC_DIR / "examples_index.md", "w+") as f:
        f.write(examples_index.generate())
