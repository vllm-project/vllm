import re
from pathlib import Path


def fix_case(text: str) -> str:
    subs = [
        ("api", "API"),
        ("llm", "LLM"),
        ("vllm", "vLLM"),
        ("openai", "OpenAI"),
        ("multilora", "MultiLoRA"),
    ]
    for sub in subs:
        text = re.sub(*sub, text, flags=re.IGNORECASE)
    return text


def generate_title(filename: str) -> str:
    # Turn filename into a title
    title = filename.replace("_", " ").title()
    # Handle acronyms and names
    title = fix_case(title)
    return f"# {title}"


def generate_examples():
    root_dir = Path(__file__).parent.parent.parent.resolve()

    # Source paths
    script_dir = root_dir / "examples"
    script_paths = sorted(script_dir.glob("*.py"))

    # Destination paths
    doc_dir = root_dir / "docs/source/getting_started/examples"
    doc_paths = [doc_dir / f"{path.stem}.md" for path in script_paths]

    # Generate the example docs for each example script
    for script_path, doc_path in zip(script_paths, doc_paths):
        # Make script_path relative to doc_path and call it include_path
        include_path = '../../../..' / script_path.relative_to(root_dir)
        content = (f"{generate_title(doc_path.stem)}\n\n"
                   f"Source: <gh-file:examples/{script_path.name}>.\n\n"
                   f"```{{literalinclude}} {include_path}\n"
                   ":language: python\n"
                   ":linenos:\n```")
        with open(doc_path, "w+") as f:
            f.write(content)

    # Generate the toctree for the example scripts
    with open(doc_dir / "examples_index.template.md") as f:
        examples_index = f.read()
    with open(doc_dir / "examples_index.md", "w+") as f:
        example_docs = "\n".join(path.stem + ".md" for path in script_paths)
        f.write(examples_index.replace(r"%EXAMPLE_DOCS%", example_docs))
