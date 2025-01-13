# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import inspect
import logging
import os
import sys
from typing import List

import requests
from sphinx.ext import autodoc

logger = logging.getLogger(__name__)
sys.path.append(os.path.abspath("../.."))

# -- Project information -----------------------------------------------------

project = 'vLLM'
copyright = '2024, vLLM Team'
author = 'the vLLM Team'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.linkcode",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "myst_parser",
    "sphinxarg.ext",
    "sphinx_design",
    "sphinx_togglebutton",
]
myst_enable_extensions = [
    "colon_fence",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns: List[str] = ["**/*.template.md", "**/*.inc.md"]

# Exclude the prompt "$" when copying code
copybutton_prompt_text = r"\$ "
copybutton_prompt_is_regexp = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_title = project
html_theme = 'sphinx_book_theme'
html_logo = 'assets/logos/vllm-logo-text-light.png'
html_theme_options = {
    'path_to_docs': 'docs/source',
    'repository_url': 'https://github.com/vllm-project/vllm',
    'use_repository_button': True,
    'use_edit_page_button': True,
}
html_static_path = ["_static"]
html_js_files = ["custom.js"]

myst_url_schemes = {
    'http': None,
    'https': None,
    'mailto': None,
    'ftp': None,
    "gh-issue": {
        "url":
        "https://github.com/vllm-project/vllm/issues/{{path}}#{{fragment}}",
        "title": "Issue #{{path}}",
        "classes": ["github"],
    },
    "gh-pr": {
        "url":
        "https://github.com/vllm-project/vllm/pull/{{path}}#{{fragment}}",
        "title": "Pull Request #{{path}}",
        "classes": ["github"],
    },
    "gh-dir": {
        "url": "https://github.com/vllm-project/vllm/tree/main/{{path}}",
        "title": "{{path}}",
        "classes": ["github"],
    },
    "gh-file": {
        "url": "https://github.com/vllm-project/vllm/blob/main/{{path}}",
        "title": "{{path}}",
        "classes": ["github"],
    },
}

# see https://docs.readthedocs.io/en/stable/reference/environment-variables.html # noqa
READTHEDOCS_VERSION_TYPE = os.environ.get('READTHEDOCS_VERSION_TYPE')
if READTHEDOCS_VERSION_TYPE == "tag":
    # remove the warning banner if the version is a tagged release
    header_file = os.path.join(os.path.dirname(__file__),
                               "_templates/sections/header.html")
    # The file might be removed already if the build is triggered multiple times
    # (readthedocs build both HTML and PDF versions separately)
    if os.path.exists(header_file):
        os.remove(header_file)

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']


# Generate additional rst documentation here.
def setup(app):
    from docs.source.generate_examples import generate_examples
    generate_examples()


_cached_base: str = ""
_cached_branch: str = ""


def get_repo_base_and_branch(pr_number):
    global _cached_base, _cached_branch
    if _cached_base and _cached_branch:
        return _cached_base, _cached_branch

    url = f"https://api.github.com/repos/vllm-project/vllm/pulls/{pr_number}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        _cached_base = data['head']['repo']['full_name']
        _cached_branch = data['head']['ref']
        return _cached_base, _cached_branch
    else:
        logger.error("Failed to fetch PR details: %s", response)
        return None, None


def linkcode_resolve(domain, info):
    if domain != 'py':
        return None
    if not info['module']:
        return None
    filename = info['module'].replace('.', '/')
    module = info['module']

    # try to determine the correct file and line number to link to
    obj = sys.modules[module]

    # get as specific as we can
    lineno: int = 0
    filename: str = ""
    try:
        for part in info['fullname'].split('.'):
            obj = getattr(obj, part)

            if not (inspect.isclass(obj) or inspect.isfunction(obj)
                    or inspect.ismethod(obj)):
                obj = obj.__class__  # Get the class of the instance

            lineno = inspect.getsourcelines(obj)[1]
            filename = (inspect.getsourcefile(obj)
                        or f"{filename}.py").split("vllm/", 1)[1]
    except Exception:
        # For some things, like a class member, won't work, so
        # we'll use the line number of the parent (the class)
        pass

    if filename.startswith("checkouts/"):
        # a PR build on readthedocs
        pr_number = filename.split("/")[1]
        filename = filename.split("/", 2)[2]
        base, branch = get_repo_base_and_branch(pr_number)
        if base and branch:
            return f"https://github.com/{base}/blob/{branch}/{filename}#L{lineno}"

    # Otherwise, link to the source file on the main branch
    return f"https://github.com/vllm-project/vllm/blob/main/{filename}#L{lineno}"


# Mock out external dependencies here, otherwise the autodoc pages may be blank.
autodoc_mock_imports = [
    "blake3",
    "compressed_tensors",
    "cpuinfo",
    "cv2",
    "torch",
    "transformers",
    "psutil",
    "prometheus_client",
    "sentencepiece",
    "vllm._C",
    "PIL",
    "numpy",
    'triton',
    "tqdm",
    "tensorizer",
    "pynvml",
    "outlines",
    "xgrammar",
    "librosa",
    "soundfile",
    "gguf",
    "lark",
    "decord",
]

for mock_target in autodoc_mock_imports:
    if mock_target in sys.modules:
        logger.info(
            "Potentially problematic mock target (%s) found; "
            "autodoc_mock_imports cannot mock modules that have already "
            "been loaded into sys.modules when the sphinx build starts.",
            mock_target)


class MockedClassDocumenter(autodoc.ClassDocumenter):
    """Remove note about base class when a class is derived from object."""

    def add_line(self, line: str, source: str, *lineno: int) -> None:
        if line == "   Bases: :py:class:`object`":
            return
        super().add_line(line, source, *lineno)


autodoc.ClassDocumenter = MockedClassDocumenter

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "typing_extensions":
    ("https://typing-extensions.readthedocs.io/en/latest", None),
    "aiohttp": ("https://docs.aiohttp.org/en/stable", None),
    "pillow": ("https://pillow.readthedocs.io/en/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "psutil": ("https://psutil.readthedocs.io/en/stable", None),
}

autodoc_preserve_defaults = True
autodoc_warningiserror = True

navigation_with_keys = False
