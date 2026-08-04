# SPDX-License-Identifier: Apache-2.0
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# Standard
from dataclasses import asdict
from typing import Any
import os
import sys

# Third Party
from sphinx.ext import autodoc
from sphinxawesome_theme import ThemeOptions

sys.path.insert(0, os.path.abspath("../.."))

project = "LMCache"
copyright = "2024, The LMCache Team"
author = "The LMCache Team"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.githubpages",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
    "sphinxcontrib.mermaid",
    # "sphinx_copybutton",
    "sphinx_multiversion",
    "sphinxcontrib.images",
    "sphinx_design",
]

copybutton_prompt_text = r"^(\$ |>>> |\# )"
copybutton_prompt_is_regexp = True
autosectionlabel_prefix_document = True


class MockedClassDocumenter(autodoc.ClassDocumenter):
    """Remove note about base class when a class is
    derived from object."""

    def add_line(self, line: str, source: str, *lineno: int) -> None:
        if line == "   Bases: :py:class:`object`":
            return
        super().add_line(line, source, *lineno)


autodoc.ClassDocumenter = MockedClassDocumenter

# autodoc_default_options = {
#     "members": True,
#     "undoc-members": True,
#     "private-members": True
# }

templates_path = ["_templates"]
exclude_patterns: list[Any] = []
add_module_names = False
language = os.environ.get("SPHINX_LANGUAGE", "en")
locale_dirs = ["locale/"]
gettext_compact = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html style
html_title = project
html_theme = "sphinxawesome_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css", "scroll.css"]
html_js_files = ["custom.js"]
html_favicon = "assets/lmcache-logo.png"
html_permalinks_icon = "<span>#</span>"
pygments_style = "sphinx"
pygments_style_dark = "fruity"

theme_options = ThemeOptions(  # Add your theme options.
    extra_header_link_icons={
        "GitHub": {
            "link": "https://github.com/LMCache/LMCache/",
            "icon": (
                '<svg height="26px" style="margin-top:-2px;display:inline" '
                'viewBox="0 0 45 44" '
                'fill="currentColor" xmlns="http://www.w3.org/2000/svg">'
                '<path fill-rule="evenodd" clip-rule="evenodd" '
                'd="M22.477.927C10.485.927.76 10.65.76 22.647c0 9.596 6.223 \
                17.736 '
                "14.853 20.608 1.087.2 1.483-.47 1.483-1.047 "
                "0-.516-.019-1.881-.03-3.693-6.04 "
                "1.312-7.315-2.912-7.315-2.912-.988-2.51-2.412-3.178-2.412 \
                -3.178-1.972-1.346.149-1.32.149-1.32 "  # noqa
                "2.18.154 3.327 2.24 3.327 2.24 1.937 3.318 5.084 2.36 6.321 "
                "1.803.197-1.403.759-2.36 "
                "1.379-2.903-4.823-.548-9.894-2.412-9.894-10.734 "
                "0-2.37.847-4.31 2.236-5.828-.224-.55-.969-2.759.214-5.748 0 0 "
                "1.822-.584 5.972 2.226 "
                "1.732-.482 3.59-.722 5.437-.732 1.845.01 3.703.25 5.437.732 "
                "4.147-2.81 5.967-2.226 "
                "5.967-2.226 1.185 2.99.44 5.198.217 5.748 1.392 1.517 2.232 \
                 3.457 "
                "2.232 5.828 0 "
                "8.344-5.078 10.18-9.916 10.717.779.67 1.474 1.996 1.474 \
                4.021 0 "
                "2.904-.027 5.247-.027 "
                "5.96 0 .58.392 1.256 1.493 1.044C37.981 40.375 44.2 32.24 \
                 44.2 "
                '22.647c0-11.996-9.726-21.72-21.722-21.72" '
                'fill="currentColor"/></svg>'
            ),
        }
    },
)

images_config = {
    "default_image_width": "80%",
    "default_image_target": "_blank",
}

html_theme_options = asdict(theme_options)

# more_options = {
#     # navigation and sidebar
#     'show_toc_level': 2,
#     'announcement': None,
#     'secondary_sidebar_items': [
#         'page-toc',
#     ],
#     'navigation_depth': 3,
#     'primary_sidebar_end': [],
# }

# html_theme_options.update(more_options)

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "typing_extensions": (
        "https://typing-extensions.readthedocs.io/en/latest",
        None,
    ),
    "numpy": ("https://numpy.org/doc/stable", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "psutil": ("https://psutil.readthedocs.io/en/stable", None),
}

# Mock import
autodoc_mock_imports = [
    "sortedcontainers",
    "torch",
    "prometheus_client",
    "yaml",
    "vllm",
    "nvtx",
    "redis",
    "lmcache.c_ops",
    "aiofiles",
    "zmq",
    "transformers",
    "safetensors",
    "torch.Tensor",
]

# -- sphinx-multiversion configuration -------------------------------------------

# Whitelist pattern for tags (build docs for all v* tags)
smv_tag_whitelist = r"^v\d+\.\d+.*$"

# Whitelist pattern for branches (build docs for dev and main)
smv_branch_whitelist = r"^(dev|main)$"

# Pattern for released versions (tags only)
smv_released_pattern = r"^tags/v.*$"

# Remote whitelist pattern (for security)
smv_remote_whitelist = r"^(origin)$"

# Output directories
smv_latest_version = "dev"  # Point latest to dev branch
smv_prefer_remote_refs = False
