# Tips for Developing Documentation with MkDocs

# Table of Contents

- [Introduction to MkDocs](#introduction-to-mkdocs)
- [Installation Steps](#installation-steps)
    - [Set Up a Virtual Environment](#set-up-a-virtual-environment)
    - [Install MkDocs and Plugins](#install-mkdocs-and-plugins)
    - [Verify Installation](#verify-installation)
    - [Clone the vLLM Repository](#clone-the-vllm-repository)
    - [Start the Development Server](#start-the-development-server)
    - [View in Your Browser](#view-in-your-browser)
    - [Learn More](#learn-more)

## Introduction to MkDocs

[MkDocs](https://github.com/mkdocs/mkdocs) is a fast, simple and downright gorgeous static site generator that's geared towards building project documentation. Documentation source files are written in Markdown, and configured with a single YAML configuration file.

## Installation Steps

### Set Up a Virtual Environment

It's recommended to use a virtual environment to manage dependencies. For example, using Conda:

```bash
conda create -n mkdocs python=3.10
conda activate mkdocs
```

### Install MkDocs and Plugins

Install MkDocs along with the [plugins](https://github.com/vllm-project/vllm/blob/main/mkdocs.yaml) used in the vLLM documentation, as well as required dependencies:

```console
pip install mkdocs \
    pymdown-extensions \
    mkdocs-material \
    python-markdown-math \
    mkdocs-autorefs \
    mkdocs-awesome-nav \
    mkdocs-api-autonav \
    regex
```

> **Note:** Ensure that your Python version is compatible with the plugins (e.g., mkdocs-awesome-nav requires Python 3.10+)

### Verify Installation

Confirm that MkDocs is correctly installed::

```bash
mkdocs --version
```

Example output:

```console
mkdocs, version 1.6.1 from /opt/miniconda3/envs/mkdoc/lib/python3.9/site-packages/mkdocs (Python 3.9)
```

### Clone the `vLLM` repository

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
```

### Start the Development Server

MkDocs comes with a built-in dev-server that lets you preview your documentation as you work on it. Make sure you're in the same directory as the `mkdocs.yml` configuration file, and then start the server by running the `mkdocs serve` command:

```bash
mkdocs serve
```

Example output:

```console
INFO    -  Documentation built in 106.83 seconds
INFO    -  [22:02:02] Watching paths for changes: 'docs', 'mkdocs.yaml'
INFO    -  [22:02:02] Serving on http://127.0.0.1:8000/
```

### View in Your Browser

Open up [http://127.0.0.1:8000/](http://127.0.0.1:8000/) in your browser to see a live preview:.

### Learn More

For additional features and advanced configurations, refer to the official [MkDocs Documentation](https://www.mkdocs.org/).
