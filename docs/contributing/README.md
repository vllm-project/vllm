# Contributing to vLLM

Thank you for your interest in contributing to vLLM! Our community is open to everyone and welcomes all kinds of contributions, no matter how small or large. There are several ways you can contribute to the project:

- Identify and report any issues or bugs.
- Request or add support for a new model.
- Suggest or implement new features.
- Improve documentation or contribute a how-to guide.

We also believe in the power of community support; thus, answering queries, offering PR reviews, and assisting others are also highly regarded and beneficial contributions.

Finally, one of the most impactful ways to support us is by raising awareness about vLLM. Talk about it in your blog posts and highlight how it's driving your incredible projects. Express your support on social media if you're using vLLM, or simply offer your appreciation by starring our repository!

## Job Board

Unsure on where to start? Check out the following links for tasks to work on:

- [Good first issues](https://github.com/vllm-project/vllm/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22good%20first%20issue%22)
    - [Selected onboarding tasks](https://github.com/orgs/vllm-project/projects/6)
- [New model requests](https://github.com/vllm-project/vllm/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22new-model%22)
    - [Models with multi-modal capabilities](https://github.com/orgs/vllm-project/projects/10)

## License

See [LICENSE](../../LICENSE).

## Developing

The first step of contributing to vLLM is to clone the GitHub repository:

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
```

Then, configure your Python virtual environment.

--8<-- "docs/getting_started/installation/python_env_setup.inc.md"

If you are only developing vLLM's Python code, install vLLM using:

```bash
VLLM_USE_PRECOMPILED=1 uv pip install -e .
```

If you are developing vLLM's Python and CUDA/C++ code, install Pytorch first:

```bash
uv pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu129
```

then install vLLM using:

```bash
uv pip install -e . --no-build-isolation
```

For more details about installing from source and installing for other hardware, check out the [installation instructions](../getting_started/installation/README.md) for your hardware and head to the "Build wheel from source" section.

For an optimized workflow when iterating on C++/CUDA kernels, see the [Incremental Compilation Workflow](./incremental_build.md) for recommendations.

!!! tip
    vLLM is compatible with Python versions 3.10 to 3.13. However, vLLM's default [Dockerfile](../../docker/Dockerfile) ships with Python 3.12 and tests in CI (except `mypy`) are run with Python 3.12.

    Therefore, we recommend developing with Python 3.12 to minimise the chance of your local environment clashing with our CI environment.

### Linting

vLLM uses `pre-commit` to lint and format the codebase. See <https://pre-commit.com/#usage> if `pre-commit` is new to you. Setting up `pre-commit` is as easy as:

```bash
uv pip install pre-commit
pre-commit install
```

vLLM's `pre-commit` hooks will now run automatically every time you commit.

!!! tip "Tips"
    You can manually run the `pre-commit` hooks using:

    ```bash
    pre-commit run     # runs on staged files
    pre-commit run -a  # runs on all files (short for --all-files)
    ```

    ---

    Some `pre-commit` hooks only run in CI. If you need to, you can run them locally with:

    ```bash
    pre-commit run --hook-stage manual markdownlint
    pre-commit run --hook-stage manual mypy-3.10
    ```

### Documentation

MkDocs is a fast, simple and downright gorgeous static site generator that's geared towards building project documentation. Documentation source files are written in Markdown, and configured with a single YAML configuration file, [mkdocs.yaml](../../mkdocs.yaml).

Get started with:

```bash
uv pip install -r requirements/docs.txt
```

!!! tip
    Ensure that your Python version is compatible with the plugins
    (e.g., `mkdocs-awesome-nav` requires Python 3.10+)

MkDocs comes with a built-in dev-server that lets you preview your documentation as you work on it.
From the root of the repository, run:

```bash
mkdocs serve                           # with API ref (~10 minutes)
API_AUTONAV_EXCLUDE=vllm mkdocs serve  # API ref off (~15 seconds)
```

Once you see `Serving on http://127.0.0.1:8000/` in the logs, the live preview is ready!
Open <http://127.0.0.1:8000/> in your browser to see it.

For additional features and advanced configurations, refer to the:

- [MkDocs documentation](https://www.mkdocs.org/)
- [Material for MkDocs documentation](https://squidfunk.github.io/mkdocs-material/) (the MkDocs theme we use)

### Testing

vLLM uses `pytest` to test the codebase.

```bash
# Install the test dependencies used in CI (CUDA only)
uv pip install -r requirements/common.txt -r requirements/dev.txt --torch-backend=auto

# Install some common test dependencies (hardware agnostic)
uv pip install pytest pytest-asyncio

# Run all tests
pytest tests/

# Run tests for a single test file with detailed output
pytest -s -v tests/test_logger.py
```

!!! tip "Install python3-dev if Python.h is missing"
    If any of the above commands fails with `Python.h: No such file or directory`, install
    `python3-dev` with `sudo apt install python3-dev`.

!!! warning "Warnings"
    Currently, the repository is not fully checked by `mypy`.

    ---

    Currently, not all unit tests pass when run on CPU platforms. If you don't have access to a GPU
    platform to run unit tests locally, rely on the continuous integration system to run the tests for
    now.

## Issues

If you encounter a bug or have a feature request, please [search existing issues](https://github.com/vllm-project/vllm/issues?q=is%3Aissue) first to see if it has already been reported. If not, please [file a new issue](https://github.com/vllm-project/vllm/issues/new/choose), providing as much relevant information as possible.

!!! important
    If you discover a security vulnerability, please follow the instructions [here](../../SECURITY.md).

## Pull Requests & Code Reviews

Thank you for your contribution to vLLM! Before submitting the pull request,
please ensure the PR meets the following criteria. This helps vLLM maintain the
code quality and improve the efficiency of the review process.

### DCO and Signed-off-by

When contributing changes to this project, you must agree to the [DCO](../../DCO).
Commits must include a `Signed-off-by:` header which certifies agreement with
the terms of the DCO.

Using `-s` with `git commit` will automatically add this header.

!!! tip
    You can enable automatic sign-off via your IDE:

    - **PyCharm**: Click on the `Show Commit Options` icon to the right of the `Commit and Push...` button in the `Commit` window.
      It will bring up a `git` window where you can modify the `Author` and enable `Sign-off commit`.
    - **VSCode**: Open the [Settings editor](https://code.visualstudio.com/docs/configure/settings)
      and enable the `Git: Always Sign Off` (`git.alwaysSignOff`) field.

### PR Title and Classification

Only specific types of PRs will be reviewed. The PR title is prefixed
appropriately to indicate the type of change. Please use one of the following:

- `[Bugfix]` for bug fixes.
- `[CI/Build]` for build or continuous integration improvements.
- `[Doc]` for documentation fixes and improvements.
- `[Model]` for adding a new model or improving an existing model. Model name
  should appear in the title.
- `[Frontend]` For changes on the vLLM frontend (e.g., OpenAI API server,
  `LLM` class, etc.)
- `[Kernel]` for changes affecting CUDA kernels or other compute kernels.
- `[Core]` for changes in the core vLLM logic (e.g., `LLMEngine`,
  `AsyncLLMEngine`, `Scheduler`, etc.)
- `[Hardware][Vendor]` for hardware-specific changes. Vendor name should
  appear in the prefix (e.g., `[Hardware][AMD]`).
- `[Misc]` for PRs that do not fit the above categories. Please use this
  sparingly.

!!! note
    If the PR spans more than one category, please include all relevant prefixes.

### Code Quality

The PR needs to meet the following code quality standards:

- We adhere to [Google Python style guide](https://google.github.io/styleguide/pyguide.html) and [Google C++ style guide](https://google.github.io/styleguide/cppguide.html).
- Pass all linter checks.
- The code needs to be well-documented to ensure future contributors can easily
  understand the code.
- Include sufficient tests to ensure the project stays correct and robust. This
  includes both unit tests and integration tests.
- Please add documentation to `docs/` if the PR modifies the user-facing behaviors of vLLM.
  It helps vLLM users understand and utilize the new features or changes.

### Adding or Changing Kernels

When actively developing or modifying kernels, using the [Incremental Compilation Workflow](./incremental_build.md) is highly recommended for faster build times.
Each custom kernel needs a schema and one or more implementations to be registered with PyTorch.

- Make sure custom ops are registered following PyTorch guidelines:
  [Custom C++ and CUDA Operators](https://pytorch.org/tutorials/advanced/cpp_custom_ops.html#cpp-custom-ops-tutorial)
  and [The Custom Operators Manual](https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU).
- Custom operations that return `Tensors` require meta-functions.
  Meta-functions should be implemented and registered in Python so that dynamic
  dims can be handled automatically. See above documents for a description of
  meta-functions.
- Use [torch.library.opcheck()](https://pytorch.org/docs/stable/library.html#torch.library.opcheck)
  to test the function registration and meta-function for any registered ops.
  See `tests/kernels` for examples.
- When changing the C++ signature of an existing op, the schema must be updated
  to reflect the changes.
- If a new custom type is needed, see the following document:
  [Custom Class Support in PT2](https://docs.google.com/document/d/18fBMPuOJ0fY5ZQ6YyrHUppw9FA332CpNtgB6SOIgyuA).

### Notes for Large Changes

Please keep the changes as concise as possible. For major architectural changes
(>500 LOC excluding kernel/data/config/test), we would expect a GitHub issue
(RFC) discussing the technical design and justification. Otherwise, we will tag
it with `rfc-required` and might not go through the PR.

### What to Expect for the Reviews

The goal of the vLLM team is to be a *transparent reviewing machine*. We would
like to make the review process transparent and efficient and make sure no
contributor feels confused or frustrated. However, the vLLM team is small, so we
need to prioritize some PRs over others. Here is what you can expect from the
review process:

- After the PR is submitted, the PR will be assigned to a reviewer. Every
  reviewer will pick up the PRs based on their expertise and availability.
- After the PR is assigned, the reviewer will provide status updates every 2-3
  days. If the PR is not reviewed within 7 days, please feel free to ping the
  reviewer or the vLLM team.
- After the review, the reviewer will put an `action-required` label on the PR
  if there are changes required. The contributor should address the comments and
  ping the reviewer to re-review the PR.
- Please respond to all comments within a reasonable time frame. If a comment
  isn't clear or you disagree with a suggestion, feel free to ask for
  clarification or discuss the suggestion.
- Note that not all CI checks will be executed due to limited computational
  resources. The reviewer will add `ready` label to the PR when the PR is
  ready to merge or a full CI run is needed.

## Thank You

Finally, thank you for taking the time to read these guidelines and for your interest in contributing to vLLM.
All of your contributions help make vLLM a great tool and community for everyone!
