Contributing Guide
==================

Thank you for your interest in contributing to LMCache! We welcome and accept all kinds of contributions, no matter how small or large. There are several ways you can contribute to the project:

- Identify and report any issues or bugs
- Request or add support for a new model
- Suggest or implement new features
- Improve documentation or contribute a how-to guide

A comprehensive list of good first issues can be found in the issue `[Onboarding 2026] Good first issues — start contributing to LMCache in one PR <https://github.com/LMCache/LMCache/issues/3372>`_.

If you'd like to support our community further, then answering queries, offering PR reviews, and assisting others are also impactful ways to contribute and take LMCache further.

Finally, you can support us by raising awareness about LMCache. Feel free to share our blog posts, check out our handle on X at `LMCache <https://x.com/lmcache>`_ and see the latest of what we are up to. If using LMCache helped your project or product in any way, you can simply offer your appreciation by starring our repository!

License
-------

See the `LICENSE <https://github.com/LMCache/LMCache/blob/dev/LICENSE>`_ file for details.

Code of Conduct
---------------

This project adheres to the `Code of Conduct <https://github.com/LMCache/LMCache/blob/dev/CODE_OF_CONDUCT.md>`_. By participating, you are expected to uphold this code.

Contribution Guidelines
-----------------------

Help on open source projects is always welcome and there is always something that can be improved. For example, documentation (like the text you are reading now) can always use improvement, code can always be clarified, variables or functions can always be renamed or commented on, and there is always a need for more test coverage. If you see something that you think should be fixed, take ownership! Here is how you get started.

How Can I Contribute?
^^^^^^^^^^^^^^^^^^^^^

When contributing, it's useful to start by looking at `issues <https://github.com/LMCache/LMCache/issues>`_. After picking up an issue, writing code, or updating a document, make a pull request and your work will be reviewed and merged. If you're adding a new feature or find a bug, it's best to `write an issue <https://github.com/LMCache/LMCache/issues/new>`_ first to discuss it with maintainers.

If you discover a security vulnerability, please follow the instructions in the `Security doc <https://github.com/LMCache/LMCache/blob/dev/SECURITY.md>`_.

To contribute to this repo, you'll use the Fork and Pull model common in many open source repositories. For details on this process, check out `The GitHub Workflow Guide <https://github.com/kubernetes/community/blob/master/contributors/guide/github-workflow.md>`_ from Kubernetes. In short:

- Fork the repository
- Create a branch
- Run code style checks and fix any issues
- Run unit tests and fix any broken tests
- Submit a pull request with detailed descriptions

When your contribution is ready, you can create a pull request. Pull requests are often referred to as "PRs". In general, we follow the standard `GitHub pull request <https://help.github.com/en/articles/about-pull-requests>`_ process. Follow the template to provide details about your pull request to the maintainers.

Please try to classify PRs for easy understanding of the type of changes. The PR title is prefixed appropriately to indicate the type of change. Please use one of the following:

- [Bugfix] for bug fixes
- [Build] for build fixes and improvements
- [CI] for continuous integration fixes and iimprovements
- [Core] for changes in the core LMCache logic (e.g., ``LMCacheEngine``, ``Backend`` etc.)
- [Doc] for documentation fixes and improvements
- [Misc] for PRs that do not fit the above categories. Please use this sparingly
- [Model] for adding a new model or improving an existing model. Model name should appear in the title
- [Test] for unit tests

.. note::

    If the PR spans more than one category, please include all relevant prefixes

It's best to break your contribution into smaller PRs with incremental changes, and include a good description of the changes. We require new unit tests to be contributed with any new functionality added and docs if user facing changes.

Before sending pull requests, make sure your changes pass code quality checks and unit tests. These checks will run when the pull request builds. Alternatively, you can run the checks manually on your local machine `as specified in Development <#development>`_ .

DCO and Signed-off-by
^^^^^^^^^^^^^^^^^^^^^

When contributing changes to the project, you must agree to the `DCO <https://github.com/LMCache/LMCache/blob/dev/DCO>`_. Commits must include a :code:`Signed-off-by` header which certifies agreement with the terms of the `DCO <https://github.com/LMCache/LMCache/blob/dev/DCO>`_.

.. note::

    Using :code:`-s` or :code:`--signoff` flag with :code:`git commit` will automatically add this header. The flags can also be used with :code:`--amend` if you forget to sign your last commit.

Code Review
^^^^^^^^^^^

Once you've created a pull request, maintainers will review your code and may make suggestions to fix before merging. It will be easier for your pull request to receive reviews if you consider the criteria the reviewers follow while working. Remember to:

- Document the code well to help future contributors and maintainers
- Follow the project coding conventions for consistency and best practices
- Include sufficient tests to maintain robustness of the code
- Add or update documentation in :code:`/docs` if PR modifies user facing behavior to help people using LMCache
- Run coding style checks to ensure consistency and correctness
- Run tests locally and ensure they pass and don't break the existing code base
- Write detailed commit messages to help future contributors and maintainers
- Break large changes into a logical series of smaller patches, which are easy to understand individually and combine to solve a broader issue

.. note::

    Maintainers will perform "squash and merge" actions on PRs in this repo, so it doesn't matter how many commits your PR has, as they will end up being a single commit after merging.

Development
-----------

Set up your dev environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following prerequisites are required:

- OS: Linux
- GPU: NVIDIA compute capability 7.0+ (e.g., V100, T4, RTX20xx, A100, L4, H100, etc.)
- CUDA 12.8+

The following tools are required:

- `git <https://git-scm.com>`_
- `python <https://www.python.org>`_ (v3.10 -- v3.13)
- `pip <https://pypi.org/project/pip/>`_ (v23.0+)

The first step is to install the necessary Python packages required for development. The commands to do this are as follows:

.. code-block:: bash

    # Equivalent to pip install -r requirements/common.txt
    pip install -e .

    pip install -r requirements/lint.txt
    pip install -r requirements/test.txt

Before pushing changes to GitHub, you need to run the coding style checks and unit tests as shown below.

Coding style
^^^^^^^^^^^^

LMCache follows the Python `pep8 <https://peps.python.org/pep-0008/>`_ coding style for Python and `Google C++ style guide <https://google.github.io/styleguide/cppguide.html>`_ for C++. We use the following tools to enforce the coding style:

- Python linting and formatting: `Ruff <https://docs.astral.sh/ruff/>`_, and `isort <https://pycqa.github.io/isort/>`_
- Python static code checking: `mypy <https://github.com/python/mypy>`_
- Spell checking: `codespell <https://github.com/codespell-project/codespell>`_
- C++ formatting: `clang-format <https://clang.llvm.org/docs/ClangFormat.html>`_

The tools are managed by `pre-commit <https://pre-commit.com/>`_. It is installed as follows:

.. code-block:: bash

    pip install -r requirements/lint.txt
    pre-commit install

It will run automatically when you add a commit. You can also run it manually on all files with the following command:

.. code-block:: bash

    pre-commit run --all-files

.. note::

    For all new code added, please write docstrings in `sphinx-doc format <https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html>`_.

Unit tests
^^^^^^^^^^

When making changes, run the tests before pushing the changes. Running unit tests ensures your contributions do not break existing code. We use the `pytest <https://docs.pytest.org/>`_ framework to run unit tests. The framework is setup to run all files in the `tests <https://github.com/LMCache/LMCache/tree/dev/tests>`_ directory which have a prefix or posfix of "test".

Running unit tests is as simple as:

.. code-block:: bash

    pytest

.. note::

    ``vLLM`` (``pip install vllm``) and the dependencies in ``requirements/test.txt`` need to be installed prior to running unit tests.

By default, all tests found within the tests directory are run. However, specific unit tests can run by passing filenames, classes and/or methods to `pytest`. The following example invokes a single test method "test_lm_connector" that is declared in the "tests/test_connector.py" file:

.. code-block:: bash

    pytest tests/test_connector.py::test_lm_connector

.. note::

    Some unit tests require a NVIDIA GPU. This means that on a non Linux NVIDIA GPU system, the full suite of tests will not be run (tests requiring CUDA will be skipped). The Buildkite continuous integration (CI)  system executes a full run of all the tests.

.. note::

    The `NVIDIA Inference Xfer Library (NIXL) <https://github.com/ai-dynamo/nixl>`_ unit tests require NIXL to be to be installed. This is not installed by LMCache by and therefore requires you to install it separately. Please follow the details in the NIXL GitHub repo to install.
    If the NIXL package is not installed, the NIXL unit tests are skipped.

Building the docs
^^^^^^^^^^^^^^^^^

Install the dependencies:

.. code-block:: bash

    pip install -r requirements/docs.txt

After that, you can build the docs (from :code:`docs/` directory) using `make`:

.. code-block:: bash

    make clean
    make html

Serve docs page locally at http://localhost:8000: :code:`python -m http.server -d build/html/`

Thank You
---------

Thank you for your contribution to LMCache and making it a better, and accessible framework for all. 

