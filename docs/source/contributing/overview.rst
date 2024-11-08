Contributing to vLLM
=====================

Thank you for your interest in contributing to vLLM! Our community is open to everyone and welcomes all kinds of contributions, no matter how small or large. There are several ways you can contribute to the project:

- Identify and report any issues or bugs.
- Request or add support for a new model.
- Suggest or implement new features.
- Improve documentation or contribute a how-to guide.

We also believe in the power of community support; thus, answering queries, offering PR reviews, and assisting others are also highly regarded and beneficial contributions.

Finally, one of the most impactful ways to support us is by raising awareness about vLLM. Talk about it in your blog posts and highlight how it's driving your incredible projects. Express your support on social media if you're using vLLM, or simply offer your appreciation by starring our repository!

License
-------

See `LICENSE <https://github.com/vllm-project/vllm/tree/main/LICENSE>`_.

Developing
----------

Depending on the kind of development you'd like to do (e.g. Python, CUDA), you can choose to build vLLM with or without compilation. Check out the `building from source <https://docs.vllm.ai/en/latest/getting_started/installation.html#build-from-source>`_ documentation for details.

Testing
-------

.. code-block:: bash

    pip install -r requirements-dev.txt

    # linting and formatting
    bash format.sh
    # Static type checking
    mypy
    # Unit tests
    pytest tests/

.. note:: Currently, the repository does not pass the ``mypy`` tests.

Contribution Guidelines
=======================

DCO and Signed-off-by
----------------------

When contributing changes to this project, you must agree to the `DCO <https://github.com/vllm-project/vllm/tree/main/DCO>`_.
Commits must include a ``Signed-off-by:`` header which certifies agreement with
the terms of the `DCO <https://github.com/vllm-project/vllm/tree/main/DCO>`_.

Using ``-s`` with ``git commit`` will automatically add this header.

Issues
------

If you encounter a bug or have a feature request, please `search existing issues <https://github.com/vllm-project/vllm/issues?q=is%3Aissue>`_ first to see if it has already been reported. If not, please `file a new issue <https://github.com/vllm-project/vllm/issues/new/choose>`_, providing as much relevant information as possible.

.. important::
   If you discover a security vulnerability, please follow the instructions `here <https://github.com/vllm-project/vllm/tree/main/SECURITY.md#reporting-a-vulnerability>`_.

Pull Requests & Code Reviews
----------------------------

Please check the PR checklist in the `PR template <https://github.com/vllm-project/vllm/tree/main/.github/PULL_REQUEST_TEMPLATE.md>`_ for a detailed guide for contribution.

Thank You
---------

Finally, thank you for taking the time to read these guidelines and for your interest in contributing to vLLM.
All of your contributions help make vLLM a great tool and community for everyone!
