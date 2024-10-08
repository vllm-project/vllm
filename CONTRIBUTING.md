# Contributing to vLLM

Thank you for your interest in contributing to vLLM! Our community is open to everyone and welcomes all kinds of contributions, no matter how small or large. There are several ways you can contribute to the project:

- Identify and report any issues or bugs.
- Request or add support for a new model.
- Suggest or implement new features.
- Improve documentation or contribute a how-to guide. 

We also believe in the power of community support; thus, answering queries, offering PR reviews, and assisting others are also highly regarded and beneficial contributions.

Finally, one of the most impactful ways to support us is by raising awareness about vLLM. Talk about it in your blog posts and highlight how it's driving your incredible projects. Express your support on social media if you're using vLLM, or simply offer your appreciation by starring our repository!


## Developing

Depending on the kind of development you'd like to do (e.g. Python, CUDA), you can choose to build vLLM either [without compilation](https://docs.vllm.ai/en/latest/getting_started/installation.html#build-from-source-without-compilation) or [with compilation](https://docs.vllm.ai/en/latest/getting_started/installation.html#build-from-source-with-compilation).

## Testing

```bash
pip install -r requirements-dev.txt

# linting and formatting
bash format.sh
# Static type checking
mypy
# Unit tests
pytest tests/
```
**Note:** Currently, the repository does not pass the ``mypy`` tests.

## Contributing Guidelines

### Issues

If you encounter a bug or have a feature request, please check our issues page first to see if someone else has already reported it.
If not, please file a new issue, providing as much relevant information as possible.

### Pull Requests & Code Reviews

Please check the PR checklist in the [PR template](.github/PULL_REQUEST_TEMPLATE.md) for detailed guide for contribution.

### Thank You

Finally, thank you for taking the time to read these guidelines and for your interest in contributing to vLLM.
All of your contributions help make vLLM a great tool and community for everyone!
