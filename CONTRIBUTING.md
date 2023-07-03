# Contributing to vLLM

Thank you for your interest in contributing to vLLM!
Our community is open to everyone and welcomes all kinds of contributions, no matter how small or large.
There are several ways you can contribute to the project:

- Identify and report any issues or bugs.
- Request or add a new model.
- Suggest or implement new features.

However, remember that contributions aren't just about code.
We believe in the power of community support; thus, answering queries, assisting others, and enhancing the documentation are highly regarded and beneficial contributions.

Finally, one of the most impactful ways to support us is by raising awareness about vLLM.
Talk about it in your blog posts, highlighting how it's driving your incredible projects.
Express your support on Twitter if vLLM aids you, or simply offer your appreciation by starring our repository.


## Setup for development

### Build from source

```bash
pip install -r requirements.txt
pip install -e .  # This may take several minutes.
```

### Testing

```bash
pip install -r requirements-dev.txt

# Static type checking
mypy
# Unit tests
pytest tests/
```
**Note:** Currently, the repository does not pass the mypy tests.


## Contributing Guidelines

### Issue Reporting

If you encounter a bug or have a feature request, please check our issues page first to see if someone else has already reported it.
If not, please file a new issue, providing as much relevant information as possible.

### Coding Style Guide

In general, we adhere to [Google Python style guide](https://google.github.io/styleguide/pyguide.html) and [Google C++ style guide](https://google.github.io/styleguide/cppguide.html).

We include a formatting script [`format.sh`](./format.sh) to format the code.

### Pull Requests

When submitting a pull request:

1. Make sure your code has been rebased on top of the latest commit on the main branch.
2. Ensure code is properly formatted by running [`format.sh`](./format.sh).
3. Include a detailed description of the changes in the pull request.
Explain why you made the changes you did.
If your pull request fixes an open issue, please include a reference to it in the description.

### Code Reviews

All submissions, including submissions by project members, require a code review.
To make the review process as smooth as possible, please:

1. Keep your changes as concise as possible.
If your pull request involves multiple unrelated changes, consider splitting it into separate pull requests.
2. Respond to all comments within a reasonable time frame.
If a comment isn't clear or you disagree with a suggestion, feel free to ask for clarification or discuss the suggestion.

### Thank You

Finally, thank you for taking the time to read these guidelines and for your interest in contributing to vLLM.
Your contributions make vLLM a great tool for everyone!
