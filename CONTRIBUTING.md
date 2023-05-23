# Contributing to CacheFlow

Thank you for your interest in contributing to CacheFlow!

## Contributing Guidelines

### Issue Reporting

If you encounter a bug or have a feature request, please check our issues page first to see if someone else has already reported it. If not, please file a new issue, providing as much relevant information as possible.

### Pull Requests

When submitting a pull request:

1. Make sure your code has been rebased on top of the latest commit on the main branch.
2. Include a detailed description of the changes in the pull request. Explain why you made the changes you did.
If your pull request fixes an open issue, please include a reference to it in the description.

### Code Reviews

All submissions, including submissions by project members, require a code review. To make the review process as smooth as possible, please:

1. Keep your changes as concise as possible. If your pull request involves multiple unrelated changes, consider splitting it into separate pull requests.
2. Respond to all comments within a reasonable time frame. If a comment isn't clear or you disagree with a suggestion, feel free to ask for clarification or discuss the suggestion.

### Installation for development

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

## Coding Style Guide

In general, we adhere to [Google Python style guide](https://google.github.io/styleguide/pyguide.html) and [Google C++ style guide](https://google.github.io/styleguide/cppguide.html).
