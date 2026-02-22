# Task Completion Checklist

When completing a development task in vLLM, ensure the following steps are completed:

## 1. Code Quality Checks

### Automatic Pre-commit Hooks
Pre-commit hooks will run automatically on `git commit` and check:
- **ruff-check**: Python linting (pycodestyle, Pyflakes, pyupgrade, flake8-bugbear, etc.)
- **ruff-format**: Python code formatting
- **typos**: Spell checking across all file types
- **clang-format**: C++/CUDA code formatting
- **mypy**: Type checking (for your Python version locally)
- **shellcheck**: Shell script linting
- **SPDX header check**: Ensure license headers are present
- **Signed-off-by**: DCO sign-off is added automatically

### Manual Checks
If needed, run manually:
```bash
pre-commit run -a
```

## 2. Testing

### Run Relevant Tests
```bash
# Run tests related to your changes
pytest tests/<relevant_test_directory>/

# Run specific test file
pytest -s -v tests/test_<your_feature>.py
```

### Add New Tests
- Add unit tests for new functionality
- Add integration tests for end-to-end features
- Ensure tests are in the appropriate `tests/` subdirectory
- Use appropriate pytest markers if needed

## 3. Documentation

### Update Documentation if Needed
- Update `docs/` if the PR modifies user-facing behaviors
- Add docstrings to new classes, methods, and functions
- Update README.md if adding major features
- Check documentation builds: `mkdocs serve`

## 4. Commit Guidelines

### DCO Sign-off (Required)
All commits must include a `Signed-off-by:` header:
```bash
git commit -s -m "Your commit message"
```

### Commit Message Format
Follow this format for commit messages:
```
Brief summary of changes (50 chars or less)

More detailed explanatory text if needed. Wrap at 72 characters.
Explain the problem that this commit is solving, focus on why you
are making this change as opposed to how.

Signed-off-by: Your Name <your.email@example.com>
```

## 5. PR Requirements

### PR Title Prefix
Use one of these prefixes:
- `[Bugfix]` - Bug fixes
- `[CI/Build]` - Build or CI improvements
- `[Doc]` - Documentation fixes/improvements
- `[Model]` - Adding/improving models (include model name)
- `[Frontend]` - vLLM frontend changes (API server, LLM class, etc.)
- `[Kernel]` - CUDA/compute kernel changes
- `[Core]` - Core vLLM logic (LLMEngine, Scheduler, etc.)
- `[Hardware][Vendor]` - Hardware-specific changes
- `[Misc]` - Other changes (use sparingly)

### Code Review Checklist
- [ ] Code follows Google Python/C++ style guides
- [ ] All linter checks pass
- [ ] Code is well-documented with clear comments
- [ ] Sufficient tests are included
- [ ] Documentation updated if needed
- [ ] All commits are signed off (DCO)
- [ ] PR description clearly explains changes
- [ ] Large changes (>500 LOC) have an associated GitHub issue/RFC

## 6. Special Considerations

### For Kernel Changes
- Follow PyTorch custom ops guidelines
- Implement and register meta-functions for ops returning Tensors
- Use `torch.library.opcheck()` to test function registration
- Update schema if changing C++ signature
- See `tests/kernels` for examples

### For Large Changes
- Create GitHub issue (RFC) for major architectural changes (>500 LOC)
- Discuss technical design and justification before implementation

## 7. CI/CD

### CI Checks
- Some checks only run in CI (not locally)
- Reviewer will add `ready` label for full CI run
- Address any CI failures promptly

## 8. Final Steps

Before marking your work as complete:
1. ✅ All pre-commit hooks pass
2. ✅ All tests pass locally
3. ✅ Documentation is updated
4. ✅ Commits are signed off
5. ✅ PR title has correct prefix
6. ✅ PR description is clear and complete
