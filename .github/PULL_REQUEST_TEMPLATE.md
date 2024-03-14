## PR Checklist

Before submitting your pull request, please ensure you have met the following criteria. This helps us maintain quality and efficiency in our review process. Check each box to indicate your understanding and compliance:

### PR Title And Classification

- [ ] **I understand that** only specific types of PRs will be reviewed. My PR title is prefixed appropriately to indicate the type of change, using one of the following:

  - [ ] `[Doc]` for documentation fixes.
  - [ ] `[Bugfix]` for bug fixes.
  - [ ] `[CI/Build]` for build or continuous integration improvements.
  - [ ] `[Model]` for adding a new model or improving an existing model. Model name should appear in the title.
  - [ ] `[Kernel]` for changes affecting computation kernels.
  - [ ] `[Hardware][Vendor]` for hardware-specific changes. Vendor name should appear in the prefix, e.g., `[Hardware][AMD]`.
  - [ ] `[Misc]` for PRs that do not fit the above categories. Please use this sparingly.

  - **Note:** If my PR spans more than one category, I will include all relevant prefixes.

### Coding Standards

- [ ] **I understand that** I must run `./format.sh` **before submitting the PR and after any new commits** to ensure compliance with linter checks. PRs failing to meet linter standards will not be reviewed or merged.

### Code Quality

- [ ] **I understand that** my code must be well-commented and include sufficient tests, ensuring future contributors can easily understand and modify the codebase.

### Documentation for User-Facing Changes

- [ ] **I understand that** if my PR introduces user-facing changes, it must be accompanied by relevant documentation to help users understand and utilize the new features or changes.

Thank you for your contribution!

## PR Justification

Please provide a brief explanation of the motivation behind the PR and the changes it introduces. This helps reviewers understand the context and rationale for the contribution.
