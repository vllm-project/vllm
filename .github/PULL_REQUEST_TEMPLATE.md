<details>
<!-- inside this <details> section, markdown rendering does not work, so we have to use raw html here. -->
<summary><h2> PR Checklist (Click to expand. Please do have a look before submitting a PR!) </h2></summary>
<p>Before submitting your pull request, please ensure you have met the following criteria. This helps us maintain quality and efficiency in our review process.</p>

<h3>PR Title And Classification</h3>
<p><strong>I understand that</strong> only specific types of PRs will be reviewed. My PR title is prefixed appropriately to indicate the type of change, using one of the following:</p>
<ul>
    <li><code>[Doc]</code> for documentation fixes.</li>
    <li><code>[Bugfix]</code> for bug fixes.</li>
    <li><code>[CI/Build]</code> for build or continuous integration improvements.</li>
    <li><code>[Model]</code> for adding a new model or improving an existing model. Model name should appear in the title.</li>
    <li><code>[Kernel]</code> for changes affecting computation kernels.</li>
    <li><code>[Hardware][Vendor]</code> for hardware-specific changes. Vendor name should appear in the prefix, e.g., <code>[Hardware][AMD]</code>.</li>
    <li><code>[Misc]</code> for PRs that do not fit the above categories. Please use this sparingly.</li>
</ul>
<p><strong>Note:</strong> If my PR spans more than one category, I will include all relevant prefixes.</p>

<h3>Coding Standards</h3>
<p><strong>I understand that</strong> I must run <code>./format.sh</code> <strong>before submitting the PR and after any new commits</strong> to ensure compliance with linter checks. PRs failing to meet linter standards will not be merged.</p>

<h3>Code Quality</h3>
<p><strong>I understand that</strong> my code must be well-commented and include sufficient tests, ensuring future contributors can easily understand and modify the codebase.</p>

<h3>Documentation for User-Facing Changes</h3>
<p><strong>I understand that</strong> if my PR introduces user-facing changes, it must be accompanied by relevant documentation to help users understand and utilize the new features or changes.</p>

<p>Thank you for your contribution!</p>

</details>

## PR Justification

Please provide a brief explanation of the motivation behind the PR and the changes it introduces. This helps reviewers understand the context and rationale for the contribution.

If possible, please link existing issues this PR will resolve.

## Note for large changes

For major architectural changes (>500 LOC excluding kernel/data/config/test), we would expect a GitHub issue discussing the technical design and justification. Otherwise, we will tag it with `rfc-required` and might not take a look at the PR.
