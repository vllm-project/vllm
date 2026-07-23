# Contributing to vLLM

You may find information about contributing to vLLM on [docs.vllm.ai](https://docs.vllm.ai/en/latest/contributing).

## Fork PR Link Rule (vLLM-HUST)

`vLLM-HUST/vllm-hust` is a fork of `vllm-project/vllm`.
GitHub may default to the upstream repository when opening PRs via a short URL
like `/pull/new/<branch>`. This can cause contributors to accidentally open a PR
to `vllm-project/vllm` instead of `vLLM-HUST/vllm-hust`.

Use one of the following safe methods:

1. Preferred URL template (explicit base/head):
 `https://github.com/vLLM-HUST/vllm-hust/compare/main...vLLM-HUST:<branch>?expand=1`
2. GitHub CLI (explicit repo/base/head):
 `gh pr create --repo vLLM-HUST/vllm-hust --base main --head <branch>`

Avoid using this short form in docs/messages:

- `https://github.com/vLLM-HUST/vllm-hust/pull/new/<branch>`

This rule prevents cross-repo PR misrouting for all fork-based workflows.
