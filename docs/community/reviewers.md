# Reviewers

Reviewers are trusted community members who can perform official reviews, helping contributors get their pull requests ready to merge alongside committers, who give final approval. Authors are expected to respond to reviewer feedback, either by making changes or by explaining why a change isn't needed. The role was introduced in [RFC #58253](https://github.com/vllm-project/vllm/issues/58253).

This page lists the active reviewers and describes the role. See the [committers](../governance/committers.md) page for the committers and area owners who hold final acceptance and merge authority.

## Active Reviewers

Community members can reach out to the reviewers below for help with PRs in their areas, for example in the `#pr-reviews` Slack channel. Reviewers will add their areas of expertise to this list.
Sorted alphabetically by GitHub handle:

- [@andylolu2](https://github.com/andylolu2)
- [@bnellnm](https://github.com/bnellnm) MoE, Quantization layers
- [@BowenBao](https://github.com/BowenBao)
- [@cjackal](https://github.com/cjackal): Multimodality; KV Connector and offload
- [@divakar-amd](https://github.com/divakar-amd)
- [@etelis](https://github.com/etelis): KV cache offloading (CPU, filesystem, and P2P connectors; KV transfer and cache layouts); MoE serving and expert parallelism (developing review focus)
- [@Fangzhou-Ai](https://github.com/Fangzhou-Ai)
- [@fxmarty-amd](https://github.com/fxmarty-amd): Quantization, linear/MOE oracles and modeling definition
- [@GirasoleY](https://github.com/GirasoleY): Kernels and performance
- [@itayalroy](https://github.com/itayalroy): MoE serving (kernels, EP, All2All, EPLB, elasticity/fault tolerance); KV connectors and NIXL integrations
- [@JartX](https://github.com/JartX): ROCm / HIP on Radeon (RDNA3), quantization (W4A16/GPTQ, MXFP4), KV cache quantization
- [@kliuae](https://github.com/kliuae): ROCm feature enablement, model performance, fused kernels
- [@lengrongfu](https://github.com/lengrongfu)
- [@linitra24](https://github.com/linitra24)
- [@LopezCastroRoberto](https://github.com/LopezCastroRoberto): Kernels and quantization
- [@mawong-amd](https://github.com/mawong-amd)
- [@micah-wil](https://github.com/micah-wil): ROCm / AMD GPU integration, CI
- [@netanel-haber](https://github.com/netanel-haber): Nemotron/NVIDIA models, Mamba/Linear attention hybrids, VLMs
- [@Rohan138](https://github.com/Rohan138): ROCm performance, CI, torch.compile fusions
- [@simondanielsson](https://github.com/simondanielsson): ROCm performance (CDNA), MoRI-IO.
- [@taneem-ibrahim](https://github.com/taneem-ibrahim): Pooling models
- [@TheEpicDolphin](https://github.com/TheEpicDolphin)
- [@varun-sundar-rabindranath](https://github.com/varun-sundar-rabindranath): KV cache offloading, LoRA
- [@vllmellm](https://github.com/vllmellm)
- [@wangxiyuan](https://github.com/wangxiyuan): Platform, KV cache, Mooncake

## The Reviewer Role

Reviewers assess the problem and proposed solution, guide validation, resolve concerns with authors, and determine when a contribution is ready for a committer. A reviewer's approval means a PR is ready for a committer's final look.

Reviewers can:

- Approve PRs or request changes.
- Trigger CI when a contribution is ready for evaluation.
- Triage PRs and issues, apply labels, and flag important PRs for expedited committer attention.
- Close duplicate PRs and issues.
- Escalate design questions to the appropriate committer or SIG.
- Flag PRs as ready for committer review.

### Becoming a Reviewer

Candidates should demonstrate a sustained record of useful reviews, sound engineering judgment, constructive communication, and familiarity with vLLM's contribution standards. Deep ownership of a subsystem is not required.

A reviewer is nominated by two committers from different organizations, who post the nomination in the committers Slack channel. As a rule of thumb, not a hard requirement, candidates have either 15 merged PRs and 30 reviewed PRs, or 45 reviewed PRs. A reviewed PR is one where the candidate gave a substantive review and the PR reached a resolution (merged or closed).
