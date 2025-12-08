# Collaboration Policy

This page outlines how vLLM collaborates with model providers, hardware vendors, and other stakeholders.

## Adding New Major Features

Anyone can contribute to vLLM. For major features, submit an RFC (request for comments) first. To submit an RFC, create an [issue](https://github.com/vllm-project/vllm/issues/new/choose) and select the `RFC` template.
RFCs are similar to design docs that discuss the motivation, problem solved, alternatives considered, and proposed change.

Once you submit the RFC, please post it in the #contributors channel in vLLM Slack, and loop in area owners and committers for feedback.
For high-interest features, the committers nominate a person to help with the RFC process and PR review. This makes sure someone is guiding you through the process. It is reflected as the "assignee" field in the RFC issue.
If the assignee and lead maintainers find the feature to be contentious, the maintainer team aims to make decisions quickly after learning the details from everyone. This involves assigning a committer as the DRI (Directly Responsible Individual) to make the decision and shepherd the code contribution process.

For features that you intend to maintain, please feel free to add yourself in [`mergify.yml`](https://github.com/vllm-project/vllm/blob/main/.github/mergify.yml) to receive notifications and auto-assignment when the PRs touching the feature you are maintaining. Over time, the ownership will be evaluated and updated through the committers nomination and voting process.

## Adding New Models

If you use vLLM, we recommend you making the model work with vLLM by following the [model registration](../contributing/model/registration.md) process before you release it publicly.

The vLLM team helps with new model architectures not supported by vLLM, especially models pushing architectural frontiers.
Here's how the vLLM team works with model providers. The vLLM team includes all [committers](./committers.md) of the project. model providers can exclude certain members but shouldn't, as this may harm release timelines due to missing expertise. Contact [project leads](./process.md) if you want to collaborate.

Once we establish the connection between the vLLM team and model provider:

- The vLLM team learns the model architecture and relevant changes, then plans which area owners to involve and what features to include.
- The vLLM team creates a private communication channel (currently a Slack channel in the vLLM workspace) and a private fork within the vllm-project organization. The model provider team can invite others to the channel and repo.
- Third parties like compute providers, hosted inference providers, hardware vendors, and other organizations often work with both the model provider and vLLM on model releases. We establish direct communication (with permission) or three-way communication as needed.

The vLLM team works with model providers on features, integrations, and release timelines. We work to meet release timelines, but engineering challenges like feature development, model accuracy alignment, and optimizations can cause delays.

The vLLM maintainers will not publicly share details about model architecture, release timelines, or upcoming releases. We maintain model weights on secure servers with security measures (though we can work with security reviews and testing without certification). We delete pre-release weights or artifacts upon request.

The vLLM team collaborates on marketing and promotional efforts for model releases. model providers can use vLLM's trademark and logo in publications and materials.

## Adding New Hardware

vLLM is designed as a platform for frontier model architectures and high-performance accelerators.
For new hardware, follow the [hardware plugin](../design/plugin_system.md) system to add support.
Use the platform plugin system to add hardware support.
As hardware gains popularity, we help endorse it in our documentation and marketing materials.
The vLLM GitHub organization can host hardware plugin repositories, especially for collaborative efforts among companies.

We rarely add new hardware to vLLM directly. Instead, we make existing hardware platforms modular to keep the vLLM core hardware-agnostic.
