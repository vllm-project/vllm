# Collaboration Policy

This page outlines how vLLM collaborates with model vendors, hardware vendors, and other stakeholders.

## Adding New Major Features

Anyone can contribute to vLLM. For major features, submit an RFC first.
Submit the RFC, post it in the #contributors channel, and loop in area owners and committers for feedback.
For high-interest features, we assign a committer to help with the RFC process and PR review.
For contentious features, the vLLM team aims makes decisions quickly after learning the details from everyone. This involves assigning a committer as the DRI (Decision Responsible Individual) to make the decision and shepehrd the process.

## Adding New Models

If you use vLLM, make your model work with vLLM by following the [model registration](../contributing/model/registration.md) before you release it publicly.

The vLLM team helps with new model architectures not supported by vLLM, especially models pushing architectural frontiers.
Here's how the vLLM team works with model vendors. The vLLM team includes all [committers](./committers.md) of the project. Model vendors can exclude certain members but shouldn't, as this may harm release timelines due to missing expertise. Contact project leads if you want to collaborate.

Once we establish the connection between the vLLM team and model vendor:

- The vLLM team learns the model architecture and relevant changes, then plans which area owners to involve and what features to include.
- The vLLM team creates a private communication channel (currently a Slack channel in the vLLM workspace) and a private fork within the vllm-project organization. The model vendor team can invite others to the channel and repo.
- Third parties like CSPs, chip companies, and other organizations often work with both the model vendor and vLLM on model releases. We establish direct communication (with vendor permission) or three-way communication as needed.

The vLLM team works with model vendors on features, integrations, and release timelines. We work to meet release timelines but engineering challenges like feature development, model accuracy alignment, and optimizations can cause delays.

The vLLM maintainers will not publicly share details about model architecture, release timelines, or upcoming releases. We maintain model weights on secure servers with security measures (though we can work with security reviews and testing without certification). We delete pre-release weights or artifacts upon request.

The vLLM team collaborates on marketing and promotional efforts for model releases. Model vendors can use vLLM's trademark and logo in publications and materials.

## Adding New Hardware

vLLM is designed as a platform for frontier model architectures and high-performance accelerators.
For new hardware, follow the [hardware plugin](../design/plugin_system.md) system to add support.
Use the platform plugin system to add hardware support.
As hardware gains popularity, we help endorse it in our documentation and marketing materials.
The vLLM GitHub organization can host hardware plugin repositories, especially for collaborative efforts among companies.

We rarely add new hardware to vLLM directly. Instead, we make existing hardware platforms modular to keep the vLLM core hardware-agnostic.

