# Collaboration Policy

This page outlines how vLLM collaborates with model vendors, hardware vendors, and other stakeholders.

## Adding new major features

Anyone can contribute to vLLM. However, for major features, we encourage you to submit an RFC first.
You should submit the RFC, post it in #contributors channel, and actively loop in area owners and committers for feedback.
For features that are of high interest, we will assign a committer to help you with the RFC process and review the PR.
For features that are contentious, vLLM team strives to make decision quickly after learning the details. 

## Adding new models

If you are already using vLLM, you can make your model work with vLLM by following the [model registration](../contributing/model/registration.md) before you publically release it. 

The vLLM team can help you with new model architectures that are not supported by vLLM, especially for models pushing the frontier of model architectures.
Here is an explainer on how the vLLM team works with model vendors. The vLLM team is defined broadly as the [committers](./committers.md) of the project. The model vendors can exclude certain members but are not encouraged to do so as it may harm the ability to meet the release timeline due to lack of area expertise. Please reach out the project leads if you are interested to collaborate.

Upon establishing the connection between vLLM team and model vendor, the following will happen:

- vLLM team will learn about the model architecture and relevant changes, plan out which area owners to involve and what feature works to be included. 
- vLLM team will create a private communication channel (currently, a slack channel in vLLM workspace) for communication and a private fork within vllm-project organization. The model vendor team will have the ability to invite others to the channel and repo. 
- Typically, third parties such as CSPs, chips, and companies will also work with model vendor AND vLLM to collaborate on the model release. In that case, we are happy to establish either direct communication (vLLM talking with the third party with the permission of the vendor) or 3-way communication as desired. 

Over time, the vLLM team will work with model vendors on features, integrations, and release timelines. We will work to meet the release timeline to the best of our ability. Overall, the engineering challenges such as feature development timeline, model accuracy alignment, and optimizations can delay the timeline. 

Throughout the process, the vLLM maintainers will not share publicly details related to model architecture, release timeline, and information related to upcoming release. The model weights will be maintained in exclusive servers with security measures (although we will be able to work with security reviews and testing, nor have any certification to attest our practices). Upon request, we will delete any pre-release weights or artifacts. 

vLLM team is happy to collaborate with any marketing and promotional efforts related to the model releases. The model vendor can use vLLM’s trademark and logo in its publications and materials. 

## Adding new hardware

vLLM is designed to be a platform for frontier model architectures and the most performant accelerators to shine. 
For new hardware, we recommend you to follow the [hardware plugin](../design/plugin_system.md) system to add support for your hardware. 
In particular, you should leverage the platform plugin system to add support for your hardware. 
As the hardware gains popularity, we can help endorse the hardware in our documentation and marketing materials.
The vLLM GitHub organization can also be used to host the hardware plugin repository especially you want it to be a collaborative effort among many companies. 

We rarely consider adding new hardware to vLLM directly; in fact, we want to make existing hardware platforms modular so the vLLM core is as agnostic as possible. 

