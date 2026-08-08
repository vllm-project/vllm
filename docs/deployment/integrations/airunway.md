# AI Runway

[AI Runway](https://github.com/kaito-project/airunway) is a Kubernetes-native platform for deploying and managing LLM inference across multiple providers behind a single `ModelDeployment` API. It can serve models with vLLM directly — running the OpenAI-compatible `vllm serve` server as native Kubernetes `Deployment` and `Service` resources — or through vLLM-backed providers such as NVIDIA Dynamo, llm-d, KAITO, and KubeRay, all selected automatically from provider capabilities.

For deploying vLLM directly, see the [Direct vLLM provider guide](https://kaito-project.github.io/airunway/providers/vllm). For an overview and installation, see the [AI Runway documentation](https://kaito-project.github.io/airunway/).
