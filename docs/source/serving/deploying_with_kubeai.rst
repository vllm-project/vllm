.. _deploying_with_kubeai:

Deploying with KubeAI
=====================

`KubeAI <https://github.com/substratusai/kubeai>`_ is a Kubernetes operator that enables you to deploy and manage AI models on Kubernetes. It provides a simple and scalable way to deploy vLLM in production. Functionality such as scale-from-zero, load based autoscaling, model caching, and much more is provided out of the box with zero external dependencies.


Please see the Installation Guides for environment specific instructions:

* `Any Kubernetes Cluster <https://www.kubeai.org/installation/any/>`_
* `EKS <https://www.kubeai.org/installation/eks/>`_
* `GKE <https://www.kubeai.org/installation/gke/>`_

Once you have KubeAI installed, you can
`configure text generation models <https://www.kubeai.org/how-to/configure-text-generation-models/>`_
using vLLM.