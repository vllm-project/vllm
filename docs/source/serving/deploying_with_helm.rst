.. _deploying_with_helm:

Deploying with Helm
===================

A Helm chart to deploy vLLM for Kubernetes

Helm is a package manager for Kubernetes. It will help you to deploy vLLM on k8s and automate the deployment of vLLMm Kubernetes applications. With Helm, you can deploy the same framework architecture with different configurations to multiple namespaces by overriding variables values.

This guide will walk you through the process of deploying vLLM with Helm, including the necessary prerequisites, steps for helm install and documentation on architecture and values file.

Prerequisites
-------------
Before you begin, ensure that you have the following:

- A running Kubernetes cluster
- NVIDIA Kubernetes Device Plugin (`k8s-device-plugin`): This can be found at `https://github.com/NVIDIA/k8s-device-plugin/`
- Available GPU resources in your cluster
- S3 with the model which will be deployed

Installing the chart
--------------------

To install the chart with the release name `test-vllm`

.. code-block:: console

    helm upgrade --install --create-namespace --namespace=ns-vllm test-vllm . -f values.yaml --set secrets.s3endpoint=$ACCESS_POINT --set secrets.s3buckername=$BUCKET --set secrets.s3accesskeyid=$ACCESS_KEY --set secrets.s3accesskey=$SECRET_KEY

Uninstalling the Chart
----------------------

To uninstall the `test-vllm` deployment

.. code-block:: console

    helm uninstall test-vllm --namespace=ns-vllm

The command removes all the Kubernetes components associated with the
chart **including persistent volumes** and deletes the release.

Architecture
------------

.. image:: architecture_helm_deployment.excalidraw.png

Values
------

+---------------------+---------+-----------------------+---------------+
| Key                 | Type    | Default               | Description   |
+=====================+=========+=======================+===============+
| autoscaling         | object  | ``{"ena               | Autoscaling   |
|                     |         | bled":false,"maxRepli | configuration |
|                     |         | cas":100,"minReplicas |               |
|                     |         | ":1,"targetCPUUtiliza |               |
|                     |         | tionPercentage":80}`` |               |
+---------------------+---------+-----------------------+---------------+
| autoscaling.enabled | bool    | ``false``             | Enable        |
|                     |         |                       | autoscaling   |
+---------------------+---------+-----------------------+---------------+
| autoscaling.        | int     | ``100``               | Maximum       |
| maxReplicas         |         |                       | replicas      |
+---------------------+---------+-----------------------+---------------+
| autoscaling.        | int     | ``1``                 | Minimum       |
| minReplicas         |         |                       | replicas      |
+---------------------+---------+-----------------------+---------------+
| autoscaling.        | int     | ``80``                | Target CPU    |
| targetCPUUt         |         |                       | utilization   |
| ilizationPercentage |         |                       | for           |
|                     |         |                       | autoscaling   |
+---------------------+---------+-----------------------+---------------+
| configs             | object  | ``{}``                | Configmap     |
+---------------------+---------+-----------------------+---------------+
| containerPort       | int     | ``8000``              | Container     |
|                     |         |                       | port          |
+---------------------+---------+-----------------------+---------------+
| customObjects       | list    | ``[]``                | Custom        |
|                     |         |                       | Objects       |
|                     |         |                       | configuration |
|                     |         |                       |               |
+---------------------+---------+-----------------------+---------------+
| deploymentStrategy  | object  | ``{}``                | Deployment    |
|                     |         |                       | strategy      |
|                     |         |                       | configuration |
|                     |         |                       |               |
+---------------------+---------+-----------------------+---------------+
| externalConfigs     | list    | ``[]``                | External      |
|                     |         |                       | configuration |
|                     |         |                       |               |
+---------------------+---------+-----------------------+---------------+
| extraContainers     | list    | ``[]``                | Additional    |
|                     |         |                       | containers    |
|                     |         |                       | configuration |
|                     |         |                       |               |
+---------------------+---------+-----------------------+---------------+
| extraInit           | object  | ``{"pvcStorage":"1Gi  | Additional    |
|                     |         | ","s3modelpath":"rela | configuration |
|                     |         | tive_s3_model_path/op | for the       |
|                     |         | t-125m", "awsEc2Metad | init          |
|                     |         | ataDisabled": true}`` | container     |
|                     |         |                       |               |
+---------------------+---------+-----------------------+---------------+
| extraInit.          | string  | ``"50Gi"``            | Storage       |
| pvcStorage          |         |                       | size of the   |
|                     |         |                       | s3            |
+---------------------+---------+-----------------------+---------------+
| exraInit.           | string  | ``"relative_s3_m      | Path of the   |
| s3modelpath         |         | odel_path/opt-125m"`` | model on      |
|                     |         |                       | the s3        |
|                     |         |                       | which hosts   |
|                     |         |                       | model         |
|                     |         |                       | weights and   |
|                     |         |                       | config        |
|                     |         |                       | files         |
+---------------------+---------+-----------------------+---------------+
| extraInit.aws       | boolean | ``true``              | Disables      |
| Ec2MetadataDisabled |         |                       | the use of    |
|                     |         |                       | the Amazon    |
|                     |         |                       | EC2           |
|                     |         |                       | instance      |
|                     |         |                       | metadata      |
|                     |         |                       | service       |
+---------------------+---------+-----------------------+---------------+
| extraPorts          | list    | ``[]``                | Additional    |
|                     |         |                       | ports         |
|                     |         |                       | configuration |
|                     |         |                       |               |
+---------------------+---------+-----------------------+---------------+
| gpuModels           | list    | ``["TYPE_GPU_USED"]`` | Type of gpu   |
|                     |         |                       | used          |
+---------------------+---------+-----------------------+---------------+
| image               | object  | ``{"command":         | Image         |
|                     |         | ["vllm","serve","     | configuration |
|                     |         | /data/","--served-mod |               |
|                     |         | el-name","opt-125m"," |               |
|                     |         | --host","0.0.0.0","-- |               |
|                     |         | port","8000"],"reposi |               |
|                     |         | tory":"vllm/vllm-open |               |
|                     |         | ai","tag":"latest"}`` |               |
+---------------------+---------+-----------------------+---------------+
| image.command       | list    | ``["vllm","se         | Container     |
|                     |         | rve","/data/","--serv | launch        |
|                     |         | ed-model-name","opt-1 | command       |
|                     |         | 25m","--host","0.0.0. |               |
|                     |         | 0","--port","8000"]`` |               |
+---------------------+---------+-----------------------+---------------+
| image.repository    | string  | `                     | Image         |
|                     |         | `"vllm/vllm-openai"`` | repository    |
+---------------------+---------+-----------------------+---------------+
| image.tag           | string  | ``"latest"``          | Image tag     |
+---------------------+---------+-----------------------+---------------+
| livenessProbe       | object  | ``{"fa                | Liveness      |
|                     |         | ilureThreshold":3,"ht | probe         |
|                     |         | tpGet":{"path":"/heal | configuration |
|                     |         | th","port":8000},"ini |               |
|                     |         | tialDelaySeconds":15, |               |
|                     |         | "periodSeconds":10}`` |               |
+---------------------+---------+-----------------------+---------------+
| livenessProbe.      | int     | ``3``                 | Number of     |
| failureThreshold    |         |                       | times after   |
|                     |         |                       | which if a    |
|                     |         |                       | probe fails   |
|                     |         |                       | in a row,     |
|                     |         |                       | Kubernetes    |
|                     |         |                       | considers     |
|                     |         |                       | that the      |
|                     |         |                       | overall       |
|                     |         |                       | check has     |
|                     |         |                       | failed: the   |
|                     |         |                       | container     |
|                     |         |                       | is not        |
|                     |         |                       | alive         |
+---------------------+---------+-----------------------+---------------+
| livenessProbe.      | object  | ``{"path":"/h         | Configuration |
| httpGet             |         | ealth","port":8000}`` | of the        |
|                     |         |                       | Kubelet       |
|                     |         |                       | http          |
|                     |         |                       | request on    |
|                     |         |                       | the server    |
+---------------------+---------+-----------------------+---------------+
| livenessProbe.      | string  | ``"/health"``         | Path to       |
| httpGet.path        |         |                       | access on     |
|                     |         |                       | the HTTP      |
|                     |         |                       | server        |
+---------------------+---------+-----------------------+---------------+
| livenessProbe.      | int     | ``8000``              | Name or       |
| httpGet.port        |         |                       | number of     |
|                     |         |                       | the port to   |
|                     |         |                       | access on     |
|                     |         |                       | the           |
|                     |         |                       | container,    |
|                     |         |                       | on which      |
|                     |         |                       | the server    |
|                     |         |                       | is            |
|                     |         |                       | listening     |
+---------------------+---------+-----------------------+---------------+
| livenessProbe.      | int     | ``15``                | Number of     |
| initialDelaySeconds |         |                       | seconds       |
|                     |         |                       | after the     |
|                     |         |                       | container     |
|                     |         |                       | has started   |
|                     |         |                       | before        |
|                     |         |                       | liveness      |
|                     |         |                       | probe is      |
|                     |         |                       | initiated     |
+---------------------+---------+-----------------------+---------------+
| livenessProbe.      | int     | ``10``                | How often     |
| periodSeconds       |         |                       | (in           |
|                     |         |                       | seconds) to   |
|                     |         |                       | perform the   |
|                     |         |                       | liveness      |
|                     |         |                       | probe         |
+---------------------+---------+-----------------------+---------------+
| maxUnavailable      | string  | ``""``                | Disruption    |
| PodDisruptionBudget |         |                       | Budget        |
|                     |         |                       | Configuration |
+---------------------+---------+-----------------------+---------------+
| readinessProbe      | object  | ``{"                  | Readiness     |
|                     |         | failureThreshold":3," | probe         |
|                     |         | httpGet":{"path":"/he | configuration |
|                     |         | alth","port":8000},"i |               |
|                     |         | nitialDelaySeconds":5 |               |
|                     |         | ,"periodSeconds":5}`` |               |
+---------------------+---------+-----------------------+---------------+
| readinessProbe.     | int     | ``3``                 | Number of     |
| failureThreshold    |         |                       | times after   |
|                     |         |                       | which if a    |
|                     |         |                       | probe fails   |
|                     |         |                       | in a row,     |
|                     |         |                       | Kubernetes    |
|                     |         |                       | considers     |
|                     |         |                       | that the      |
|                     |         |                       | overall       |
|                     |         |                       | check has     |
|                     |         |                       | failed: the   |
|                     |         |                       | container     |
|                     |         |                       | is not        |
|                     |         |                       | ready         |
+---------------------+---------+-----------------------+---------------+
| readinessProbe.     | object  | ``{"path":"/h         | Configuration |
| httpGet             |         | ealth","port":8000}`` | of the        |
|                     |         |                       | Kubelet       |
|                     |         |                       | http          |
|                     |         |                       | request on    |
|                     |         |                       | the server    |
+---------------------+---------+-----------------------+---------------+
| readinessProbe.     | string  | ``"/health"``         | Path to       |
| httpGet.path        |         |                       | access on     |
|                     |         |                       | the HTTP      |
|                     |         |                       | server        |
+---------------------+---------+-----------------------+---------------+
| readinessProbe.     | int     | ``8000``              | Name or       |
| httpGet.port        |         |                       | number of     |
|                     |         |                       | the port to   |
|                     |         |                       | access on     |
|                     |         |                       | the           |
|                     |         |                       | container,    |
|                     |         |                       | on which      |
|                     |         |                       | the server    |
|                     |         |                       | is            |
|                     |         |                       | listening     |
+---------------------+---------+-----------------------+---------------+
| readinessProbe.     | int     | ``5``                 | Number of     |
| initialDelaySeconds |         |                       | seconds       |
|                     |         |                       | after the     |
|                     |         |                       | container     |
|                     |         |                       | has started   |
|                     |         |                       | before        |
|                     |         |                       | readiness     |
|                     |         |                       | probe is      |
|                     |         |                       | initiated     |
+---------------------+---------+-----------------------+---------------+
| readinessProbe.     | int     | ``5``                 | How often     |
| periodSeconds       |         |                       | (in           |
|                     |         |                       | seconds) to   |
|                     |         |                       | perform the   |
|                     |         |                       | readiness     |
|                     |         |                       | probe         |
+---------------------+---------+-----------------------+---------------+
| replicaCount        | int     | ``1``                 | Number of     |
|                     |         |                       | replicas      |
+---------------------+---------+-----------------------+---------------+
| resources           | object  | ``{"limits            | Resource      |
|                     |         | ":{"cpu":4,"memory":" | configuration |
|                     |         | 16Gi","nvidia.com/gpu |               |
|                     |         | ":1},"requests":{"cpu |               |
|                     |         | ":4,"memory":"16Gi"," |               |
|                     |         | nvidia.com/gpu":1}}`` |               |
+---------------------+---------+-----------------------+---------------+
| resources.limits.   | int     | ``1``                 | Number of     |
| ”nvidia.com/gpu”    |         |                       | gpus used     |
+---------------------+---------+-----------------------+---------------+
| resources.          | int     | ``4``                 | Number of     |
| limits.cpu          |         |                       | CPUs          |
+---------------------+---------+-----------------------+---------------+
| resources.          | string  | ``"16Gi"``            | CPU memory    |
| limits.memory       |         |                       | configuration |
+---------------------+---------+-----------------------+---------------+
| resources.requests. | int     | ``1``                 | Number of     |
| ”nvidia.com/gpu”    |         |                       | gpus used     |
+---------------------+---------+-----------------------+---------------+
| resources.          | int     | ``4``                 | Number of     |
| requests.cpu        |         |                       | CPUs          |
+---------------------+---------+-----------------------+---------------+
| resources.          | string  | ``"16Gi"``            | CPU memory    |
| requests.memory     |         |                       | configuration |
+---------------------+---------+-----------------------+---------------+
| secrets             | object  | ``{}``                | Secrets       |
|                     |         |                       | configuration |
+---------------------+---------+-----------------------+---------------+
| serviceName         | string  |                       | Service       |
|                     |         |                       | name          |
+---------------------+---------+-----------------------+---------------+
| servicePort         | int     | ``80``                | Service       |
|                     |         |                       | port          |
+---------------------+---------+-----------------------+---------------+
| labels.environment  | string  | ``test``              | Environment   |
|                     |         |                       | name          |
+---------------------+---------+-----------------------+---------------+
| labels.release      | string  | ``test``              | Release       |
|                     |         |                       | name          |
+---------------------+---------+-----------------------+---------------+