.. _deploying_with_helm:

Deploying with Helm
===================

A Helm chart to deploy vLLM for Kubernetes

Helm is a package manager for Kubernetes. It will help you to deploy vLLM on k8s and automate the deployment of vLLMm Kubernetes applications. With Helm, you can deploy the same framework architecture to multiple namespaces by overriding variables values.

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

+---------------------+---------+-----------------------+-------------+
| Key                 | Type    | Default               | Description |
+=====================+=========+=======================+=============+
| autoscaling         | object  | ``{"ena               | Autoscaling |
|                     |         | bled":false,"maxRepli | co          |
|                     |         | cas":100,"minReplicas | nfiguration |
|                     |         | ":1,"targetCPUUtiliza |             |
|                     |         | tionPercentage":80}`` |             |
+---------------------+---------+-----------------------+-------------+
| autoscaling.enabled | bool    | ``false``             | Enable      |
|                     |         |                       | autoscaling |
+---------------------+---------+-----------------------+-------------+
| auto                | int     | ``100``               | Maximum     |
| scaling.maxReplicas |         |                       | replicas    |
+---------------------+---------+-----------------------+-------------+
| auto                | int     | ``1``                 | Minimum     |
| scaling.minReplicas |         |                       | replicas    |
+---------------------+---------+-----------------------+-------------+
| auto                | int     | ``80``                | Target CPU  |
| scaling.targetCPUUt |         |                       | utilization |
| ilizationPercentage |         |                       | for         |
|                     |         |                       | autoscaling |
+---------------------+---------+-----------------------+-------------+
| configs             | object  | ``{}``                | Configmap   |
+---------------------+---------+-----------------------+-------------+
| containerPort       | int     | ``8000``              | Container   |
|                     |         |                       | port        |
+---------------------+---------+-----------------------+-------------+
| customObjects       | list    | ``[]``                | Custom      |
|                     |         |                       | Objects     |
|                     |         |                       | co          |
|                     |         |                       | nfiguration |
+---------------------+---------+-----------------------+-------------+
| deploymentStrategy  | object  | ``{}``                | Deployment  |
|                     |         |                       | strategy    |
|                     |         |                       | co          |
|                     |         |                       | nfiguration |
+---------------------+---------+-----------------------+-------------+
| externalConfigs     | list    | ``[]``                | External    |
|                     |         |                       | co          |
|                     |         |                       | nfiguration |
+---------------------+---------+-----------------------+-------------+
| extraContainers     | list    | ``[]``                | Additional  |
|                     |         |                       | containers  |
|                     |         |                       | co          |
|                     |         |                       | nfiguration |
+---------------------+---------+-----------------------+-------------+
| extraInit           | object  | ``{"pvcStorage":"1Gi  | Additional  |
|                     |         | ","s3modelpath":"rela | co          |
|                     |         | tive_s3_model_path/op | nfiguration |
|                     |         | t-125m", "awsEc2Metad | for the     |
|                     |         | ataDisabled": true}`` | init        |
|                     |         |                       | container   |
+---------------------+---------+-----------------------+-------------+
| e                   | string  | ``"50Gi"``            | Storage     |
| xtraInit.pvcStorage |         |                       | size of the |
|                     |         |                       | s3          |
+---------------------+---------+-----------------------+-------------+
| ex                  | string  | ``"relative_s3_m      | Path of the |
| traInit.s3modelpath |         | odel_path/opt-125m"`` | model on    |
|                     |         |                       | the s3      |
|                     |         |                       | which hosts |
|                     |         |                       | model       |
|                     |         |                       | weights and |
|                     |         |                       | config      |
|                     |         |                       | files       |
+---------------------+---------+-----------------------+-------------+
| extraInit.aws       | boolean | ``true``              | Disables    |
| Ec2MetadataDisabled |         |                       | the use of  |
|                     |         |                       | the Amazon  |
|                     |         |                       | EC2         |
|                     |         |                       | instance    |
|                     |         |                       | metadata    |
|                     |         |                       | service     |
+---------------------+---------+-----------------------+-------------+
| extraPorts          | list    | ``[]``                | Additional  |
|                     |         |                       | ports       |
|                     |         |                       | co          |
|                     |         |                       | nfiguration |
+---------------------+---------+-----------------------+-------------+
| gpuModels           | list    | ``["TYPE_GPU_USED"]`` | Type of gpu |
|                     |         |                       | used        |
+---------------------+---------+-----------------------+-------------+
| image               | object  | ``{"command":         | Image       |
|                     |         | ["vllm","serve","     | co          |
|                     |         | /data/","--served-mod | nfiguration |
|                     |         | el-name","opt-125m"," |             |
|                     |         | --host","0.0.0.0","-- |             |
|                     |         | port","8000"],"reposi |             |
|                     |         | tory":"vllm/vllm-open |             |
|                     |         | ai","tag":"latest"}`` |             |
+---------------------+---------+-----------------------+-------------+
| image.command       | list    | ``["vllm","se         | Container   |
|                     |         | rve","/data/","--serv | launch      |
|                     |         | ed-model-name","opt-1 | command     |
|                     |         | 25m","--host","0.0.0. |             |
|                     |         | 0","--port","8000"]`` |             |
+---------------------+---------+-----------------------+-------------+
| image.repository    | string  | `                     | Image       |
|                     |         | `"vllm/vllm-openai"`` | repository  |
+---------------------+---------+-----------------------+-------------+
| image.tag           | string  | ``"latest"``          | Image tag   |
+---------------------+---------+-----------------------+-------------+
| livenessProbe       | object  | ``{"fa                | Liveness    |
|                     |         | ilureThreshold":3,"ht | probe       |
|                     |         | tpGet":{"path":"/heal | co          |
|                     |         | th","port":8000},"ini | nfiguration |
|                     |         | tialDelaySeconds":15, |             |
|                     |         | "periodSeconds":10}`` |             |
+---------------------+---------+-----------------------+-------------+
| livenessPro         | int     | ``3``                 | Number of   |
| be.failureThreshold |         |                       | times after |
|                     |         |                       | which if a  |
|                     |         |                       | probe fails |
|                     |         |                       | in a row,   |
|                     |         |                       | Kubernetes  |
|                     |         |                       | considers   |
|                     |         |                       | that the    |
|                     |         |                       | overall     |
|                     |         |                       | check has   |
|                     |         |                       | failed: the |
|                     |         |                       | container   |
|                     |         |                       | is not      |
|                     |         |                       | alive       |
+---------------------+---------+-----------------------+-------------+
| li                  | object  | ``{"path":"/h         | Co          |
| venessProbe.httpGet |         | ealth","port":8000}`` | nfiguration |
|                     |         |                       | of the      |
|                     |         |                       | Kubelet     |
|                     |         |                       | http        |
|                     |         |                       | request on  |
|                     |         |                       | the server  |
+---------------------+---------+-----------------------+-------------+
| livenes             | string  | ``"/health"``         | Path to     |
| sProbe.httpGet.path |         |                       | access on   |
|                     |         |                       | the HTTP    |
|                     |         |                       | server      |
+---------------------+---------+-----------------------+-------------+
| livenes             | int     | ``8000``              | Name or     |
| sProbe.httpGet.port |         |                       | number of   |
|                     |         |                       | the port to |
|                     |         |                       | access on   |
|                     |         |                       | the         |
|                     |         |                       | container,  |
|                     |         |                       | on which    |
|                     |         |                       | the server  |
|                     |         |                       | is          |
|                     |         |                       | listening   |
+---------------------+---------+-----------------------+-------------+
| livenessProbe.      | int     | ``15``                | Number of   |
| initialDelaySeconds |         |                       | seconds     |
|                     |         |                       | after the   |
|                     |         |                       | container   |
|                     |         |                       | has started |
|                     |         |                       | before      |
|                     |         |                       | liveness    |
|                     |         |                       | probe is    |
|                     |         |                       | initiated   |
+---------------------+---------+-----------------------+-------------+
| liveness            | int     | ``10``                | How often   |
| Probe.periodSeconds |         |                       | (in         |
|                     |         |                       | seconds) to |
|                     |         |                       | perform the |
|                     |         |                       | liveness    |
|                     |         |                       | probe       |
+---------------------+---------+-----------------------+-------------+
| maxUnavailable      | string  | ``""``                | Disruption  |
| PodDisruptionBudget |         |                       | Budget      |
|                     |         |                       | Co          |
|                     |         |                       | nfiguration |
+---------------------+---------+-----------------------+-------------+
| readinessProbe      | object  | ``{"                  | Readiness   |
|                     |         | failureThreshold":3," | probe       |
|                     |         | httpGet":{"path":"/he | co          |
|                     |         | alth","port":8000},"i | nfiguration |
|                     |         | nitialDelaySeconds":5 |             |
|                     |         | ,"periodSeconds":5}`` |             |
+---------------------+---------+-----------------------+-------------+
| readinessPro        | int     | ``3``                 | Number of   |
| be.failureThreshold |         |                       | times after |
|                     |         |                       | which if a  |
|                     |         |                       | probe fails |
|                     |         |                       | in a row,   |
|                     |         |                       | Kubernetes  |
|                     |         |                       | considers   |
|                     |         |                       | that the    |
|                     |         |                       | overall     |
|                     |         |                       | check has   |
|                     |         |                       | failed: the |
|                     |         |                       | container   |
|                     |         |                       | is not      |
|                     |         |                       | ready       |
+---------------------+---------+-----------------------+-------------+
| rea                 | object  | ``{"path":"/h         | Co          |
| dinessProbe.httpGet |         | ealth","port":8000}`` | nfiguration |
|                     |         |                       | of the      |
|                     |         |                       | Kubelet     |
|                     |         |                       | http        |
|                     |         |                       | request on  |
|                     |         |                       | the server  |
+---------------------+---------+-----------------------+-------------+
| readines            | string  | ``"/health"``         | Path to     |
| sProbe.httpGet.path |         |                       | access on   |
|                     |         |                       | the HTTP    |
|                     |         |                       | server      |
+---------------------+---------+-----------------------+-------------+
| readines            | int     | ``8000``              | Name or     |
| sProbe.httpGet.port |         |                       | number of   |
|                     |         |                       | the port to |
|                     |         |                       | access on   |
|                     |         |                       | the         |
|                     |         |                       | container,  |
|                     |         |                       | on which    |
|                     |         |                       | the server  |
|                     |         |                       | is          |
|                     |         |                       | listening   |
+---------------------+---------+-----------------------+-------------+
| readinessProbe.     | int     | ``5``                 | Number of   |
| initialDelaySeconds |         |                       | seconds     |
|                     |         |                       | after the   |
|                     |         |                       | container   |
|                     |         |                       | has started |
|                     |         |                       | before      |
|                     |         |                       | readiness   |
|                     |         |                       | probe is    |
|                     |         |                       | initiated   |
+---------------------+---------+-----------------------+-------------+
| readiness           | int     | ``5``                 | How often   |
| Probe.periodSeconds |         |                       | (in         |
|                     |         |                       | seconds) to |
|                     |         |                       | perform the |
|                     |         |                       | readiness   |
|                     |         |                       | probe       |
+---------------------+---------+-----------------------+-------------+
| replicaCount        | int     | ``1``                 | Number of   |
|                     |         |                       | replicas    |
+---------------------+---------+-----------------------+-------------+
| resources           | object  | ``{"limits            | Resource    |
|                     |         | ":{"cpu":4,"memory":" | co          |
|                     |         | 16Gi","nvidia.com/gpu | nfiguration |
|                     |         | ":1},"requests":{"cpu |             |
|                     |         | ":4,"memory":"16Gi"," |             |
|                     |         | nvidia.com/gpu":1}}`` |             |
+---------------------+---------+-----------------------+-------------+
| resources.limi      | int     | ``1``                 | Number of   |
| ts.”nvidia.com/gpu” |         |                       | gpus used   |
+---------------------+---------+-----------------------+-------------+
| r                   | int     | ``4``                 | Number of   |
| esources.limits.cpu |         |                       | CPUs        |
+---------------------+---------+-----------------------+-------------+
| reso                | string  | ``"16Gi"``            | CPU memory  |
| urces.limits.memory |         |                       | co          |
|                     |         |                       | nfiguration |
+---------------------+---------+-----------------------+-------------+
| resources.requests. | int     | ``1``                 | Number of   |
| ”nvidia.com/gpu”    |         |                       | gpus used   |
+---------------------+---------+-----------------------+-------------+
| resources.          | int     | ``4``                 | Number of   |
| requests.cpu        |         |                       | CPUs        |
+---------------------+---------+-----------------------+-------------+
| resources.          | string  | ``"16Gi"``            | CPU memory  |
| requests.memory     |         |                       | co          |
|                     |         |                       | nfiguration |
+---------------------+---------+-----------------------+-------------+
| secrets             | object  | ``{}``                | Secrets     |
|                     |         |                       | co          |
|                     |         |                       | nfiguration |
+---------------------+---------+-----------------------+-------------+
| serviceName         | string  |                       | Service     |
|                     |         |                       | name        |
+---------------------+---------+-----------------------+-------------+
| servicePort         | int     | ``80``                | Service     |
|                     |         |                       | port        |
+---------------------+---------+-----------------------+-------------+
| labels.environment  | string  | ``test``              | Environment |
|                     |         |                       | name        |
+---------------------+---------+-----------------------+-------------+
| labels.release      | string  | ``test``              | Release     |
|                     |         |                       | name        |
+---------------------+---------+-----------------------+-------------+