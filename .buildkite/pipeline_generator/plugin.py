from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from utils import HF_HOME   

DOCKER_PLUGIN_NAME = "docker#v5.2.0"
KUBERNETES_PLUGIN_NAME = "kubernetes"

class DockerPluginConfig(BaseModel):
    image: str = ""
    always_pull: bool = Field(default=True, alias="always-pull")
    propagate_environment: bool = Field(default=True, alias="propagate-environment")
    gpus: Optional[str] = "all"
    mount_buildkite_agent: Optional[bool] = Field(default=False, alias="mount-buildkite-agent")
    command: List[str] = Field(default_factory=list)
    environment: List[str] = [f"HF_HOME={HF_HOME}", "VLLM_USAGE_SOURCE=ci-test", "HF_TOKEN", "BUILDKITE_ANALYTICS_TOKEN"]
    volumes: List[str] = ["/dev/shm:/dev/shm", f"{HF_HOME}:{HF_HOME}"]

class KubernetesPodSpec(BaseModel):
    containers: List[Dict[str, Any]]
    node_selector: Dict[str, Any] = Field(default={"nvidia.com/gpu.product": "NVIDIA-A100-SXM4-80GB"}, alias="nodeSelector")
    volumes: List[Dict[str, Any]] = Field(default=[{"name": "devshm", "emptyDir": {"medium": "Memory"}}, {"name": "hf-cache", "hostPath": {"path": HF_HOME, "type": "Directory"}}])

class KubernetesPluginConfig(BaseModel):
    pod_spec: KubernetesPodSpec = Field(alias="podSpec")

def get_kubernetes_plugin_config(docker_image_path: str, test_bash_command: List[str], num_gpus: int) -> Dict:
    pod_spec = KubernetesPodSpec(
        containers=[{
            "image": docker_image_path, 
            "command": test_bash_command
            "resources": {
                "limits": {
                    "nvidia.com/gpu": num_gpus
                }
            },
            "volumeMounts": [
                {"name": "devshm", "mountPath": "/dev/shm"},
                {"name": "hf-cache", "mountPath": HF_HOME}
            ],
            "env": [
                {"name": "HF_HOME", "value": HF_HOME},
                {"name": "VLLM_USAGE_SOURCE", "value": "ci-test"},
                {
                    "name": "HF_TOKEN", 
                    "valueFrom": {
                        "secretKeyRef": {
                            "name": "huggingface-token",
                            "key": "token"
                        }
                    }
                },
            ]
        }]
    )
    return {KUBERNETES_PLUGIN_NAME: KubernetesPluginConfig(pod_spec=pod_spec).dict(by_alias=True)}

def get_docker_plugin_config(docker_image_path: str, test_bash_command: List[str], no_gpu: bool) -> Dict:
    docker_plugin_config = DockerPluginConfig(
        image=docker_image_path, 
        command=test_bash_command
    )
    if no_gpu:
        docker_plugin_config.gpus = None
    return {DOCKER_PLUGIN_NAME: docker_plugin_config.dict(exclude_none=True, by_alias=True)}
