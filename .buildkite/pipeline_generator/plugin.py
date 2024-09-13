from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from utils import HF_HOME   

DOCKER_PLUGIN_NAME = "docker#v5.2.0"

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
    node_selector: Dict[str, Any] = Field(default_factory=dict)
    volumes: List[Dict[str, Any]] = Field(default_factory=list)

class KubernetesPluginConfig(BaseModel):
    pod_spec: KubernetesPodSpec