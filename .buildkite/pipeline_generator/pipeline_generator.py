import yaml
import click
import enum
from typing import List, Dict, Any, Optional
import os
from pydantic import BaseModel, Field

HF_HOME = "/root/.cache/huggingface"
DEFAULT_WORKING_DIR = "/vllm-workspace/tests"
ECR_REPO = "public.ecr.aws/q9t5s3a7/vllm-ci-test-repo"

class AgentQueue(str, enum.Enum):
    AWS_CPU = "cpu_queue"
    AWS_SMALL_CPU = "small_cpu_queue"
    AWS_1xL4 = "gpu_1_queue"
    AWS_4xL4 = "gpu_4_queue"
    A100 = "a100-queue"

class Step(BaseModel):
    label: str
    fast_check: bool = False
    commands: List[str] = Field(default_factory=list)
    mirror_hardwares: List[str] = Field(default_factory=list)
    gpu: str = ""
    num_gpus: int = 1
    num_nodes: int = 1
    working_dir: str = "/vllm-workspace/tests"
    source_file_dependencies: List[str] = Field(default_factory=list)
    no_gpu: bool = False
    soft_fail: bool = False
    parallelism: int = 1
    optional: bool = False

class DockerPluginConfig(BaseModel):
    image: str = ""
    always_pull: bool = Field(default=True, alias="always-pull")
    propagate_environment: bool = Field(default=True, alias="propagate-environment")
    gpus: Optional[str] = "all"
    mount_buildkite_agent: Optional[bool] = Field(default=False, alias="mount-buildkite-agent")
    command: List[str] = Field(default_factory=list)
    environment: List[str] = Field(default_factory=list)
    volumes: List[str] = Field(default_factory=list)

class KubernetesPodSpec(BaseModel):
    containers: List[Dict[str, Any]]
    node_selector: Dict[str, Any] = Field(default_factory=dict)
    volumes: List[Dict[str, Any]] = Field(default_factory=list)

class KubernetesPluginConfig(BaseModel):
    pod_spec: KubernetesPodSpec

class BuildkiteStep(BaseModel):
    label: str
    key: str
    agents: Dict[str, Any] = {"queue": AgentQueue.AWS_CPU}
    commands: Optional[List[str]] = None
    plugins: Optional[List[Dict]] = None
    parallelism: Optional[int] = None
    soft_fail: Optional[bool] = None
    depends_on: Optional[str] = "build"
    env: Optional[Dict[str, str]] = None
    retry: Optional[Dict[str, Any]] = None

class BuildkiteBlockStep(BaseModel):
    block: str
    depends_on: Optional[str] = "build"
    key: str

def _get_image_path() -> str:
    commit = os.getenv("BUILDKITE_COMMIT")
    commit = "40184a07427f2a0b06094e98a9ad631e702225cd"
    return f"{ECR_REPO}:{commit}"

def _get_step_key(label: str) -> str:
    return label.replace(" ", "-").lower().replace("(", "").replace(")", "").replace("%", "").replace(",", "-")

def _get_agent_queue(step: Step) -> AgentQueue:
    if step.no_gpu:
        return AgentQueue.AWS_SMALL_CPU
    if step.gpu == "a100":
        return AgentQueue.A100
    if step.num_gpus == 1:
        return AgentQueue.AWS_1xL4
    return AgentQueue.AWS_4xL4

def _get_block_step(step: Step) -> BuildkiteBlockStep:
    return BuildkiteBlockStep(block=f"Run {step.label}", key=f"block-{_get_step_key(step.label)}")

def _convert_command(step: Step) -> List[str]:
    commands = ""
    if len(step.commands) > 1:
        commands = " && ".join(step.commands)
    else:
        commands = step.commands[0]
    working_dir = step.working_dir or DEFAULT_WORKING_DIR
    final_command = f"cd {working_dir} && {commands}"
    return ["bash", "-c", final_command]

def _get_plugin_config(step: Step) -> Dict:
    if step.gpu == "a100":
        pod_spec = KubernetesPodSpec(containers=[{"image": _get_image_path(), "command": _convert_command(step)}])
        return {"kubernetes": KubernetesPluginConfig(pod_spec=pod_spec).dict(by_alias=True)}
    
    docker_plugin_config = DockerPluginConfig(image=_get_image_path(), command=_convert_command(step))
    docker_plugin_config.environment = [f"HF_HOME={HF_HOME}", "VLLM_USAGE_SOURCE=ci-test", "HF_TOKEN", "BUILDKITE_ANALYTICS_TOKEN"]
    docker_plugin_config.volumes = ["/dev/shm:/dev/shm", f"{HF_HOME}:{HF_HOME}"]
    if step.no_gpu:
        docker_plugin_config.gpus = None
    docker_plugin_config_dict = docker_plugin_config.dict(exclude_none=True, by_alias=True)
    return {"docker#v5.2.0": docker_plugin_config_dict}

def read_test_steps(file_path: str) -> List[Step]:
    with open(file_path, "r") as f:
        content = yaml.safe_load(f)
    return [Step(**step) for step in content["steps"]]

def process_step(step: Step, run_all: str, list_file_diff: List[str]):
    steps = []
    current_step = BuildkiteStep(label=step.label, key=_get_step_key(step.label), parallelism=step.parallelism, soft_fail=step.soft_fail, plugins=[_get_plugin_config(step)])
    current_step.agents["queue"] = _get_agent_queue(step).value
    if step.num_nodes > 1:
        test_commands = []
        for command in step.commands:
            test_commands.append(f"'{command}'")
        all_commands = ["./.buildkite/run-multi-node-test.sh", str(step.working_dir or DEFAULT_WORKING_DIR), str(step.num_nodes), str(step.num_gpus), _get_image_path()]
        all_commands.extend(test_commands)
        current_step.commands = " ".join(all_commands)
        current_step.plugins = None
    def step_should_run():
        if not step.source_file_dependencies:
            return True
        if run_all == "1":
            return True
        for source_file in step.source_file_dependencies:
            for diff_file in list_file_diff:
                if source_file in diff_file:
                    return True

    if not step_should_run():
        block_step = _get_block_step(step)
        steps.append(block_step)
        current_step.depends_on = block_step.key
    steps.append(current_step)
    return steps

def build_step() -> BuildkiteStep:
    buildkite_commit = os.getenv("BUILDKITE_COMMIT")
    docker_image = _get_image_path()
    command = ["aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin public.ecr.aws/q9t5s3a7", f"docker build --build-arg max_jobs=16 --build-arg buildkite_commit={buildkite_commit} --build-arg USE_SCCACHE=1 --tag {docker_image} --target test --progress plain .", f"docker push {docker_image}"]
    step = BuildkiteStep(label=":docker: build image", key="build", agents={"queue": AgentQueue.AWS_CPU.value}, env={"DOCKER_BUILDKIT": "1"}, retry={"automatic": [{"exit_status": -1, "limit": 2}, {"exit_status": -10, "limit": 2}]}, commands=command)
    step.depends_on = None
    return step

def mock_build_step() -> BuildkiteStep:
    docker_image = f"{ECR_REPO}:40184a07427f2a0b06094e98a9ad631e702225cd"
    command = ["echo 'mock build step'"]
    step = BuildkiteStep(label=":docker: build image", key="build", agents={"queue": AgentQueue.AWS_CPU.value}, env={"DOCKER_BUILDKIT": "1"}, retry={"automatic": [{"exit_status": -1, "limit": 2}, {"exit_status": -10, "limit": 2}]}, commands=command)
    step.depends_on = None
    return step

@click.command()
@click.option("--run_all", type=str)
@click.option("--list_file_diff", type=str)
def main(run_all: str = -1, list_file_diff: str = None):
    list_file_diff = list_file_diff.split("|") if list_file_diff else []
    run_all = -1
    #yaml_obj = ruamel.yaml.YAML()
    path = ".buildkite/test-pipeline.yaml"
    test_steps = read_test_steps(path)
    out_path = ".buildkite/pipeline.yaml"

    final_steps = [mock_build_step()]
    for step in test_steps:
        out_steps = process_step(step, run_all, list_file_diff)
        final_steps.extend(out_steps)

    final = {"steps": [step.dict(exclude_none=True) for step in final_steps]}
    with open(out_path, "w") as f:
        yaml.dump(final, f, sort_keys=False)


if __name__ == "__main__":
    main()