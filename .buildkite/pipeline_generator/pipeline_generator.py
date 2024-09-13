import yaml
import click
from typing import List, Dict, Any, Optional, Union
import os
from pydantic import BaseModel, Field

from plugin import (
    DockerPluginConfig, 
    KubernetesPodSpec, 
    KubernetesPluginConfig, 
    DOCKER_PLUGIN_NAME, 
    KUBERNETES_PLUGIN_NAME,
    get_kubernetes_plugin_config,
    get_docker_plugin_config,
)
from utils import (
    AgentQueue,
    HF_HOME,
    DEFAULT_WORKING_DIR,
    VLLM_ECR_REPO,
    AMD_REPO,
    TEST_PATH,
    EXTERNAL_HARDWARE_TEST_PATH,
    PIPELINE_FILE_PATH,
    STEPS_TO_BLOCK,
    get_agent_queue,
    get_full_test_command,
    get_image_path,
    get_multi_node_test_command,
)
from step import Step, BuildkiteStep, BuildkiteBlockStep, get_block_step, get_step_key


def _get_plugin_config(step: Step) -> Dict:
    """
    Returns the plugin configuration for the step.
    If the step is run on A100 GPU, use k8s plugin since A100 node is on k8s.
    Otherwise, use Docker plugin.
    """
    test_bash_command = ["bash", "-c", get_full_test_command(step.commands, step.working_dir)]
    docker_image_path = get_image_path()

    if step.gpu == "a100":
        return get_kubernetes_plugin_config(docker_image_path, test_bash_command)
    return get_docker_plugin_config(docker_image_path, test_bash_command, step.no_gpu)

def read_test_steps(file_path: str) -> List[Step]:
    """Read test steps from test pipeline yaml and parse them into Step objects."""
    with open(file_path, "r") as f:
        content = yaml.safe_load(f)
    return [Step(**step) for step in content["steps"]]

def process_step(step: Step, run_all: str, list_file_diff: List[str]):
    """Process test step and return corresponding BuildkiteStep."""
    steps = []
    current_step = BuildkiteStep(label=step.label, key=get_step_key(step.label), parallelism=step.parallelism, soft_fail=step.soft_fail, plugins=[_get_plugin_config(step)])
    current_step.agents["queue"] = get_agent_queue(step.no_gpu, step.gpu, step.num_gpus).value

    # If the step is a multi-node test, run it with a different script instead of Docker plugin
    if step.num_nodes > 1:
        current_step.commands = get_multi_node_test_command(
            step.commands, 
            step.working_dir, 
            step.num_nodes, 
            step.num_gpus, 
            get_image_path()
        )
        current_step.plugins = None

    def step_should_run():
        """Determine whether the step should automatically run or not."""
        if not step.source_file_dependencies or run_all == "1":
            return True
        return any(source_file in diff_file 
            for source_file in step.source_file_dependencies 
            for diff_file in list_file_diff)

    # If step does not have to run automatically, create a block step
    # and have the current step depend on it
    if not step_should_run():
        block_step = get_block_step(step.label)
        steps.append(block_step)
        current_step.depends_on = block_step.key
    steps.append(current_step)
    return steps

def generate_build_step() -> BuildkiteStep:
    """Build the Docker image and push it to ECR."""
    buildkite_commit = os.getenv("BUILDKITE_COMMIT")
    docker_image = get_image_path()
    build_commands = [
        "aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin public.ecr.aws/q9t5s3a7",
        f"docker build --build-arg max_jobs=16 --build-arg buildkite_commit={buildkite_commit} --build-arg USE_SCCACHE=1 --tag {docker_image} --target test --progress plain .", 
        f"docker push {docker_image}"
    ]
    step = BuildkiteStep(
        label=":docker: build image", 
        key="build", 
        agents={"queue": AgentQueue.AWS_CPU.value}, 
        env={"DOCKER_BUILDKIT": "1"}, 
        retry={"automatic": [{"exit_status": -1, "limit": 2}, {"exit_status": -10, "limit": 2}]}, 
        commands=build_commands,
        depends_on=None,
    )
    return step

def mock_build_step() -> BuildkiteStep:
    docker_image = f"{VLLM_ECR_REPO}:40184a07427f2a0b06094e98a9ad631e702225cd"
    command = ["echo 'mock build step'"]
    step = BuildkiteStep(label=":docker: build image", key="build", agents={"queue": AgentQueue.AWS_CPU.value}, env={"DOCKER_BUILDKIT": "1"}, retry={"automatic": [{"exit_status": -1, "limit": 2}, {"exit_status": -10, "limit": 2}]}, commands=command)
    step.depends_on = None
    return step

def get_external_hardware_tests(test_steps: List[Step]) -> List[Union[BuildkiteStep, BuildkiteBlockStep]]:
    """Process the external hardware tests from the yaml file and convert to Buildkite steps."""
    with open(EXTERNAL_HARDWARE_TEST_PATH, "r") as f:
        content = yaml.safe_load(f)
    buildkite_steps = []
    for step in content["steps"]:
        # Fill in AMD Docker image path
        step["commands"] = [cmd.replace("DOCKER_IMAGE_AMD", get_image_path(AMD_REPO)) for cmd in step["commands"]]

        # Convert step to BuildkiteStep object
        buildkite_step = BuildkiteStep(**step)

        # Add block step if step in block list
        # and have the current step depend on it
        if buildkite_step.key in STEPS_TO_BLOCK:
            block_step = get_block_step(buildkite_step.label)
            buildkite_steps.append(block_step)
            buildkite_step.depends_on = block_step.key

        buildkite_steps.append(buildkite_step)

    # Mirror test steps for AMD
    for test_step in test_steps:
        if "amd" in test_step.mirror_hardwares:
            amd_test_command = [
                "bash", 
                ".buildkite/run-amd-test.sh", 
                get_full_test_command(test_step.commands, test_step.working_dir)
            ]
            mirrored_buildkite_step = BuildkiteStep(
                label = f"AMD: {test_step.label}",
                key = f"amd_{get_step_key(test_step.label)}",
                depends_on = "amd-build",
                agents = {"queue": AgentQueue.AMD_GPU.value},
                soft_fail = test_step.soft_fail,
                env = {"DOCKER_BUILDKIT": "1"},
                commands = amd_test_command,
            )
            buildkite_steps.append(mirrored_buildkite_step)
    return buildkite_steps

@click.command()
@click.option("--run_all", type=str)
@click.option("--list_file_diff", type=str)
def main(run_all: str = -1, list_file_diff: str = None):
    list_file_diff = list_file_diff.split("|") if list_file_diff else []

    # Read test from yaml file and convert to Buildkite format steps
    test_steps = read_test_steps(TEST_PATH)
    buildkite_steps = [mock_build_step()]
    for step in test_steps:
        buildkite_test_step = process_step(step, run_all, list_file_diff)
        buildkite_steps.extend(buildkite_test_step)
    
    # Add external hardware tests
    external_hardware_tests = get_external_hardware_tests(test_steps)
    buildkite_steps.extend(external_hardware_tests)

    buildkite_steps_dict = {"steps": [step.dict(exclude_none=True) for step in buildkite_steps]}
    with open(PIPELINE_FILE_PATH, "w") as f:
        yaml.dump(buildkite_steps_dict, f, sort_keys=False)


if __name__ == "__main__":
    main()