import yaml
import click
from typing import List, Dict, Any, Optional
import os
from pydantic import BaseModel, Field

from plugin import DockerPluginConfig, KubernetesPodSpec, KubernetesPluginConfig, DOCKER_PLUGIN_NAME
from utils import AgentQueue, HF_HOME, DEFAULT_WORKING_DIR, VLLM_ECR_REPO, get_agent_queue, EXTERNAL_HARDWARE_TESTS, AMD_REPO
from step import Step, BuildkiteStep, BuildkiteBlockStep, get_block_step, get_step_key

def _get_image_path(repo: str) -> str:
    """Get path to image of the current commit on ECR."""
    commit = os.getenv("BUILDKITE_COMMIT")
    commit = "40184a07427f2a0b06094e98a9ad631e702225cd"
    return f"{repo}:{commit}"

def _convert_command(step: Step) -> List[str]:
    """Convert test commands into bash command for plugin."""
    step_commands = ""
    if len(step.commands) > 1:
        step_commands = " && ".join(step.commands)
    else:
        step_commands = step.commands[0]
    working_dir = step.working_dir or DEFAULT_WORKING_DIR
    bash_command = f"cd {working_dir} && {step_commands}"
    return ["bash", "-c", bash_command]

def _get_plugin_config(step: Step) -> Dict:
    """
    Returns the plugin configuration for the step.
    If the step is run on A100 GPU, use k8s plugin since A100 node is on k8s.
    Otherwise, use Docker plugin.
    """
    if step.gpu == "a100":
        pod_spec = KubernetesPodSpec(containers=[{"image": _get_image_path(VLLM_ECR_REPO), "command": _convert_command(step)}])
        return {"kubernetes": KubernetesPluginConfig(pod_spec=pod_spec).dict(by_alias=True)}
    docker_plugin_config = DockerPluginConfig(image=_get_image_path(VLLM_ECR_REPO), command=_convert_command(step))
    if step.no_gpu:
        docker_plugin_config.gpus = None
    return {DOCKER_PLUGIN_NAME: docker_plugin_config.dict(exclude_none=True, by_alias=True)}

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
        test_commands = []
        for command in step.commands:
            test_commands.append(f"'{command}'")
        all_commands = ["./.buildkite/run-multi-node-test.sh", str(step.working_dir or DEFAULT_WORKING_DIR), str(step.num_nodes), str(step.num_gpus), _get_image_path(VLLM_ECR_REPO)]
        all_commands.extend(test_commands)
        current_step.commands = " ".join(all_commands)
        current_step.plugins = None

    def step_should_run():
        """Determine whether the step should automatically run or not."""
        # Run if the step has no source file dependencies
        if not step.source_file_dependencies:
            return True
        # Run if RUN_ALL is set to 1
        if run_all == "1":
            return True
        # Run if any source file dependency is in the list of changed files patterns
        for source_file in step.source_file_dependencies:
            for diff_file in list_file_diff:
                if source_file in diff_file:
                    return True

    # If step does not have to run automatically, create a block step
    if not step_should_run():
        block_step = get_block_step(step.label)
        steps.append(block_step)
        current_step.depends_on = block_step.key
    steps.append(current_step)
    return steps

def generate_build_step() -> BuildkiteStep:
    """Build the Docker image and push it to ECR."""
    buildkite_commit = os.getenv("BUILDKITE_COMMIT")
    docker_image = _get_image_path(VLLM_ECR_REPO)
    command = ["aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin public.ecr.aws/q9t5s3a7", f"docker build --build-arg max_jobs=16 --build-arg buildkite_commit={buildkite_commit} --build-arg USE_SCCACHE=1 --tag {docker_image} --target test --progress plain .", f"docker push {docker_image}"]
    step = BuildkiteStep(label=":docker: build image", key="build", agents={"queue": AgentQueue.AWS_CPU.value}, env={"DOCKER_BUILDKIT": "1"}, retry={"automatic": [{"exit_status": -1, "limit": 2}, {"exit_status": -10, "limit": 2}]}, commands=command)
    step.depends_on = None
    return step

def mock_build_step() -> BuildkiteStep:
    docker_image = f"{VLLM_ECR_REPO}:40184a07427f2a0b06094e98a9ad631e702225cd"
    command = ["echo 'mock build step'"]
    step = BuildkiteStep(label=":docker: build image", key="build", agents={"queue": AgentQueue.AWS_CPU.value}, env={"DOCKER_BUILDKIT": "1"}, retry={"automatic": [{"exit_status": -1, "limit": 2}, {"exit_status": -10, "limit": 2}]}, commands=command)
    step.depends_on = None
    return step

def get_external_hardware_tests(test_steps: List[Step]) -> List[BuildkiteStep]:
    """Get the external hardware tests from the yaml file."""
    with open(EXTERNAL_HARDWARE_TESTS, "r") as f:
        content = yaml.safe_load(f)
    for step in content["steps"]:
        for command in step["commands"]:
            command = command.replace("DOCKER_IMAGE_AMD", _get_image_path(AMD_REPO))
    buildkite_steps = [BuildkiteStep(**step) for step in content["steps"]]
    # Fill in AMD Docker image path
    for step in buildkite_steps:
        step.commands = [cmd.replace("DOCKER_IMAGE_AMD", _get_image_path(AMD_REPO)) for cmd in step.commands]
    
    # Mirror test steps for AMD
    for test_step in test_steps:
        if "amd" in test_step.mirror_hardwares:
            test_command = _convert_command(test_step)[-1]
            amd_command = ["bash", ".buildkite/run-amd-test.sh", test_command]
            mirrored_buildkite_step = BuildkiteStep(
                label = f"AMD: {test_step.label}",
                key = f"amd_{get_step_key(test_step.label)}",
                depends_on = "amd-build",
                agents = {"queue": AgentQueue.AMD_GPU.value},
                soft_fail = test_step.soft_fail,
                env = {"DOCKER_BUILDKIT": "1"},
                commands = amd_command,
            )
            buildkite_steps.append(mirrored_buildkite_step)
    return buildkite_steps

@click.command()
@click.option("--run_all", type=str)
@click.option("--list_file_diff", type=str)
def main(run_all: str = -1, list_file_diff: str = None):
    list_file_diff = list_file_diff.split("|") if list_file_diff else []
    run_all = -1
    path = ".buildkite/test-pipeline.yaml"
    test_steps = read_test_steps(path)
    out_path = ".buildkite/pipeline.yaml"

    final_steps = [mock_build_step()]
    for step in test_steps:
        out_steps = process_step(step, run_all, list_file_diff)
        final_steps.extend(out_steps)
    
    # Add external hardware tests
    external_hardware_tests = get_external_hardware_tests(test_steps)
    final_steps.extend(external_hardware_tests)

    final = {"steps": [step.dict(exclude_none=True) for step in final_steps]}
    with open(out_path, "w") as f:
        yaml.dump(final, f, sort_keys=False)


if __name__ == "__main__":
    main()