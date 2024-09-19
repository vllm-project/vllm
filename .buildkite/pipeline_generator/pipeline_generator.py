import yaml
import click
from typing import List, Dict, Union
import os

from plugin import (
    get_kubernetes_plugin_config,
    get_docker_plugin_config,
)
from utils import (
    AgentQueue,
    AMD_REPO,
    A100_GPU,
    TEST_PATH,
    EXTERNAL_HARDWARE_TEST_PATH,
    PIPELINE_FILE_PATH,
    STEPS_TO_BLOCK,
    VLLM_ECR_URL,
    VLLM_ECR_REPO,
    get_agent_queue,
    get_full_test_command,
    get_image_path,
    get_multi_node_test_command,
)
from step import (
    TestStep, 
    BuildkiteStep, 
    BuildkiteBlockStep, 
    get_block_step, 
    get_step_key
)

class PipelineGenerator:
    def __init__(self, run_all: bool, list_file_diff: List[str]):
        self.run_all = run_all
        self.list_file_diff = list_file_diff
        self.commit = os.getenv("BUILDKITE_COMMIT")
    
    def read_test_steps(self, file_path: str) -> List[TestStep]:
        """Read test steps from test pipeline yaml and parse them into Step objects."""
        with open(file_path, "r") as f:
            content = yaml.safe_load(f)
        return [TestStep(**step) for step in content["steps"]]
    
    def step_should_run(self, step: TestStep) -> bool:
        """Determine whether the step should automatically run or not."""
        if step.optional:
            return False

        # If run_all is set to True, run all steps
        # If step does not specify source file dependencies, run it
        if not step.source_file_dependencies or self.run_all:
            return True
        
        # If any source file dependency is in the list of file differences, run it
        return any(source_file in diff_file 
            for source_file in step.source_file_dependencies 
            for diff_file in self.list_file_diff)

    def process_step(self, step: TestStep) -> List[Union[BuildkiteStep, BuildkiteBlockStep]]:
        """Process test step and return corresponding BuildkiteStep."""
        steps = []
        current_step = BuildkiteStep(
            label=step.label, 
            key=get_step_key(step.label), 
            parallelism=step.parallelism, 
            soft_fail=step.soft_fail, 
            plugins=[self._get_plugin_config(step)]
        )
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

        # If step does not have to run automatically, create a block step
        # and have the current step depend on it
        if not self.step_should_run(step):
            block_step = get_block_step(step.label)
            steps.append(block_step)
            current_step.depends_on = block_step.key
        steps.append(current_step)
        return steps
    
    def generate_build_step(self) -> BuildkiteStep:
        """Build the Docker image and push it to ECR."""
        docker_image = f"{VLLM_ECR_REPO}:{self.commit}"

        ecr_login_command = (
            "aws ecr-public get-login-password --region us-east-1 | "
            f"docker login --username AWS --password-stdin {VLLM_ECR_URL}"
        )
        docker_build_command = (
            f"docker build "
            f"--build-arg max_jobs=64 "
            f"--build-arg buildkite_commit={self.commit} "
            f"--build-arg USE_SCCACHE=1 "
            f"--tag {docker_image} "
            f"--target test "
            f"--progress plain ."
        )
        docker_push_command = f"docker push {docker_image}"
        build_commands = [
            ecr_login_command,
            docker_build_command,
            docker_push_command,
        ]

        step = BuildkiteStep(
            label=":docker: build image", 
            key="build", 
            agents={"queue": AgentQueue.AWS_CPU.value}, 
            env={"DOCKER_BUILDKIT": "1"}, 
            retry={
                "automatic": [
                    {"exit_status": -1, "limit": 2}, 
                    {"exit_status": -10, "limit": 2}
                ]
            }, 
            commands=build_commands,
            depends_on=None,
        )
        return step

    def get_external_hardware_tests(self, test_steps: List[TestStep]) -> List[Union[BuildkiteStep, BuildkiteBlockStep]]:
        """Process the external hardware tests from the yaml file and convert to Buildkite steps."""
        with open(EXTERNAL_HARDWARE_TEST_PATH, "r") as f:
            content = yaml.safe_load(f)
        buildkite_steps = []
        amd_docker_image = f"{AMD_REPO}:{self.commit}"
        for step in content["steps"]:
            # Fill in AMD Docker image path
            step["commands"] = [cmd.replace("DOCKER_IMAGE_AMD", amd_docker_image) for cmd in step["commands"]]

            # Convert step to BuildkiteStep object
            buildkite_step = BuildkiteStep(**step)
            buildkite_step.depends_on = "bootstrap"

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
                    f"'{get_full_test_command(test_step.commands, test_step.working_dir)}'",
                ]
                mirrored_buildkite_step = BuildkiteStep(
                    label = f"AMD: {test_step.label}",
                    key = f"amd_{get_step_key(test_step.label)}",
                    depends_on = "amd-build",
                    agents = {"queue": AgentQueue.AMD_GPU.value},
                    soft_fail = test_step.soft_fail,
                    env = {"DOCKER_BUILDKIT": "1"},
                    commands = [" ".join(amd_test_command)],
                )
                buildkite_steps.append(mirrored_buildkite_step)
        return buildkite_steps

    def _get_plugin_config(self, step: TestStep) -> Dict:
        """
        Returns the plugin configuration for the step.
        If the step is run on A100 GPU, use k8s plugin since A100 node is on k8s.
        Otherwise, use Docker plugin.
        """
        test_bash_command = [
            "bash", 
            "-c", 
            get_full_test_command(step.commands, step.working_dir)
        ]
        docker_image_path = f"{VLLM_ECR_REPO}:{self.commit}"

        if step.gpu == A100_GPU:
            test_bash_command[-1] = f"'{test_bash_command[-1]}'"
            return get_kubernetes_plugin_config(
                docker_image_path, 
                test_bash_command,
                step.num_gpus
            )
        return get_docker_plugin_config(
            docker_image_path, 
            test_bash_command, 
            step.no_gpu
        )

def mock_build_step() -> BuildkiteStep:
    step = BuildkiteStep(
        label=":docker: build image",
        key="build",
        agents={"queue": AgentQueue.AWS_CPU.value},
        env={"DOCKER_BUILDKIT": "1"},
        commands=["echo 'Mock build step'"],
        depends_on=None,
    )
    return step

@click.command()
@click.option("--run_all", type=str)
@click.option("--list_file_diff", type=str)
def main(run_all: str = -1, list_file_diff: str = None):
    list_file_diff = list_file_diff.split("|") if list_file_diff else []
    pipeline_generator = PipelineGenerator(run_all == "1", list_file_diff)
    buildkite_steps = []

    test_steps = pipeline_generator.read_test_steps(TEST_PATH)
    # Add Docker build step
    buildkite_steps.append(pipeline_generator.generate_build_step())

    # Process all test steps and convert to Buildkite-format steps
    for step in test_steps:
        buildkite_test_step = pipeline_generator.process_step(step)
        buildkite_steps.extend(buildkite_test_step)
    
    # Add external hardware tests
    external_hardware_tests = pipeline_generator.get_external_hardware_tests(test_steps)
    buildkite_steps.extend(external_hardware_tests)

    # Write Buildkite steps to pipeline.yaml
    buildkite_steps_dict = {"steps": [step.dict(exclude_none=True) for step in buildkite_steps]}
    with open(PIPELINE_FILE_PATH, "w") as f:
        yaml.dump(buildkite_steps_dict, f, sort_keys=False)


if __name__ == "__main__":
    main()
