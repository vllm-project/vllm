import enum
import os
from typing import Optional, List

HF_HOME = "/root/.cache/huggingface"
DEFAULT_WORKING_DIR = "/vllm-workspace/tests"

VLLM_ECR_URL = "public.ecr.aws/q9t5s3a7"
VLLM_ECR_REPO = "public.ecr.aws/q9t5s3a7/vllm-ci-test-repo"
AMD_REPO = "rocm/vllm-ci"

TEST_PATH = ".buildkite/test-pipeline.yaml"
EXTERNAL_HARDWARE_TEST_PATH = ".buildkite/external-tests.yaml"
PIPELINE_FILE_PATH = ".buildkite/pipeline.yaml"

A100_GPU = "a100"

STEPS_TO_BLOCK = []

class AgentQueue(str, enum.Enum):
    AWS_CPU = "cpu_queue"
    AWS_SMALL_CPU = "small_cpu_queue"
    AWS_1xL4 = "gpu_1_queue"
    AWS_4xL4 = "gpu_4_queue"
    A100 = "a100-queue"
    AMD_GPU = "amd"
    AMD_CPU = "amd-cpu"

def get_agent_queue(no_gpu: Optional[bool], gpu_type: Optional[str], num_gpus: Optional[int]) -> AgentQueue:
    if no_gpu:
        return AgentQueue.AWS_SMALL_CPU
    if gpu_type == A100_GPU:
        return AgentQueue.A100
    if num_gpus == 1:
        return AgentQueue.AWS_1xL4
    return AgentQueue.AWS_4xL4

def get_full_test_command(test_commands: List[str], step_working_dir: str) -> str:
    """Convert test commands into one-line command with the right directory."""
    test_commands = "; ".join(test_commands)
    working_dir = step_working_dir or DEFAULT_WORKING_DIR
    full_test_command = f"cd {working_dir}; {test_commands}"
    return full_test_command

def get_image_path(repo: Optional[str] = None) -> str:
    """Get path to image of the current commit on specified container registry."""
    commit = os.getenv("BUILDKITE_COMMIT")
    if repo is None:
        repo = VLLM_ECR_REPO
    return f"{repo}:{commit}"

def get_multi_node_test_command(test_commands: List[str], working_dir: str, num_nodes: int, num_gpus: int, docker_image_path: str) -> str:
    test_command = [f"'{command}'" for command in test_commands]
    multi_node_command = [
        ".buildkite/run-multi-node-test.sh", 
        str(working_dir or DEFAULT_WORKING_DIR), 
        str(num_nodes), 
        str(num_gpus), 
        docker_image_path
    ]
    multi_node_command.extend(test_command)
    return " ".join(multi_node_command)
