import enum
from typing import Optional, List

# Constants
HF_HOME = "/root/.cache/huggingface"
DEFAULT_WORKING_DIR = "/vllm-workspace/tests"
VLLM_ECR_URL = "public.ecr.aws/q9t5s3a7"
VLLM_ECR_REPO = f"{VLLM_ECR_URL}/vllm-ci-test-repo"
AMD_REPO = "rocm/vllm-ci"
A100_GPU = "a100"

# File paths
TEST_PATH = ".buildkite/test-pipeline.yaml"
EXTERNAL_HARDWARE_TEST_PATH = ".buildkite/external-tests.yaml"
PIPELINE_FILE_PATH = ".buildkite/pipeline.yaml"
MULTI_NODE_TEST_SCRIPT = ".buildkite/run-multi-node-test.sh"

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
    return AgentQueue.AWS_1xL4 if num_gpus == 1 else AgentQueue.AWS_4xL4


def get_full_test_command(test_commands: List[str], step_working_dir: str) -> str:
    """Convert test commands into one-line command with the right directory."""
    working_dir = step_working_dir or DEFAULT_WORKING_DIR
    test_commands_str = ";\n".join(test_commands)
    return f"cd {working_dir};\n{test_commands_str}"


def get_multi_node_test_command(
        test_commands: List[str],
        working_dir: str,
        num_nodes: int,
        num_gpus: int,
        docker_image_path: str
        ) -> str:
    quoted_commands = [f"'{command}'" for command in test_commands]
    multi_node_command = [
        MULTI_NODE_TEST_SCRIPT,
        working_dir or DEFAULT_WORKING_DIR,
        str(num_nodes),
        str(num_gpus),
        docker_image_path,
        *quoted_commands
    ]
    return " ".join(map(str, multi_node_command))
