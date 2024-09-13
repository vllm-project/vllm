import enum
from typing import Optional
HF_HOME = "/root/.cache/huggingface"
DEFAULT_WORKING_DIR = "/vllm-workspace/tests"
VLLM_ECR_REPO = "public.ecr.aws/q9t5s3a7/vllm-ci-test-repo"
EXTERNAL_HARDWARE_TESTS = ".buildkite/external-tests.yaml"
AMD_REPO = "rocm/vllm-ci"

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
    if gpu_type == "a100":
        return AgentQueue.A100
    if num_gpus == 1:
        return AgentQueue.AWS_1xL4
    return AgentQueue.AWS_4xL4