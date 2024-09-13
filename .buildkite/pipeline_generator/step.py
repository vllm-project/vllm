from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from utils import AgentQueue

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

def get_step_key(step_label: str) -> str:
    return step_label.replace(" ", "-").lower().replace("(", "").replace(")", "").replace("%", "").replace(",", "-")

def get_block_step(step_label: str) -> BuildkiteBlockStep:
    return BuildkiteBlockStep(block=f"Run {step_label}", key=f"block-{get_step_key(step_label)}")