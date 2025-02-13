# SPDX-License-Identifier: Apache-2.0
import logging
import os
import shutil

import boto3
from huggingface_hub import HfApi, snapshot_download
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTransfer:

    def __init__(self,
                 model_id,
                 s3_bucket,
                 aws_access_key_id=None,
                 aws_secret_access_key=None,
                 aws_region=None):
        """
        Initialize the ModelTransfer class.
        
        Args:
            model_id (str): HuggingFace model ID 
            s3_bucket (str): Name of the S3 bucket
            aws_access_key_id (str, optional)
            aws_secret_access_key (str, optional)
            aws_region (str, optional): AWS region. Defaults to None.
        """
        self.model_id = model_id
        self.s3_bucket = s3_bucket
        self.model_name = model_id.split('/')[-1]

        # Initialize S3 client
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_region)

        # Initialize Hugging Face API
        self.hf_api = HfApi()

    def download_model(self, local_dir):
        """
        Download the model from HuggingFace.
        
        Args:
            local_dir (str): Local directory to save the model
        
        Returns:
            str: Path to the downloaded model directory
        """
        logger.info("Downloading model %s...", self.model_id)

        try:
            local_dir_with_model = os.path.join(local_dir, self.model_name)
            snapshot_download(repo_id=self.model_id,
                              local_dir=local_dir_with_model,
                              local_dir_use_symlinks=False,
                              token=os.getenv("HF_TOKEN"))
            logger.info("Model downloaded successfully to %s",
                        local_dir_with_model)
            return local_dir_with_model

        except Exception as e:
            logger.error("Error downloading model: %s", str(e))
            raise

    def upload_to_s3(self, local_dir):
        """
        Upload the model directory to S3.
        
        Args:
            local_dir (str): Local directory containing the model files
        """
        logger.info("Uploading model to S3 bucket %s...", self.s3_bucket)

        try:
            # Walk through all files in the directory
            for root, _, files in os.walk(local_dir):
                for filename in files:
                    # Get the full local path
                    local_path = os.path.join(root, filename)

                    # Calculate S3 path (preserve directory structure)
                    relative_path = os.path.relpath(local_path, local_dir)
                    s3_path = f"{self.model_name}/{relative_path}"

                    # Upload file with progress bar
                    file_size = os.path.getsize(local_path)
                    with tqdm(total=file_size,
                              unit='B',
                              unit_scale=True,
                              desc=f"Uploading {filename}") as pbar:
                        self.s3_client.upload_file(
                            local_path,
                            self.s3_bucket,
                            s3_path,
                            Callback=lambda bytes_transferred: pbar.update(
                                bytes_transferred))

                    logger.info("Uploaded %s to s3://%s/%s", filename,
                                self.s3_bucket, s3_path)

            logger.info("Model upload completed successfully!")

        except Exception as e:
            logger.error("Error uploading to S3: %s", str(e))
            raise


# "ibm/PowerMoE-3b", "internlm/internlm-chat-7b",
#         "internlm/internlm2-chat-7b", "OpenGVLab/Mono-InternVL-2B",
#         "internlm/internlm3-8b-instruct", "inceptionai/jais-13b-chat",
#         "ai21labs/AI21-Jamba-1.5-Mini", "meta-llama/Meta-Llama-3-8B",
#         "decapoda-research/llama-7b-hf", "state-spaces/mamba-130m-hf",
#         "tiiuae/falcon-mamba-7b-instruct", "openbmb/MiniCPM-2B-sft-bf16",
#         "openbmb/MiniCPM3-4B", "mistralai/Mistral-7B-Instruct-v0.1",
#         "mistralai/Mixtral-8x7B-Instruct-v0.1",
#         "mistral-community/Mixtral-8x22B-v0.1-AWQ", "mpt", "mosaicml/mpt-7b",
#         "nvidia/Minitron-8B-Base", "allenai/OLMo-1B-hf",
#         "shanearora/OLMo-7B-1124-hf", "allenai/OLMoE-1B-7B-0924-Instruct",
#         "facebook/opt-iml-max-1.3b", "OrionStarAI/Orion-14B-Chat",
#         "adept/persimmon-8b-chat", "microsoft/phi-2",
#         "microsoft/Phi-3-mini-4k-instruct",
#         "microsoft/Phi-3-small-8k-instruct", "microsoft/Phi-3.5-MoE-instruct",
#         "Qwen/Qwen2-7B-Instruct", "Qwen/Qwen1.5-MoE-A2.7B-Chat",
#         "tiiuae/falcon-40b", "stabilityai/stablelm-zephyr-3b",
#         "stabilityai/stablelm-3b-4e1t", "bigcode/starcoder2-3b",
#         "upstage/solar-pro-preview-instruct", "Tele-AI/TeleChat2-3B",
#         "xverse/XVERSE-7B-Chat", "facebook/bart-base",
#         "facebook/bart-large-cnn", "microsoft/Florence-2-base",
#         "BAAI/bge-base-en-v1.5", "BAAI/bge-multilingual-gemma2",
#         "parasail-ai/GritLM-7B-vllm", "internlm/internlm2-1_8b-reward",
#         "ai21labs/Jamba-tiny-reward-dev", "llama",
#         "intfloat/e5-mistral-7b-instruct",
#         "ssmits/Qwen2-7B-Instruct-embed-base", "Qwen/Qwen2.5-Math-RM-72B",
#         "Qwen/Qwen2.5-Math-PRM-7B", "jason9693/Qwen2.5-1.5B-apeach",
#         "sentence-transformers/stsb-roberta-base-v2",
#         "sentence-transformers/all-roberta-large-v1",
#         "intfloat/multilingual-e5-large", "royokong/e5-v",
#         "TIGER-Lab/VLM2Vec-Full", "MrLight/dse-qwen2-2b-mrl-v1",
#         "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11",
#         "cross-encoder/ms-marco-MiniLM-L-6-v2",
#         "cross-encoder/quora-roberta-base", "BAAI/bge-reranker-v2-m3",
#         "THUDM/glm-4v-9b", "chatglm2-6b", "deepseek-ai/deepseek-vl2-tiny",
#         "adept/fuyu-8b", "h2oai/h2ovl-mississippi-800m",
#         "OpenGVLab/InternVL2-1B", "HuggingFaceM4/Idefics3-8B-Llama3",
#         "llava-hf/llava-1.5-7b-hf", "llava-hf/llava-v1.6-mistral-7b-hf",
#         "llava-hf/LLaVA-NeXT-Video-7B-hf",
#         "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
#         "TIGER-Lab/Mantis-8B-siglip-llama3", "openbmb/MiniCPM-o-2_6",
#         "openbmb/MiniCPM-V-2_6", "allenai/Molmo-7B-D-0924",
#         "nvidia/NVLM-D-72B", "google/paligemma-3b-pt-224",
#         "microsoft/Phi-3-vision-128k-instruct", "mistralai/Pixtral-12B-2409",
#         "Qwen/Qwen-VL-Chat", "Qwen/Qwen2-Audio-7B-Instruct",
#         "Qwen/Qwen2-VL-2B-Instruct", "Qwen/Qwen2.5-VL-3B-Instruct",
#         "fixie-ai/ultravox-v0_5-llama-3_2-1b",
#         "meta-llama/Llama-3.2-11B-Vision-Instruct", "openai/whisper-large-v3",
#         "JackFram/llama-68m", "JackFram/llama-68m", "JackFram/llama-160m",
#         "ArthurZ/Ilama-3.2-1B"


def main():
    # Configuration
    MODEL_ID = [
        "deepseek-ai/DeepSeek-V2-Lite-Chat",
    ]
    S3_BUCKET = "vllm-ci-model-weights"
    # Local directory to temporarily store the model
    LOCAL_DIR = "/home/ec2-user/models"

    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION = "us-west-2"

    # Create transfer object
    for model_id in MODEL_ID:
        transfer = ModelTransfer(model_id=model_id,
                                 s3_bucket=S3_BUCKET,
                                 aws_access_key_id=AWS_ACCESS_KEY_ID,
                                 aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                                 aws_region=AWS_REGION)

        try:
            # Create local directory if it doesn't exist
            os.makedirs(LOCAL_DIR, exist_ok=True)

            # Download model
            model_dir = transfer.download_model(LOCAL_DIR)

            # Upload to S3 and cleanup
            transfer.upload_to_s3(model_dir)
            shutil.rmtree(model_dir)

        except Exception as e:
            logger.error("Error in transfer process: %s", str(e))
            raise


if __name__ == "__main__":
    main()
