import io
import json
import logging
import os
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, Optional

import safetensors.torch
import torch
from botocore.config import Config
from botocore.exceptions import ClientError, NoCredentialsError

logger = logging.getLogger(__name__)

try:
    import boto3
except ImportError:
    boto3 = None


class LoRASourceError(Exception):
    """Base exception for LoRA source errors"""
    pass


class S3LoRASourceError(LoRASourceError):
    """S3-specific LoRA loading errors with original error preservation"""

    def __init__(self,
                 message: str,
                 original_error: Optional[Exception] = None):
        super().__init__(message)
        self.original_error = original_error


class LoRASource(ABC):
    """Abstract base class for different LoRA sources"""

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get adapter config directly to memory"""
        pass

    @abstractmethod
    def get_tensors(self) -> Dict[str, torch.Tensor]:
        """Get model tensors directly to memory"""
        pass

    @abstractmethod
    def get_embeddings(self) -> Optional[Dict[str, torch.Tensor]]:
        """Get additional embeddings if they exist"""
        pass

    @abstractmethod
    def get_local_path(self) -> str:
        """Get or create local path for compatibility with existing code"""
        pass


class LocalLoRASource(LoRASource):
    """Handles local filesystem LoRA sources"""

    def __init__(self, path: str):
        self.path = path

    def get_config(self) -> Dict[str, Any]:
        config_path = os.path.join(self.path, "adapter_config.json")
        with open(config_path) as f:
            return json.load(f)

    def get_tensors(self) -> Dict[str, torch.Tensor]:
        tensor_path = os.path.join(self.path, "adapter_model.safetensors")
        bin_path = os.path.join(self.path, "adapter_model.bin")

        if os.path.isfile(tensor_path):
            return safetensors.torch.load_file(tensor_path)
        elif os.path.isfile(bin_path):
            return torch.load(bin_path)
        raise ValueError(f"{self.path} doesn't contain tensors")

    def get_embeddings(self) -> Optional[Dict[str, torch.Tensor]]:
        embeddings_path = os.path.join(self.path, "new_embeddings.safetensors")
        bin_path = os.path.join(self.path, "new_embeddings.bin")

        if os.path.isfile(embeddings_path):
            return safetensors.torch.load_file(embeddings_path)
        elif os.path.isfile(bin_path):
            return torch.load(bin_path)
        return None

    def get_local_path(self) -> str:
        return self.path


class S3LoRASource(LoRASource):
    """Handles S3 LoRA sources with direct memory loading"""

    def __init__(self, s3_path: str):
        if not s3_path.startswith("s3://"):
            raise S3LoRASourceError(f"Invalid S3 path format: {s3_path}")

        if boto3 is None:
            raise LoRASourceError(
                "S3 support requires boto3. Install with: pip install vllm[s3] "
                "or pip install boto3"
            )

        try:
            self.bucket, self.prefix = self._parse_s3_path(s3_path)
            config = Config(retries=dict(max_attempts=3),
                            connect_timeout=5,
                            read_timeout=60,
                            max_pool_connections=50)
            self.s3_client = boto3.client("s3", config=config)
            self._local_cache_dir: Optional[str] = None
            # Validate bucket access and required files
            self._validate_access()
        except (ClientError, NoCredentialsError) as e:
            raise S3LoRASourceError("Failed to initialize S3 client") from e

    @staticmethod
    def _parse_s3_path(path: str) -> tuple[str, str]:
        """Parse S3 path into bucket and prefix.
        
        Args:
            path: Full S3 path (e.g. s3://bucket/path/to/adapter)
            
        Returns:
            Tuple of (bucket_name, key_prefix)
        """
        # Remove s3:// prefix
        path = path.replace("s3://", "")

        # Split into bucket and prefix parts
        parts = path.split("/")
        if len(parts) < 1:
            raise S3LoRASourceError(f"Invalid S3 path format: {path}")

        bucket = parts[0]
        # Join remaining parts as prefix, preserving structure
        prefix = "/".join(parts[1:]) if len(parts) > 1 else ""

        return bucket, prefix

    def _validate_access(self) -> None:
        """Validate bucket access and required files existence"""
        try:
            # Check adapter_config.json
            config_key = (f"{self.prefix}/adapter_config.json"
                          if self.prefix else "adapter_config.json")
            self.s3_client.head_object(Bucket=self.bucket, Key=config_key)

            # Check for either safetensors or bin model file
            try:
                model_key = (f"{self.prefix}/adapter_model.safetensors"
                             if self.prefix else "adapter_model.safetensors")
                self.s3_client.head_object(Bucket=self.bucket, Key=model_key)
            except ClientError:
                model_key = (f"{self.prefix}/adapter_model.bin"
                             if self.prefix else "adapter_model.bin")
                self.s3_client.head_object(Bucket=self.bucket, Key=model_key)
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                msg = f"Required files not found in {self.bucket}/{self.prefix}"
                raise S3LoRASourceError(msg) from e
            elif error_code == '403':
                msg = f"No permission to access {self.bucket}/{self.prefix}"
                raise S3LoRASourceError(msg) from e
            raise S3LoRASourceError(f"S3 error: {str(e)}") from e

    def get_config(self) -> Dict[str, Any]:
        try:
            config_key = (f"{self.prefix}/adapter_config.json"
                          if self.prefix else "adapter_config.json")
            response = self.s3_client.get_object(Bucket=self.bucket,
                                                 Key=config_key)
            return json.loads(response["Body"].read())
        except ClientError as e:
            raise S3LoRASourceError("Failed to load adapter config") from e
        except json.JSONDecodeError as e:
            raise S3LoRASourceError("Invalid adapter config JSON") from e

    @contextmanager
    def _get_streaming_buffer(self, key: str, chunk_size: int = 1024 * 1024):
        """Get a file-like buffer that streams from S3.
        
        Args:
            key: S3 key to stream
            chunk_size: Size of chunks to stream (default 1MB)
            
        Yields:
            BytesIO object that can be used as a file
        """
        try:
            response = self.s3_client.get_object(Bucket=self.bucket, Key=key)
            buffer = io.BytesIO()
            stream = response['Body']

            while True:
                chunk = stream.read(chunk_size)
                if not chunk:
                    break
                buffer.write(chunk)

            buffer.seek(0)
            yield buffer
        finally:
            stream.close()
            buffer.close()

    def get_tensors(self) -> Dict[str, torch.Tensor]:
        """Get model tensors using streaming to avoid memory issues"""
        try:
            # Try safetensors first
            try:
                model_key = (f"{self.prefix}/adapter_model.safetensors"
                             if self.prefix else "adapter_model.safetensors")
                with self._get_streaming_buffer(model_key) as buffer:
                    return safetensors.torch.load_buffer(buffer.read())
            except ClientError as e:
                if e.response['Error']['Code'] != '404':
                    raise

                # Fallback to .bin format
                model_key = (f"{self.prefix}/adapter_model.bin"
                             if self.prefix else "adapter_model.bin")
                with self._get_streaming_buffer(model_key) as buffer:
                    return torch.load(buffer)
        except ClientError as e:
            raise S3LoRASourceError("Failed to load adapter model") from e
        except Exception as e:
            msg = f"Failed to parse adapter model: {str(e)}"
            raise S3LoRASourceError(msg) from e

    def get_embeddings(self) -> Optional[Dict[str, torch.Tensor]]:
        """Optional embeddings - gracefully handle missing files"""
        try:
            # Try safetensors first
            try:
                embed_key = (f"{self.prefix}/new_embeddings.safetensors"
                             if self.prefix else "new_embeddings.safetensors")
                with self._get_streaming_buffer(embed_key) as buffer:
                    return safetensors.torch.load_buffer(buffer.read())
            except ClientError as e:
                if e.response['Error']['Code'] != '404':
                    raise

                # Try .bin format
                try:
                    embed_key = (f"{self.prefix}/new_embeddings.bin"
                                 if self.prefix else "new_embeddings.bin")
                    with self._get_streaming_buffer(embed_key) as buffer:
                        return torch.load(buffer)
                except ClientError as e2:
                    if e2.response['Error']['Code'] != '404':
                        raise
            return None
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return None
            logger.warning("Error loading embeddings: %s", str(e))
            return None
        except Exception as e:
            logger.warning("Failed to parse embeddings: %s", str(e))
            return None

    def get_local_path(self) -> str:
        """Create a local cache of the S3 files for compatibility"""
        if self._local_cache_dir is None:
            # Create temporary directory
            prefix_path = self.prefix.replace('/', '_')
            cache_name = f"vllm_lora_cache_{self.bucket}_{prefix_path}"
            self._local_cache_dir = os.path.join("/tmp", cache_name)
            os.makedirs(self._local_cache_dir, exist_ok=True)

            try:
                # Download required files
                for file in [
                        "adapter_config.json", "adapter_model.safetensors",
                        "adapter_model.bin"
                ]:
                    key = f"{self.prefix}/{file}" if self.prefix else file
                    local_path = os.path.join(self._local_cache_dir, file)
                    try:
                        self.s3_client.download_file(self.bucket, key,
                                                     local_path)
                    except ClientError as e:
                        if e.response['Error']['Code'] != '404':
                            raise

                # Try downloading embeddings if they exist
                for file in [
                        "new_embeddings.safetensors", "new_embeddings.bin"
                ]:
                    key = f"{self.prefix}/{file}" if self.prefix else file
                    local_path = os.path.join(self._local_cache_dir, file)
                    try:
                        self.s3_client.download_file(self.bucket, key,
                                                     local_path)
                    except ClientError as e:
                        if e.response['Error']['Code'] != '404':
                            raise

            except Exception as e:
                # Cleanup on failure
                import shutil
                shutil.rmtree(self._local_cache_dir, ignore_errors=True)
                self._local_cache_dir = None
                raise S3LoRASourceError("Failed to create local cache") from e

        assert self._local_cache_dir is not None
        return self._local_cache_dir
