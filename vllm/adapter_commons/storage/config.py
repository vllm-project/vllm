"""Configuration models for storage providers."""

import os
from typing import Optional, List, Dict, Any, Type, TypeVar, Union
from pathlib import Path
from pydantic import BaseModel, Field, validator, root_validator

T = TypeVar('T', bound='StorageConfig')

def parse_size(v: Union[int, str]) -> int:
    """Parse size string with units into bytes."""
    if isinstance(v, int):
        return v
    
    units = {
        'k': 1024,
        'm': 1024 * 1024,
        'g': 1024 * 1024 * 1024,
        't': 1024 * 1024 * 1024 * 1024,
    }
    
    v = v.lower().strip()
    for unit, multiplier in units.items():
        if v.endswith(unit) or v.endswith(f"{unit}b"):
            try:
                return int(float(v[:-1]) * multiplier)
            except ValueError:
                raise ValueError(f"Invalid size format: {v}")
    
    try:
        return int(v)
    except ValueError:
        raise ValueError(f"Invalid size format: {v}")

def parse_list(v: Union[str, List[str]]) -> List[str]:
    """Parse comma-separated string into list."""
    if isinstance(v, list):
        return v
    return [x.strip() for x in v.split(",") if x.strip()]

class StorageConfig(BaseModel):
    """Base configuration for storage providers."""
    
    provider_type: str = Field(
        ...,
        description="Storage provider type identifier",
        env="VLLM_STORAGE_TYPE"
    )
    
    max_concurrent_ops: int = Field(
        default=4,
        ge=1,
        description="Maximum number of concurrent operations",
        env="VLLM_MAX_CONCURRENT_OPS"
    )
    
    class Config:
        """Pydantic model configuration."""
        extra = "allow"  # Allow extra fields for provider-specific config
        env_prefix = "VLLM_STORAGE_"
        case_sensitive = False
        
    @classmethod
    def from_env(cls: Type[T]) -> T:
        """Create configuration from environment variables."""
        return cls.parse_obj({
            key.lower(): value
            for key, value in os.environ.items()
            if key.startswith(cls.Config.env_prefix)
        })

class LocalStorageConfig(StorageConfig):
    """Configuration for local filesystem storage provider."""
    
    provider_type: str = Field(
        "local",
        const=True,
        description="Local storage provider type"
    )
    
    allowed_paths: Optional[List[str]] = Field(
        default=None,
        description="List of allowed base directories for security",
        env="VLLM_LOCAL_ALLOWED_PATHS"
    )
    
    create_dirs: bool = Field(
        default=True,
        description="Create directories if they don't exist",
        env="VLLM_LOCAL_CREATE_DIRS"
    )
    
    allow_symlinks: bool = Field(
        default=False,
        description="Allow symlinks in paths",
        env="VLLM_LOCAL_ALLOW_SYMLINKS"
    )
    
    verify_permissions: bool = Field(
        default=True,
        description="Verify file permissions for security",
        env="VLLM_LOCAL_VERIFY_PERMS"
    )
    
    preserve_permissions: bool = Field(
        default=True,
        description="Preserve file permissions when copying",
        env="VLLM_LOCAL_PRESERVE_PERMS"
    )
    
    @validator("allowed_paths", pre=True)
    def validate_allowed_paths(cls, v: Optional[Union[str, List[str]]]) -> Optional[List[str]]:
        """Validate and normalize allowed paths."""
        if v is None:
            return v
            
        paths = parse_list(v) if isinstance(v, str) else v
        normalized = []
        for path in paths:
            try:
                # Expand user and resolve path
                resolved = str(Path(path).expanduser().resolve())
                normalized.append(resolved)
            except Exception as e:
                raise ValueError(f"Invalid path {path}: {e}")
                
        return normalized
    
    class Config:
        """Pydantic model configuration."""
        env_prefix = "VLLM_LOCAL_"
        case_sensitive = False

class S3StorageConfig(StorageConfig):
    """Configuration for S3 storage provider."""
    
    provider_type: str = Field(
        "s3",
        const=True,
        description="S3 storage provider type"
    )
    
    # AWS Configuration
    region_name: Optional[str] = Field(
        default=None,
        description="AWS region name",
        env="VLLM_S3_REGION"
    )
    
    endpoint_url: Optional[str] = Field(
        default=None,
        description="Custom S3 endpoint URL",
        env="VLLM_S3_ENDPOINT"
    )
    
    aws_access_key_id: Optional[str] = Field(
        default=None,
        description="AWS access key ID",
        env="AWS_ACCESS_KEY_ID"
    )
    
    aws_secret_access_key: Optional[str] = Field(
        default=None,
        description="AWS secret access key",
        env="AWS_SECRET_ACCESS_KEY"
    )
    
    aws_session_token: Optional[str] = Field(
        default=None,
        description="AWS session token",
        env="AWS_SESSION_TOKEN"
    )
    
    # Connection Settings
    max_connections: int = Field(
        default=10,
        ge=1,
        description="Maximum number of connections",
        env="VLLM_S3_MAX_CONNECTIONS"
    )
    
    connect_timeout: float = Field(
        default=10.0,
        ge=0,
        description="Connection timeout in seconds",
        env="VLLM_S3_CONNECT_TIMEOUT"
    )
    
    read_timeout: float = Field(
        default=30.0,
        ge=0,
        description="Read timeout in seconds",
        env="VLLM_S3_READ_TIMEOUT"
    )
    
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum number of retry attempts",
        env="VLLM_S3_MAX_RETRIES"
    )
    
    # Transfer Settings
    max_concurrent_downloads: int = Field(
        default=4,
        ge=1,
        description="Maximum number of concurrent downloads",
        env="VLLM_S3_MAX_CONCURRENT_DOWNLOADS"
    )
    
    multipart_threshold: int = Field(
        default=8 * 1024 * 1024,  # 8MB
        ge=5 * 1024 * 1024,  # Min 5MB
        description="Threshold for multipart downloads",
        env="VLLM_S3_MULTIPART_THRESHOLD"
    )
    
    multipart_chunksize: int = Field(
        default=8 * 1024 * 1024,  # 8MB
        ge=5 * 1024 * 1024,  # Min 5MB
        description="Chunk size for multipart downloads",
        env="VLLM_S3_MULTIPART_CHUNKSIZE"
    )
    
    @validator("endpoint_url")
    def validate_endpoint_url(cls, v: Optional[str]) -> Optional[str]:
        """Validate S3 endpoint URL."""
        if v is not None and not v.startswith(("http://", "https://")):
            raise ValueError("Endpoint URL must start with http:// or https://")
        return v
    
    @validator("multipart_threshold", "multipart_chunksize", pre=True)
    def validate_size(cls, v: Union[int, str]) -> int:
        """Validate and convert size values."""
        return parse_size(v)
    
    class Config:
        """Pydantic model configuration."""
        env_prefix = "VLLM_S3_"
        case_sensitive = False 