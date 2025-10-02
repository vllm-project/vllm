# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OCI Registry model loader for loading models from OCI registries."""

import json
import os
import tarfile
from collections.abc import Generator
from typing import Optional

import torch
from torch import nn
import requests

from vllm import envs
from vllm.config import ModelConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.weight_utils import (
    safetensors_weights_iterator)

logger = init_logger(__name__)


class OciModelLoader(BaseModelLoader):
    """Model loader that loads models from OCI registries.
    
    This loader supports pulling models packaged as OCI artifacts with:
    - Safetensors layers (application/vnd.docker.ai.safetensors)
    - Config tar layer (application/vnd.docker.ai.vllm.config.tar)
    
    The model reference format is: [registry/]repository[:tag|@digest]
    If registry is omitted, docker.io is used by default.
    
    Example:
        model="namespace/model:tag"
        model="docker.io/user/model:v1"
        model="ghcr.io/org/model@sha256:abc123..."
    """

    SAFETENSORS_MEDIA_TYPE = "application/vnd.docker.ai.safetensors"
    CONFIG_TAR_MEDIA_TYPE = "application/vnd.docker.ai.vllm.config.tar"
    DEFAULT_REGISTRY = "docker.io"

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        self.session = requests.Session()

    def _normalize_oci_reference(self, model_ref: str) -> str:
        """Normalize OCI reference to include registry.
        
        Args:
            model_ref: Model reference (e.g., "user/model:tag")
            
        Returns:
            Normalized reference (e.g., "docker.io/user/model:tag")
        """
        # If no registry is specified (no dots before first slash),
        # prepend default registry
        if "/" in model_ref:
            first_part = model_ref.split("/")[0]
            if "." not in first_part and ":" not in first_part:
                # This is a user/repo format without registry
                return f"{self.DEFAULT_REGISTRY}/{model_ref}"
        else:
            # Single name without slash, prepend library/
            return f"{self.DEFAULT_REGISTRY}/library/{model_ref}"
        
        return model_ref

    def _get_cache_dir(self, model_ref: str) -> str:
        """Get cache directory for OCI model.
        
        Args:
            model_ref: Normalized model reference
            
        Returns:
            Path to cache directory
        """
        download_dir = self.load_config.download_dir or envs.VLLM_CACHE_ROOT
        
        # Create a safe directory name from the reference
        safe_ref = model_ref.replace(":", "_").replace("/", "_").replace(
            "@", "_")
        cache_dir = os.path.join(download_dir, "oci", safe_ref)
        os.makedirs(cache_dir, exist_ok=True)
        
        return cache_dir

    def _get_anonymous_token(self, registry: str, repository: str) -> str:
        """Get anonymous authentication token for Docker Hub.
        
        Args:
            registry: Registry hostname
            repository: Repository name
            
        Returns:
            Authentication token
        """
        auth_url = "https://auth.docker.io/token"
        params = {
            "service": "registry.docker.io",
            "scope": f"repository:{repository}:pull"
        }
        
        response = self.session.get(auth_url, params=params)
        response.raise_for_status()
        
        return response.json()["token"]
    
    def _parse_oci_reference(self, model_ref: str) -> tuple[str, str, str]:
        """Parse OCI reference into registry, repository, and tag/digest.
        
        Args:
            model_ref: Normalized OCI reference
            
        Returns:
            Tuple of (registry, repository, reference)
        """
        # Format: registry/repository:tag or registry/repository@digest
        parts = model_ref.split("/", 1)
        registry = parts[0]
        
        if "@" in parts[1]:
            repository, reference = parts[1].split("@", 1)
        elif ":" in parts[1]:
            repository, reference = parts[1].rsplit(":", 1)
        else:
            repository = parts[1]
            reference = "latest"
        
        return registry, repository, reference
    
    def _pull_oci_manifest(
            self, model_ref: str,
            cache_dir: str) -> tuple[dict, list[dict], Optional[dict]]:
        """Pull OCI manifest and identify layers.
        
        Args:
            model_ref: Normalized OCI reference
            cache_dir: Cache directory
            
        Returns:
            Tuple of (manifest, safetensors_layers, config_layer)
        """
        logger.info("Pulling OCI manifest for %s", model_ref)
        
        # Parse reference
        registry, repository, reference = self._parse_oci_reference(model_ref)
        
        # Get anonymous token for public registry (MVP)
        token = self._get_anonymous_token(registry, repository)
        
        # Pull manifest using Docker Registry HTTP API V2
        manifest_url = f"https://registry-1.{registry}/v2/{repository}/manifests/{reference}"
        headers = {
            "Accept": "application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json",
            "Authorization": f"Bearer {token}"
        }
        
        try:
            response = self.session.get(manifest_url, headers=headers)
            response.raise_for_status()
            manifest = response.json()
        except Exception as e:
            raise ValueError(
                f"Failed to pull manifest for {model_ref}. "
                f"Please ensure the image exists and is accessible. "
                f"Error: {e}"
            ) from e
        
        if not manifest:
            raise ValueError(f"Failed to pull manifest for {model_ref}")
        
        # Parse layers
        safetensors_layers = []
        config_layer = None
        
        for layer in manifest.get("layers", []):
            media_type = layer.get("mediaType", "")
            
            if media_type == self.SAFETENSORS_MEDIA_TYPE:
                safetensors_layers.append(layer)
            elif media_type == self.CONFIG_TAR_MEDIA_TYPE:
                config_layer = layer
        
        if not safetensors_layers:
            raise ValueError(
                f"No safetensors layers found in OCI image {model_ref}")
        
        logger.info("Found %d safetensors layer(s) in manifest",
                    len(safetensors_layers))
        if config_layer:
            logger.info("Found config tar layer in manifest")
        
        return manifest, safetensors_layers, config_layer

    def _download_layer(self, model_ref: str, layer: dict,
                        output_path: str) -> None:
        """Download a layer from OCI registry.
        
        Args:
            model_ref: Normalized OCI reference
            layer: Layer descriptor from manifest
            output_path: Path to save the layer
        """
        if os.path.exists(output_path):
            logger.info("Layer already cached at %s", output_path)
            return
        
        digest = layer.get("digest", "")
        size = layer.get("size", 0)
        
        logger.info("Downloading layer %s (%.2f MB)", digest,
                    size / (1024 * 1024))
        
        # Parse reference
        registry, repository, _ = self._parse_oci_reference(model_ref)
        
        # Get anonymous token
        token = self._get_anonymous_token(registry, repository)
        
        # Download blob using Docker Registry HTTP API V2
        blob_url = f"https://registry-1.{registry}/v2/{repository}/blobs/{digest}"
        headers = {
            "Authorization": f"Bearer {token}"
        }
        
        response = self.session.get(blob_url, headers=headers, stream=True)
        response.raise_for_status()
        
        # Write to file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info("Downloaded layer to %s", output_path)

    def _extract_config_tar(self, tar_path: str, extract_dir: str) -> None:
        """Extract config tar file.
        
        Args:
            tar_path: Path to tar file
            extract_dir: Directory to extract to
        """
        logger.info("Extracting config tar to %s", extract_dir)
        
        os.makedirs(extract_dir, exist_ok=True)
        
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(extract_dir)
        
        logger.info("Config extracted successfully")

    def download_oci_model_simple(self, model_ref: str) -> str:
        """Download OCI model without requiring ModelConfig.
        
        This is a simplified version for early config loading.
        
        Args:
            model_ref: OCI model reference
            
        Returns:
            Path to extracted config directory
        """
        normalized_ref = self._normalize_oci_reference(model_ref)
        cache_dir = self._get_cache_dir(normalized_ref)
        
        config_dir = os.path.join(cache_dir, "config")
        
        # Check if already downloaded
        if os.path.exists(config_dir) and os.listdir(config_dir):
            logger.info("OCI model already cached at %s", cache_dir)
            return config_dir
        
        logger.info("Downloading OCI model: %s -> %s", model_ref,
                    normalized_ref)
        
        # Pull manifest
        manifest, safetensors_layers, config_layer = self._pull_oci_manifest(
            normalized_ref, cache_dir)
        
        # Save manifest
        manifest_path = os.path.join(cache_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        
        # Download safetensors layers
        layers_dir = os.path.join(cache_dir, "layers")
        os.makedirs(layers_dir, exist_ok=True)
        
        for i, layer in enumerate(safetensors_layers):
            digest = layer.get("digest", "").replace("sha256:", "")
            layer_path = os.path.join(layers_dir,
                                      f"{i:04d}_{digest}.safetensors")
            self._download_layer(normalized_ref, layer, layer_path)
        
        # Download and extract config layer if present
        if config_layer:
            digest = config_layer.get("digest", "").replace("sha256:", "")
            tar_path = os.path.join(cache_dir, f"config_{digest}.tar")
            self._download_layer(normalized_ref, config_layer, tar_path)
            
            self._extract_config_tar(tar_path, config_dir)
        
        logger.info("Model downloaded successfully to %s", cache_dir)
        return config_dir

    def download_model(self, model_config: ModelConfig) -> None:
        """Download model from OCI registry.
        
        Args:
            model_config: Model configuration
        """
        model_ref = model_config.model
        normalized_ref = self._normalize_oci_reference(model_ref)
        cache_dir = self._get_cache_dir(normalized_ref)
        
        logger.info("Downloading OCI model: %s -> %s", model_ref,
                    normalized_ref)
        
        # Pull manifest
        manifest, safetensors_layers, config_layer = self._pull_oci_manifest(
            normalized_ref, cache_dir)
        
        # Save manifest
        manifest_path = os.path.join(cache_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        
        # Download safetensors layers
        layers_dir = os.path.join(cache_dir, "layers")
        os.makedirs(layers_dir, exist_ok=True)
        
        for i, layer in enumerate(safetensors_layers):
            digest = layer.get("digest", "").replace("sha256:", "")
            layer_path = os.path.join(layers_dir, f"{i:04d}_{digest}.safetensors")
            self._download_layer(normalized_ref, layer, layer_path)
        
        # Download and extract config layer if present
        if config_layer:
            digest = config_layer.get("digest", "").replace("sha256:", "")
            tar_path = os.path.join(cache_dir, f"config_{digest}.tar")
            self._download_layer(normalized_ref, config_layer, tar_path)
            
            config_dir = os.path.join(cache_dir, "config")
            self._extract_config_tar(tar_path, config_dir)
        
        logger.info("Model downloaded successfully to %s", cache_dir)

    def _get_weights_iterator(
        self, model_config: ModelConfig
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Get iterator over model weights from safetensors layers.
        
        Args:
            model_config: Model configuration
            
        Yields:
            Tuples of (parameter_name, tensor)
        """
        model_ref = model_config.model
        
        # Check if model_ref is already a local config path
        # (this happens when loading in worker processes)
        if os.path.isdir(model_ref) and model_ref.endswith("/config"):
            cache_dir = os.path.dirname(model_ref)
        else:
            # It's an OCI reference, normalize and get cache dir
            normalized_ref = self._normalize_oci_reference(model_ref)
            cache_dir = self._get_cache_dir(normalized_ref)
        
        # Load manifest
        manifest_path = os.path.join(cache_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            raise ValueError(
                f"Manifest not found at {manifest_path}. "
                f"Cache dir: {cache_dir}, Model ref: {model_ref}")
        
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        
        # Get safetensors layers in order
        layers_dir = os.path.join(cache_dir, "layers")
        safetensors_files = []
        
        for layer in manifest.get("layers", []):
            if layer.get("mediaType") == self.SAFETENSORS_MEDIA_TYPE:
                digest = layer.get("digest", "").replace("sha256:", "")
                # Find matching file
                for filename in sorted(os.listdir(layers_dir)):
                    if digest in filename and filename.endswith(
                            ".safetensors"):
                        safetensors_files.append(
                            os.path.join(layers_dir, filename))
                        break
        
        if not safetensors_files:
            raise ValueError(f"No safetensors files found in {layers_dir}")
        
        logger.info("Loading weights from %d safetensors file(s)",
                    len(safetensors_files))
        
        # Use existing safetensors iterator
        for name, tensor in safetensors_weights_iterator(
                safetensors_files,
                use_tqdm_on_load=self.load_config.use_tqdm_on_load,
                safetensors_load_strategy=self.load_config.safetensors_load_strategy):
            yield name, tensor

    def load_weights(self, model: nn.Module,
                     model_config: ModelConfig) -> None:
        """Load weights into the model from OCI layers.
        
        Args:
            model: Model to load weights into
            model_config: Model configuration
        """
        # Get the config directory path - update model_config.model to point
        # to the extracted config for compatibility with other components
        normalized_ref = self._normalize_oci_reference(model_config.model)
        cache_dir = self._get_cache_dir(normalized_ref)
        config_dir = os.path.join(cache_dir, "config")
        
        # If config directory exists, temporarily update model path
        original_model = model_config.model
        if os.path.exists(config_dir):
            logger.info("Using config from %s", config_dir)
            # Store original and update for tokenizer/config loading
            model_config._original_model = original_model
            model_config.model = config_dir
        
        try:
            # Load weights using iterator
            weights_to_load = {name for name, _ in model.named_parameters()}
            loaded_weights = model.load_weights(
                self._get_weights_iterator(model_config))
            
            # Check if all weights were loaded (for non-quantized models)
            if model_config.quantization is None and loaded_weights is not None:
                weights_not_loaded = weights_to_load - loaded_weights
                if weights_not_loaded:
                    raise ValueError(
                        "Following weights were not initialized from "
                        f"checkpoint: {weights_not_loaded}")
            
            logger.info("Weights loaded successfully from OCI registry")
        finally:
            # Restore original model reference
            if hasattr(model_config, "_original_model"):
                model_config.model = model_config._original_model
                delattr(model_config, "_original_model")
