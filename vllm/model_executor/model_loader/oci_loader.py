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

from vllm import envs
from vllm.config import ModelConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.oci_go_client import OciGoClient
from vllm.model_executor.model_loader.weight_utils import safetensors_weights_iterator

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
        self.go_client = OciGoClient()

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
        safe_ref = model_ref.replace(":", "_").replace("/", "_").replace("@", "_")
        cache_dir = os.path.join(download_dir, "oci", safe_ref)
        os.makedirs(cache_dir, exist_ok=True)

        return cache_dir



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
        self, model_ref: str, cache_dir: str
    ) -> tuple[dict, list[dict], Optional[dict]]:
        """Pull OCI manifest and identify layers.

        Args:
            model_ref: Normalized OCI reference
            cache_dir: Cache directory

        Returns:
            Tuple of (manifest, safetensors_layers, config_layer)
        """
        logger.info("Pulling OCI manifest for %s", model_ref)

        # Use Go client to pull manifest (supports docker login authentication)
        try:
            manifest = self.go_client.pull_manifest(model_ref)
        except Exception as e:
            raise ValueError(
                f"Failed to pull manifest for {model_ref}. "
                f"Please ensure the image exists and is accessible. "
                f"If this is a private image, make sure you have run 'docker login' "
                f"for the registry. Error: {e}"
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
            raise ValueError(f"No safetensors layers found in OCI image {model_ref}")

        logger.info(
            "Found %d safetensors layer(s) in manifest", len(safetensors_layers)
        )
        if config_layer:
            logger.info("Found config tar layer in manifest")

        return manifest, safetensors_layers, config_layer

    def _download_layer(self, model_ref: str, layer: dict, output_path: str) -> None:
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

        logger.info("Downloading layer %s (%.2f MB)", digest, size / (1024 * 1024))

        # Use Go client to download blob (supports docker login authentication)
        try:
            self.go_client.pull_blob(model_ref, digest, output_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to download layer {digest}. "
                f"If this is a private image, make sure you have run 'docker login'. "
                f"Error: {e}"
            ) from e

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

    def _download_oci_model_if_needed(
        self, model_ref: str, download_weights: bool = True
    ) -> str:
        """Download OCI model and its components if not already cached.

        This is the shared logic for both download_model and
        download_oci_model_simple.

        Args:
            model_ref: OCI model reference
            download_weights: If True, download safetensors weight layers.
                            If False, only download config layer.

        Returns:
            Path to the extracted config directory
        """
        normalized_ref = self._normalize_oci_reference(model_ref)
        cache_dir = self._get_cache_dir(normalized_ref)
        config_dir = os.path.join(cache_dir, "config")
        manifest_path = os.path.join(cache_dir, "manifest.json")

        # Check if config directory is already populated and manifest exists
        config_exists = (
            os.path.exists(config_dir)
            and os.listdir(config_dir)
            and os.path.exists(manifest_path)
        )

        if config_exists and not download_weights:
            logger.info("OCI model config already cached at %s", cache_dir)
            return config_dir

        # If weights are needed, check if all layers are downloaded
        if config_exists and download_weights:
            with open(manifest_path) as f:
                manifest = json.load(f)

            safetensors_layers = [
                layer
                for layer in manifest.get("layers", [])
                if layer.get("mediaType") == self.SAFETENSORS_MEDIA_TYPE
            ]

            layers_dir = os.path.join(cache_dir, "layers")
            all_weights_cached = True

            if safetensors_layers and os.path.exists(layers_dir):
                for i, layer in enumerate(safetensors_layers):
                    digest = layer.get("digest", "").replace("sha256:", "")
                    layer_path = os.path.join(
                        layers_dir, f"{i:04d}_{digest}.safetensors"
                    )
                    if not os.path.exists(layer_path):
                        all_weights_cached = False
                        break
            else:
                all_weights_cached = False

            if all_weights_cached:
                logger.info("OCI model fully cached at %s", cache_dir)
                return config_dir

        logger.info(
            "Downloading OCI model: %s -> %s (weights=%s)",
            model_ref,
            normalized_ref,
            download_weights,
        )

        # Pull manifest (or reload if exists)
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                manifest = json.load(f)

            safetensors_layers = [
                layer
                for layer in manifest.get("layers", [])
                if layer.get("mediaType") == self.SAFETENSORS_MEDIA_TYPE
            ]
            config_layer = next(
                (
                    layer
                    for layer in manifest.get("layers", [])
                    if layer.get("mediaType") == self.CONFIG_TAR_MEDIA_TYPE
                ),
                None,
            )
        else:
            manifest, safetensors_layers, config_layer = self._pull_oci_manifest(
                normalized_ref, cache_dir
            )

            # Save manifest
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)

            # Save original OCI reference for later retrieval
            metadata_path = os.path.join(cache_dir, "oci_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump({"original_reference": model_ref}, f, indent=2)

        # Download safetensors layers only if requested
        if download_weights:
            layers_dir = os.path.join(cache_dir, "layers")
            os.makedirs(layers_dir, exist_ok=True)

            for i, layer in enumerate(safetensors_layers):
                digest = layer.get("digest", "").replace("sha256:", "")
                layer_path = os.path.join(layers_dir, f"{i:04d}_{digest}.safetensors")
                self._download_layer(normalized_ref, layer, layer_path)

        # Download and extract config layer if present
        if config_layer and not os.path.exists(config_dir):
            digest = config_layer.get("digest", "").replace("sha256:", "")
            tar_path = os.path.join(cache_dir, f"config_{digest}.tar")
            self._download_layer(normalized_ref, config_layer, tar_path)
            self._extract_config_tar(tar_path, config_dir)

        logger.info("Model download completed: %s", cache_dir)
        return config_dir

    def download_oci_model_simple(self, model_ref: str) -> str:
        """Download OCI model without requiring ModelConfig.

        This is a simplified version for early config loading that only
        downloads the config layer, not the weight files. Weights are
        downloaded later during model initialization.

        Args:
            model_ref: OCI model reference

        Returns:
            Path to extracted config directory
        """
        return self._download_oci_model_if_needed(model_ref, download_weights=False)

    def download_model(self, model_config: ModelConfig) -> None:
        """Download model from OCI registry.

        Args:
            model_config: Model configuration
        """
        model_ref = model_config.model

        # If model_ref is a local config path, read the original OCI reference
        if os.path.isdir(model_ref) and model_ref.endswith("/config"):
            cache_dir = os.path.dirname(model_ref)
            metadata_path = os.path.join(cache_dir, "oci_metadata.json")

            if os.path.exists(metadata_path):
                with open(metadata_path) as f:
                    metadata = json.load(f)
                    model_ref = metadata.get("original_reference", model_ref)
                    logger.info("Retrieved original OCI reference: %s", model_ref)

        self._download_oci_model_if_needed(model_ref)

    def _get_weights_iterator(
        self, model_config: ModelConfig
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Get iterator over model weights from safetensors layers.

        Downloads weights if they haven't been downloaded yet.

        Args:
            model_config: Model configuration

        Yields:
            Tuples of (parameter_name, tensor)
        """
        model_ref = model_config.model
        original_oci_ref = None

        # Check if model_ref is already a local config path
        # (this happens when loading in worker processes)
        if os.path.isdir(model_ref) and model_ref.endswith("/config"):
            cache_dir = os.path.dirname(model_ref)
            # Try to extract original OCI reference from attribute
            original_oci_ref = getattr(model_config, "_original_model", None)

            # If not available, try reading from metadata file
            if not original_oci_ref:
                metadata_path = os.path.join(cache_dir, "oci_metadata.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path) as f:
                        metadata = json.load(f)
                        original_oci_ref = metadata.get("original_reference")
                        if original_oci_ref:
                            logger.info(
                                "Retrieved original OCI reference from metadata: %s",
                                original_oci_ref,
                            )
        else:
            # It's an OCI reference, normalize and get cache dir
            normalized_ref = self._normalize_oci_reference(model_ref)
            cache_dir = self._get_cache_dir(normalized_ref)
            original_oci_ref = model_ref

        # Load manifest
        manifest_path = os.path.join(cache_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            raise ValueError(
                f"Manifest not found at {manifest_path}. "
                f"Cache dir: {cache_dir}, Model ref: {model_ref}"
            )

        with open(manifest_path) as f:
            manifest = json.load(f)

        # Get safetensors layers in order
        layers_dir = os.path.join(cache_dir, "layers")
        safetensors_files = []
        safetensors_layers = [
            layer
            for layer in manifest.get("layers", [])
            if layer.get("mediaType") == self.SAFETENSORS_MEDIA_TYPE
        ]

        # Check if weights need to be downloaded
        weights_missing = False
        if not os.path.exists(layers_dir):
            weights_missing = True
        else:
            for layer in safetensors_layers:
                digest = layer.get("digest", "").replace("sha256:", "")
                # Check if any matching file exists
                found = False
                for filename in os.listdir(layers_dir):
                    if digest in filename and filename.endswith(".safetensors"):
                        found = True
                        break
                if not found:
                    weights_missing = True
                    break

        # Download weights if missing and we have a valid OCI reference
        if weights_missing:
            if not original_oci_ref:
                raise ValueError(
                    f"Weights not found in cache at {layers_dir}, but cannot "
                    f"download them because the original OCI reference is not "
                    f"available. Model ref: {model_ref}"
                )
            logger.info("Weights not found in cache, downloading now...")
            self._download_oci_model_if_needed(original_oci_ref, download_weights=True)

        # Now collect safetensors files
        for layer in safetensors_layers:
            digest = layer.get("digest", "").replace("sha256:", "")
            # Find matching file
            for filename in sorted(os.listdir(layers_dir)):
                if digest in filename and filename.endswith(".safetensors"):
                    safetensors_files.append(os.path.join(layers_dir, filename))
                    break

        if not safetensors_files:
            raise ValueError(f"No safetensors files found in {layers_dir}")

        logger.info(
            "Loading weights from %d safetensors file(s)", len(safetensors_files)
        )

        # Use existing safetensors iterator
        yield from safetensors_weights_iterator(
            safetensors_files,
            use_tqdm_on_load=self.load_config.use_tqdm_on_load,
            safetensors_load_strategy=(self.load_config.safetensors_load_strategy),
        )

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
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

        # Load weights using iterator
        weights_to_load = {name for name, _ in model.named_parameters()}
        loaded_weights = model.load_weights(self._get_weights_iterator(model_config))

        # Check if all weights were loaded (for non-quantized models)
        if model_config.quantization is None and loaded_weights is not None:
            weights_not_loaded = weights_to_load - loaded_weights
            if weights_not_loaded:
                raise ValueError(
                    "Following weights were not initialized from "
                    f"checkpoint: {weights_not_loaded}"
                )

        logger.info("Weights loaded successfully from OCI registry")
