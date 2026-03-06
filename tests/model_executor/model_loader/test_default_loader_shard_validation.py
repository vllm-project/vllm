# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import tempfile
from unittest.mock import Mock

import pytest

from vllm.config import LoadConfig, ModelConfig
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader


class TestDefaultLoaderShardValidation:
    """Test shard validation for quantized models in DefaultModelLoader."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.load_config = LoadConfig()
        self.loader = DefaultModelLoader(self.load_config)
    
    def test_validate_shard_completeness_safetensors_all_present(self):
        """Test validation passes when all safetensors shards are present."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a mock safetensors index file
            index_file = os.path.join(temp_dir, "model.safetensors.index.json")
            weight_map = {
                "layer.0.weight": "model-00001-of-00002.safetensors",
                "layer.1.weight": "model-00002-of-00002.safetensors",
            }
            index_data = {"weight_map": weight_map}
            
            with open(index_file, 'w') as f:
                json.dump(index_data, f)
            
            # Create the expected shard files
            shard1 = os.path.join(temp_dir, "model-00001-of-00002.safetensors")
            shard2 = os.path.join(temp_dir, "model-00002-of-00002.safetensors")
            
            with open(shard1, 'w') as f:
                f.write("dummy content")
            with open(shard2, 'w') as f:
                f.write("dummy content")
            
            # This should not raise an exception
            hf_weights_files = [shard1, shard2]
            self.loader._validate_shard_completeness(temp_dir, hf_weights_files, True)
    
    def test_validate_shard_completeness_safetensors_missing_shard(self):
        """Test validation fails when a safetensors shard is missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a mock safetensors index file
            index_file = os.path.join(temp_dir, "model.safetensors.index.json")
            weight_map = {
                "layer.0.weight": "model-00001-of-00002.safetensors",
                "layer.1.weight": "model-00002-of-00002.safetensors",
            }
            index_data = {"weight_map": weight_map}
            
            with open(index_file, 'w') as f:
                json.dump(index_data, f)
            
            # Create only one of the expected shard files
            shard1 = os.path.join(temp_dir, "model-00001-of-00002.safetensors")
            with open(shard1, 'w') as f:
                f.write("dummy content")
            # shard2 is missing
            
            hf_weights_files = [shard1]
            
            # This should raise a ValueError about missing shards
            with pytest.raises(ValueError, match="Missing safetensors shard files"):
                self.loader._validate_shard_completeness(temp_dir, hf_weights_files, True)
    
    def test_validate_shard_completeness_no_index_file(self):
        """Test validation passes when no index file exists (non-sharded model)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # No index file created
            hf_weights_files = []
            
            # This should not raise an exception for safetensors without index
            self.loader._validate_shard_completeness(temp_dir, hf_weights_files, True)
    
    def test_validate_shard_completeness_pt_files_empty(self):
        """Test validation fails when no .pt/.bin files are present."""
        with tempfile.TemporaryDirectory() as temp_dir:
            hf_weights_files = []
            
            # This should raise a ValueError for empty weight files list
            with pytest.raises(ValueError, match="No weight files found"):
                self.loader._validate_shard_completeness(temp_dir, hf_weights_files, False)
    
    def test_validate_shard_completeness_pt_files_present(self):
        """Test validation passes when .pt/.bin files are present."""
        with tempfile.TemporaryDirectory() as temp_dir:
            hf_weights_files = [
                os.path.join(temp_dir, "pytorch_model.bin"),
                os.path.join(temp_dir, "pytorch_model_2.bin"),
            ]
            
            # This should not raise an exception
            self.loader._validate_shard_completeness(temp_dir, hf_weights_files, False)