# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
from unittest.mock import AsyncMock

from aiq.builder.builder import Builder
from aiq.builder.embedder import EmbedderProviderInfo
from aiq.embedder.dev_genai_embedder import DevGenAIEmbedderModelConfig
from aiq.embedder.dev_genai_embedder import dev_genai_embedder


class TestDevGenAIEmbedder:
    """Test cases for Dev GenAI embedder integration."""

    def test_dev_genai_embedder_config_creation(self):
        """Test creating a Dev GenAI embedder configuration with default values."""
        config = DevGenAIEmbedderModelConfig()
        
        assert config.model_name == "dev-genai-embed"
        assert config.base_url == "https://api.dev-genai.com/v1"
        assert config.dimensions is None
        assert config.max_retries == 3
        assert config.timeout == 60
        assert config.batch_size == 100
        assert config.encoding_format == "float"
        assert config.truncate_input is True
        assert config.api_key is None

    def test_dev_genai_embedder_config_custom_values(self):
        """Test creating a Dev GenAI embedder configuration with custom values."""
        config = DevGenAIEmbedderModelConfig(
            api_key="test-key",
            base_url="https://custom.api.com/v1",
            model_name="dev-genai-embed-large",
            dimensions=768,
            max_retries=5,
            timeout=30,
            batch_size=50,
            encoding_format="base64",
            truncate_input=False
        )
        
        assert config.api_key == "test-key"
        assert config.base_url == "https://custom.api.com/v1"
        assert config.model_name == "dev-genai-embed-large"
        assert config.dimensions == 768
        assert config.max_retries == 5
        assert config.timeout == 30
        assert config.batch_size == 50
        assert config.encoding_format == "base64"
        assert config.truncate_input is False

    def test_dev_genai_embedder_config_validation(self):
        """Test configuration validation."""
        # Test max_retries bounds
        with pytest.raises(ValueError):
            DevGenAIEmbedderModelConfig(max_retries=-1)
        
        # Test timeout bounds
        with pytest.raises(ValueError):
            DevGenAIEmbedderModelConfig(timeout=0)
        
        # Test batch_size bounds
        with pytest.raises(ValueError):
            DevGenAIEmbedderModelConfig(batch_size=0)
        
        # Test valid values
        config = DevGenAIEmbedderModelConfig(
            max_retries=0,
            timeout=1,
            batch_size=1
        )
        assert config.max_retries == 0
        assert config.timeout == 1
        assert config.batch_size == 1

    def test_dev_genai_embedder_config_alias_support(self):
        """Test that model_name supports alias."""
        config = DevGenAIEmbedderModelConfig(model="custom-embed-model")
        assert config.model_name == "custom-embed-model"

    def test_dev_genai_embedder_config_static_type(self):
        """Test that configuration returns correct static type."""
        config = DevGenAIEmbedderModelConfig()
        assert config.static_type() == "dev_genai"

    @pytest.mark.asyncio
    async def test_dev_genai_embedder_provider_registration(self):
        """Test that the Dev GenAI embedder provider can be registered."""
        config = DevGenAIEmbedderModelConfig(api_key="test-key")
        builder = AsyncMock(spec=Builder)
        
        # Test the registration function
        provider_gen = dev_genai_embedder(config, builder)
        provider_info = await provider_gen.__anext__()
        
        assert isinstance(provider_info, EmbedderProviderInfo)
        assert provider_info.config == config
        assert provider_info.provider_type == "dev_genai"
        assert "Dev GenAI model for use with an embedder client" in provider_info.description

    def test_dev_genai_embedder_config_extra_fields(self):
        """Test that extra fields are allowed in configuration."""
        config = DevGenAIEmbedderModelConfig(
            custom_field="custom_value",
            another_field=42
        )
        
        assert hasattr(config, 'custom_field')
        assert config.custom_field == "custom_value"
        assert hasattr(config, 'another_field')
        assert config.another_field == 42

    def test_dev_genai_embedder_config_dimensions_optional(self):
        """Test that dimensions parameter is optional."""
        config = DevGenAIEmbedderModelConfig()
        assert config.dimensions is None
        
        config = DevGenAIEmbedderModelConfig(dimensions=512)
        assert config.dimensions == 512

    def test_dev_genai_embedder_config_encoding_formats(self):
        """Test different encoding format options."""
        config = DevGenAIEmbedderModelConfig(encoding_format="float")
        assert config.encoding_format == "float"
        
        config = DevGenAIEmbedderModelConfig(encoding_format="base64")
        assert config.encoding_format == "base64"

    def test_dev_genai_embedder_config_truncate_input_flag(self):
        """Test truncate_input flag functionality."""
        # Default should be True
        config = DevGenAIEmbedderModelConfig()
        assert config.truncate_input is True
        
        # Test explicit False
        config = DevGenAIEmbedderModelConfig(truncate_input=False)
        assert config.truncate_input is False

    def test_dev_genai_embedder_config_batch_size_validation(self):
        """Test batch size validation."""
        config = DevGenAIEmbedderModelConfig(batch_size=1)
        assert config.batch_size == 1
        
        config = DevGenAIEmbedderModelConfig(batch_size=1000)
        assert config.batch_size == 1000
        
        # Test that batch_size must be positive
        with pytest.raises(ValueError):
            DevGenAIEmbedderModelConfig(batch_size=0)

    def test_dev_genai_embedder_config_model_dump(self):
        """Test model dumping functionality."""
        config = DevGenAIEmbedderModelConfig(
            api_key="test-key",
            model_name="test-model",
            dimensions=256
        )
        
        # Test model dump with exclusions
        dumped = config.model_dump(exclude={"type"}, by_alias=True)
        assert "type" not in dumped
        assert dumped["model"] == "test-model"  # by_alias=True uses serialization_alias
        assert dumped["api_key"] == "test-key"
        assert dumped["dimensions"] == 256