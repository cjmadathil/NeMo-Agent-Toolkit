# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from aiq.builder.llm import LLMProviderInfo
from aiq.llm.dev_genai_llm import DevGenAIModelConfig
from aiq.llm.dev_genai_llm import dev_genai_llm


class TestDevGenAILLM:
    """Test cases for Dev GenAI LLM integration."""

    def test_dev_genai_config_creation(self):
        """Test creating a Dev GenAI configuration with default values."""
        config = DevGenAIModelConfig()
        
        assert config.model_name == "dev-genai-chat"
        assert config.base_url == "https://api.dev-genai.com/v1"
        assert config.temperature == 0.0
        assert config.max_tokens == 1024
        assert config.timeout == 60
        assert config.max_retries == 3
        assert config.streaming is False
        assert config.top_p == 1.0
        assert config.frequency_penalty == 0.0
        assert config.presence_penalty == 0.0
        assert config.stop is None
        assert config.api_key is None

    def test_dev_genai_config_with_custom_values(self):
        """Test creating a Dev GenAI configuration with custom values."""
        config = DevGenAIModelConfig(
            api_key="test-key",
            base_url="https://custom.api.com/v1",
            model_name="dev-genai-text",
            temperature=0.8,
            max_tokens=2048,
            timeout=30,
            max_retries=5,
            streaming=True,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.2,
            stop=["<end>", "<stop>"]
        )
        
        assert config.api_key == "test-key"
        assert config.base_url == "https://custom.api.com/v1"
        assert config.model_name == "dev-genai-text"
        assert config.temperature == 0.8
        assert config.max_tokens == 2048
        assert config.timeout == 30
        assert config.max_retries == 5
        assert config.streaming is True
        assert config.top_p == 0.9
        assert config.frequency_penalty == 0.1
        assert config.presence_penalty == 0.2
        assert config.stop == ["<end>", "<stop>"]

    def test_dev_genai_config_validation(self):
        """Test configuration validation."""
        # Test temperature bounds
        with pytest.raises(ValueError):
            DevGenAIModelConfig(temperature=-0.1)
        
        with pytest.raises(ValueError):
            DevGenAIModelConfig(temperature=2.1)
        
        # Test top_p bounds
        with pytest.raises(ValueError):
            DevGenAIModelConfig(top_p=-0.1)
        
        with pytest.raises(ValueError):
            DevGenAIModelConfig(top_p=1.1)
        
        # Test penalty bounds
        with pytest.raises(ValueError):
            DevGenAIModelConfig(frequency_penalty=-2.1)
        
        with pytest.raises(ValueError):
            DevGenAIModelConfig(presence_penalty=2.1)
        
        # Test positive integers
        with pytest.raises(ValueError):
            DevGenAIModelConfig(max_tokens=0)
        
        with pytest.raises(ValueError):
            DevGenAIModelConfig(timeout=0)
        
        with pytest.raises(ValueError):
            DevGenAIModelConfig(max_retries=-1)

    def test_dev_genai_config_alias_support(self):
        """Test that model_name supports alias."""
        config = DevGenAIModelConfig(model="custom-model")
        assert config.model_name == "custom-model"

    def test_dev_genai_config_static_type(self):
        """Test that configuration returns correct static type."""
        config = DevGenAIModelConfig()
        assert config.static_type() == "dev_genai"

    @pytest.mark.asyncio
    async def test_dev_genai_llm_provider_registration(self):
        """Test that the Dev GenAI LLM provider can be registered."""
        config = DevGenAIModelConfig(api_key="test-key")
        builder = AsyncMock(spec=Builder)
        
        # Test the registration function
        provider_gen = dev_genai_llm(config, builder)
        provider_info = await provider_gen.__anext__()
        
        assert isinstance(provider_info, LLMProviderInfo)
        assert provider_info.config == config
        assert provider_info.provider_type == "dev_genai"
        assert "Dev GenAI model for use with an LLM client" in provider_info.description

    def test_dev_genai_config_extra_fields(self):
        """Test that extra fields are allowed in configuration."""
        config = DevGenAIModelConfig(
            custom_field="custom_value",
            another_field=42
        )
        
        assert hasattr(config, 'custom_field')
        assert config.custom_field == "custom_value"
        assert hasattr(config, 'another_field')
        assert config.another_field == 42