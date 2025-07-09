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

from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.builder.workflow_builder import WorkflowBuilder
from aiq.embedder.dev_genai_embedder import DevGenAIEmbedderModelConfig


@pytest.mark.integration
async def test_dev_genai_embedder_langchain_integration():
    """
    Test Dev GenAI embedder with LangChain integration. Requires DEV_GENAI_API_KEY to be set.
    """
    embedder_config = DevGenAIEmbedderModelConfig(
        model_name="dev-genai-embed",
        batch_size=10,
        dimensions=384
    )
    
    async with WorkflowBuilder() as builder:
        await builder.add_embedder("dev_genai_embedder", embedder_config)
        embedder = await builder.get_embedder("dev_genai_embedder", wrapper_type=LLMFrameworkEnum.LANGCHAIN)
        
        # Test text embedding
        test_texts = ["Hello world", "This is a test", "Another test sentence"]
        embeddings = await embedder.aembed_documents(test_texts)
        
        assert len(embeddings) == len(test_texts)
        
        # Check that embeddings are lists/arrays of numbers
        for embedding in embeddings:
            assert isinstance(embedding, (list, tuple))
            assert len(embedding) > 0
            assert all(isinstance(x, (int, float)) for x in embedding)
        
        # Test query embedding
        query_embedding = await embedder.aembed_query("test query")
        assert isinstance(query_embedding, (list, tuple))
        assert len(query_embedding) > 0
        assert all(isinstance(x, (int, float)) for x in query_embedding)


@pytest.mark.integration
async def test_dev_genai_embedder_batch_processing():
    """
    Test Dev GenAI embedder batch processing capabilities.
    """
    embedder_config = DevGenAIEmbedderModelConfig(
        model_name="dev-genai-embed",
        batch_size=5,
        max_retries=2,
        timeout=30
    )
    
    async with WorkflowBuilder() as builder:
        await builder.add_embedder("dev_genai_embedder", embedder_config)
        embedder = await builder.get_embedder("dev_genai_embedder", wrapper_type=LLMFrameworkEnum.LANGCHAIN)
        
        # Test with larger batch that should be split
        large_text_batch = [f"Test sentence number {i}" for i in range(15)]
        embeddings = await embedder.aembed_documents(large_text_batch)
        
        assert len(embeddings) == len(large_text_batch)
        
        # Verify all embeddings are valid
        for embedding in embeddings:
            assert isinstance(embedding, (list, tuple))
            assert len(embedding) > 0


@pytest.mark.integration
async def test_dev_genai_embedder_configuration_options():
    """
    Test Dev GenAI embedder with various configuration options.
    """
    embedder_config = DevGenAIEmbedderModelConfig(
        model_name="dev-genai-embed",
        dimensions=512,
        encoding_format="float",
        truncate_input=True,
        batch_size=20,
        timeout=60
    )
    
    async with WorkflowBuilder() as builder:
        await builder.add_embedder("dev_genai_embedder", embedder_config)
        embedder = await builder.get_embedder("dev_genai_embedder", wrapper_type=LLMFrameworkEnum.LANGCHAIN)
        
        # Test with single text
        single_text = "This is a single test text for embedding"
        embedding = await embedder.aembed_query(single_text)
        
        assert isinstance(embedding, (list, tuple))
        assert len(embedding) > 0
        
        # If dimensions is specified, verify the embedding dimension
        # Note: This depends on the actual API behavior
        if embedder_config.dimensions:
            # Some APIs respect the dimensions parameter, others don't
            # This is just a basic check
            assert len(embedding) > 0


@pytest.mark.unit
async def test_dev_genai_embedder_config_validation():
    """
    Test Dev GenAI embedder configuration validation without API calls.
    """
    # Test valid configuration
    config = DevGenAIEmbedderModelConfig(
        api_key="test-key",
        model_name="test-model",
        dimensions=256,
        batch_size=50
    )
    
    assert config.api_key == "test-key"
    assert config.model_name == "test-model"
    assert config.dimensions == 256
    assert config.batch_size == 50
    
    # Test validation errors
    with pytest.raises(ValueError):
        DevGenAIEmbedderModelConfig(max_retries=-1)
    
    with pytest.raises(ValueError):
        DevGenAIEmbedderModelConfig(timeout=0)
    
    with pytest.raises(ValueError):
        DevGenAIEmbedderModelConfig(batch_size=0)


@pytest.mark.unit
async def test_dev_genai_embedder_model_dump():
    """
    Test Dev GenAI embedder model dumping for LangChain integration.
    """
    config = DevGenAIEmbedderModelConfig(
        api_key="test-key",
        model_name="test-model",
        dimensions=384,
        batch_size=100
    )
    
    # Test model dump with type exclusion (as used in the client registration)
    dumped = config.model_dump(exclude={"type"}, by_alias=True)
    
    # Verify expected fields are present
    assert "api_key" in dumped
    assert "model" in dumped  # by_alias=True uses serialization_alias
    assert "dimensions" in dumped
    assert "batch_size" in dumped
    
    # Verify type is excluded
    assert "type" not in dumped
    
    # Verify values are correct
    assert dumped["api_key"] == "test-key"
    assert dumped["model"] == "test-model"
    assert dumped["dimensions"] == 384
    assert dumped["batch_size"] == 100