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
from unittest.mock import AsyncMock, MagicMock

from aiq.builder.workflow_builder import WorkflowBuilder
from aiq_system_management_agent.register import SystemManagementAgentConfig


class TestSystemManagementAgent:
    """Test cases for the System Management Agent workflow."""

    def test_system_management_agent_config_creation(self):
        """Test creating a system management agent configuration."""
        config = SystemManagementAgentConfig(llm_name="test_llm")
        
        assert config.llm_name == "test_llm"
        assert "web_search_tool" in config.tool_names
        assert "pdf_embeddings_tool" in config.tool_names
        assert "redfish_api_tool" in config.tool_names
        assert config.verbose is True
        assert config.max_iterations == 10

    def test_system_management_agent_config_custom_values(self):
        """Test creating a system management agent configuration with custom values."""
        config = SystemManagementAgentConfig(
            llm_name="custom_llm",
            tool_names=["web_search_tool"],
            verbose=False,
            max_iterations=5
        )
        
        assert config.llm_name == "custom_llm"
        assert config.tool_names == ["web_search_tool"]
        assert config.verbose is False
        assert config.max_iterations == 5

    def test_system_management_agent_config_static_type(self):
        """Test that configuration returns correct static type."""
        config = SystemManagementAgentConfig(llm_name="test_llm")
        assert config.static_type() == "system_management_agent"

    @pytest.mark.asyncio
    async def test_system_management_agent_initialization(self):
        """Test that the system management agent can be initialized."""
        config = SystemManagementAgentConfig(llm_name="test_llm")
        
        # Mock the builder
        builder = AsyncMock()
        mock_llm = MagicMock()
        builder.get_llm.return_value = mock_llm
        
        # Mock tools
        mock_tools = [MagicMock(), MagicMock(), MagicMock()]
        builder.get_tool.return_value = mock_tools
        
        # Import and test the registration function
        from aiq_system_management_agent.register import system_management_agent
        
        # Test that the function can be called
        agent_gen = system_management_agent(config, builder)
        agent_function = await agent_gen.__anext__()
        
        # Verify it returns a callable function
        assert callable(agent_function)
        
        # Verify builder methods were called
        builder.get_llm.assert_called_once()
        assert builder.get_tool.call_count == len(config.tool_names)

    @pytest.mark.asyncio
    async def test_system_management_agent_with_workflow_builder(self):
        """Test system management agent integration with WorkflowBuilder."""
        # This test would require actual LLM and tool registrations
        # For now, we'll just test the config integration
        
        config = SystemManagementAgentConfig(llm_name="test_llm")
        
        # Test that config can be used with workflow builder pattern
        assert hasattr(config, 'llm_name')
        assert hasattr(config, 'tool_names')
        assert hasattr(config, 'verbose')
        assert hasattr(config, 'max_iterations')

    def test_tool_names_validation(self):
        """Test that tool names are properly validated."""
        config = SystemManagementAgentConfig(
            llm_name="test_llm",
            tool_names=[]
        )
        
        # Should allow empty tool names
        assert config.tool_names == []
        
        # Test with custom tool names
        custom_tools = ["custom_tool_1", "custom_tool_2"]
        config = SystemManagementAgentConfig(
            llm_name="test_llm",
            tool_names=custom_tools
        )
        
        assert config.tool_names == custom_tools

    def test_max_iterations_validation(self):
        """Test that max_iterations accepts valid values."""
        config = SystemManagementAgentConfig(
            llm_name="test_llm",
            max_iterations=1
        )
        assert config.max_iterations == 1
        
        config = SystemManagementAgentConfig(
            llm_name="test_llm",
            max_iterations=100
        )
        assert config.max_iterations == 100

    def test_verbose_flag(self):
        """Test verbose flag functionality."""
        # Default should be True
        config = SystemManagementAgentConfig(llm_name="test_llm")
        assert config.verbose is True
        
        # Test explicit False
        config = SystemManagementAgentConfig(
            llm_name="test_llm",
            verbose=False
        )
        assert config.verbose is False