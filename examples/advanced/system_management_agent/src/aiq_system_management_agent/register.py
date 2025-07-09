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

# pylint: disable=unused-import
# flake8: noqa

import logging

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langgraph.graph import START
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from pydantic.fields import Field

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.cli.register_workflow import register_function
from aiq.data_models.component_ref import LLMRef
from aiq.data_models.function import FunctionBaseConfig
from aiq.profiler.decorators.function_tracking import track_function

# Import the tools
from . import pdf_embeddings_tool
from . import redfish_api_tool
from . import web_search_tool
from .prompts import SYSTEM_MANAGEMENT_AGENT_PROMPT

logger = logging.getLogger(__name__)


class SystemManagementAgentConfig(FunctionBaseConfig, name="system_management_agent"):
    """
    Configuration for the System Management Agent workflow.
    
    This agent provides comprehensive system management capabilities including:
    1. Web search for documentation and troubleshooting information
    2. PDF document analysis for manuals and guides
    3. Redfish API integration for hardware monitoring
    """
    
    tool_names: list[str] = Field(
        default_factory=lambda: [
            "web_search_tool",
            "pdf_embeddings_tool", 
            "redfish_api_tool"
        ],
        description="List of tools available to the agent"
    )
    llm_name: LLMRef = Field(
        description="LLM to use for the agent"
    )
    verbose: bool = Field(
        default=True,
        description="Whether to enable verbose logging"
    )
    max_iterations: int = Field(
        default=10,
        description="Maximum number of iterations for the agent"
    )


@register_function(config_type=SystemManagementAgentConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def system_management_agent(config: SystemManagementAgentConfig, builder: Builder):
    """
    System Management Agent workflow that orchestrates multiple tools for system administration.
    """
    
    # Get the LLM
    llm: BaseChatModel = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    
    # Get all tools specified in config
    tools = []
    for tool_name in config.tool_names:
        tool_functions = builder.get_tool(tool_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
        if isinstance(tool_functions, list):
            tools.extend(tool_functions)
        else:
            tools.append(tool_functions)
    
    # Bind tools to the LLM
    llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=True)
    
    # Define the agent function
    async def agent_function(state: MessagesState):
        """Main agent function that processes messages with the LLM."""
        if config.verbose:
            logger.info(f"Agent processing {len(state['messages'])} messages")
        
        # Create system message with the agent prompt
        system_message = SystemMessage(content=SYSTEM_MANAGEMENT_AGENT_PROMPT)
        
        # Invoke LLM with system message and conversation history
        response = await llm_with_tools.ainvoke([system_message] + state["messages"])
        
        return {"messages": [response]}
    
    # Create the state graph
    graph_builder = StateGraph(MessagesState)
    
    # Add nodes
    graph_builder.add_node("agent", agent_function)
    graph_builder.add_node("tools", ToolNode(tools))
    
    # Add edges
    graph_builder.add_edge(START, "agent")
    graph_builder.add_conditional_edges(
        "agent",
        tools_condition,
    )
    graph_builder.add_edge("tools", "agent")
    
    # Compile the graph
    agent_executor = graph_builder.compile()
    
    @track_function()
    async def _process_query(input_message: str) -> str:
        """Process a system management query."""
        try:
            if config.verbose:
                logger.info(f"Processing query: {input_message[:100]}...")
            
            # Execute the agent
            result = await agent_executor.ainvoke(
                {"messages": [HumanMessage(content=input_message)]},
                config={"recursion_limit": config.max_iterations}
            )
            
            # Extract the final response
            final_message = result["messages"][-1].content
            
            if config.verbose:
                logger.info("Query processed successfully")
            
            return final_message
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    async def _response_function(input_message: str) -> str:
        """Main response function for the agent."""
        try:
            return await _process_query(input_message)
        except Exception as e:
            logger.error(f"System Management Agent error: {str(e)}")
            return f"System Management Agent encountered an error: {str(e)}"
        finally:
            if config.verbose:
                logger.info("System Management Agent execution completed")
    
    # Initialize PDF embeddings if directory exists
    try:
        pdf_tool = builder.get_tool("pdf_embeddings_tool", wrapper_type=LLMFrameworkEnum.LANGCHAIN)
        if isinstance(pdf_tool, list):
            # Find the load_pdf_embeddings function
            for tool in pdf_tool:
                if hasattr(tool, 'name') and tool.name == 'load_pdf_embeddings':
                    await tool.arun("")
                    break
    except Exception as e:
        logger.warning(f"Could not initialize PDF embeddings: {str(e)}")
    
    yield _response_function