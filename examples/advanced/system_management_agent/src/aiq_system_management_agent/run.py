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

"""
Simple run script for the System Management Agent.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add the src directory to the path so we can import the agent
sys.path.insert(0, str(Path(__file__).parent.parent))

from aiq.builder.workflow_builder import WorkflowBuilder
from aiq.llm.dev_genai_llm import DevGenAIModelConfig
from aiq_system_management_agent.register import SystemManagementAgentConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Main function to run the System Management Agent."""
    
    # Check for API key
    api_key = os.getenv("DEV_GENAI_API_KEY")
    if not api_key:
        logger.error("DEV_GENAI_API_KEY environment variable not set")
        logger.info("Please set your Dev GenAI API key:")
        logger.info("export DEV_GENAI_API_KEY='your-api-key-here'")
        return
    
    # Configure the LLM
    llm_config = DevGenAIModelConfig(
        api_key=api_key,
        model_name="dev-genai-chat",
        temperature=0.3,
        max_tokens=2048
    )
    
    # Configure the agent
    agent_config = SystemManagementAgentConfig(
        llm_name="dev_genai_llm",
        verbose=True
    )
    
    # Example queries to demonstrate capabilities
    queries = [
        "Get the health status of all managed systems",
        "Search for NVIDIA GPU thermal management documentation",
        "Load any available PDF manuals and list them",
        "Check the power status of all systems and provide recommendations"
    ]
    
    try:
        # Create the workflow
        async with WorkflowBuilder() as builder:
            # Add LLM
            await builder.add_llm("dev_genai_llm", llm_config)
            
            # Add tools
            await builder.add_function("web_search_tool", {"_type": "web_search_tool"})
            await builder.add_function("pdf_embeddings_tool", {"_type": "pdf_embeddings_tool"})
            await builder.add_function("redfish_api_tool", {"_type": "redfish_api_tool"})
            
            # Add the main agent function
            await builder.add_function("system_management_agent", agent_config)
            
            # Get the agent function
            agent_function = await builder.get_function("system_management_agent")
            
            logger.info("System Management Agent initialized successfully!")
            logger.info("Available capabilities:")
            logger.info("- Web search for documentation")
            logger.info("- PDF document analysis")
            logger.info("- Redfish API integration (mock mode)")
            logger.info("")
            
            # Run example queries
            for i, query in enumerate(queries, 1):
                logger.info(f"Running example query {i}: {query}")
                logger.info("-" * 60)
                
                try:
                    result = await agent_function(query)
                    logger.info(f"Result: {result}")
                except Exception as e:
                    logger.error(f"Error processing query: {e}")
                
                logger.info("=" * 60)
                logger.info("")
            
            # Interactive mode
            logger.info("Entering interactive mode. Type 'quit' to exit.")
            while True:
                try:
                    user_input = input("\nEnter your system management query: ").strip()
                    if user_input.lower() in ['quit', 'exit', 'q']:
                        break
                    
                    if user_input:
                        logger.info("Processing your query...")
                        result = await agent_function(user_input)
                        print(f"\nResult: {result}")
                    
                except KeyboardInterrupt:
                    logger.info("\nExiting...")
                    break
                except Exception as e:
                    logger.error(f"Error: {e}")
    
    except Exception as e:
        logger.error(f"Failed to initialize System Management Agent: {e}")
        logger.info("Make sure all dependencies are installed and the API key is valid")


if __name__ == "__main__":
    asyncio.run(main())