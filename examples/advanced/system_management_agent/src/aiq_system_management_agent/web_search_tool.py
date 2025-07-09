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

import asyncio
import logging
from typing import List

import aiofiles
import httpx
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from pydantic import Field

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig
from aiq.profiler.decorators.function_tracking import track_function

logger = logging.getLogger(__name__)


class WebSearchToolConfig(FunctionBaseConfig, name="web_search_tool"):
    """Configuration for the Web Search Tool."""
    
    allowed_urls: List[str] = Field(
        default_factory=lambda: [
            "https://docs.nvidia.com",
            "https://developer.nvidia.com",
            "https://www.redhat.com",
            "https://docs.redhat.com",
            "https://access.redhat.com",
        ],
        description="List of allowed URLs/domains for web search"
    )
    max_content_length: int = Field(
        default=10000,
        description="Maximum content length to extract from web pages"
    )
    timeout: int = Field(
        default=30,
        description="Request timeout in seconds"
    )
    max_retries: int = Field(
        default=3,
        description="Maximum number of retry attempts"
    )


@register_function(config_type=WebSearchToolConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def web_search_tool(config: WebSearchToolConfig, builder: Builder):
    """
    Web search tool that can fetch and extract content from specific URLs.
    """
    
    @track_function()
    async def _fetch_url_content(url: str) -> str:
        """Fetch content from a specific URL."""
        try:
            # Check if URL is allowed
            if not any(allowed_url in url for allowed_url in config.allowed_urls):
                return f"Error: URL '{url}' is not in the allowed domains: {config.allowed_urls}"
            
            async with httpx.AsyncClient(timeout=config.timeout) as client:
                for attempt in range(config.max_retries):
                    try:
                        response = await client.get(url, follow_redirects=True)
                        response.raise_for_status()
                        break
                    except httpx.RequestError as e:
                        if attempt == config.max_retries - 1:
                            return f"Error fetching {url}: {str(e)}"
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
                # Parse content with BeautifulSoup
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # Extract text content
                text = soup.get_text()
                
                # Clean up text
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)
                
                # Truncate if too long
                if len(text) > config.max_content_length:
                    text = text[:config.max_content_length] + "... [truncated]"
                
                return f"Content from {url}:\n{text}"
                
        except Exception as e:
            logger.error(f"Error fetching content from {url}: {str(e)}")
            return f"Error fetching content from {url}: {str(e)}"
    
    @track_function()
    async def _search_multiple_urls(urls: List[str]) -> str:
        """Search multiple URLs concurrently."""
        tasks = [_fetch_url_content(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        formatted_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                formatted_results.append(f"Error processing {urls[i]}: {str(result)}")
            else:
                formatted_results.append(str(result))
        
        return "\n\n" + "="*50 + "\n\n".join(formatted_results)
    
    @tool
    async def search_url(url: str) -> str:
        """
        Search and extract content from a specific URL.
        
        Args:
            url: The URL to fetch content from
            
        Returns:
            The extracted text content from the URL
        """
        return await _fetch_url_content(url)
    
    @tool
    async def search_multiple_urls(urls: str) -> str:
        """
        Search and extract content from multiple URLs.
        
        Args:
            urls: Comma-separated list of URLs to search
            
        Returns:
            Combined content from all URLs
        """
        url_list = [url.strip() for url in urls.split(",")]
        return await _search_multiple_urls(url_list)
    
    @tool
    async def search_documentation(query: str) -> str:
        """
        Search for system management documentation across allowed domains.
        
        Args:
            query: Search query or topic to look for
            
        Returns:
            Relevant documentation content
        """
        # Construct search URLs based on query
        search_urls = []
        
        # Add NVIDIA documentation URLs
        if "nvidia" in query.lower() or "gpu" in query.lower():
            search_urls.extend([
                "https://docs.nvidia.com/datacenter/tesla/",
                "https://developer.nvidia.com/blog/",
            ])
        
        # Add RedHat documentation URLs
        if "redhat" in query.lower() or "rhel" in query.lower() or "system" in query.lower():
            search_urls.extend([
                "https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/9",
                "https://access.redhat.com/documentation/",
            ])
        
        if not search_urls:
            return f"No relevant documentation URLs found for query: {query}"
        
        return await _search_multiple_urls(search_urls)
    
    # Return the tools
    yield [search_url, search_multiple_urls, search_documentation]