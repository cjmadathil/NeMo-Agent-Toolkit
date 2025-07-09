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

from pydantic import AliasChoices
from pydantic import ConfigDict
from pydantic import Field
from pydantic import PositiveInt

from aiq.builder.builder import Builder
from aiq.builder.llm import LLMProviderInfo
from aiq.cli.register_workflow import register_llm_provider
from aiq.data_models.llm import LLMBaseConfig


class DevGenAIModelConfig(LLMBaseConfig, name="dev_genai"):
    """A Dev GenAI LLM provider to be used with an LLM client."""

    model_config = ConfigDict(protected_namespaces=(), extra="allow")

    api_key: str | None = Field(
        default=None, 
        description="Dev GenAI API key to interact with hosted model. Can also be set via DEV_GENAI_API_KEY environment variable."
    )
    base_url: str = Field(
        default="https://api.dev-genai.com/v1", 
        description="Base URL for the Dev GenAI API endpoint."
    )
    model_name: str = Field(
        default="dev-genai-chat",
        validation_alias=AliasChoices("model_name", "model"),
        serialization_alias="model",
        description="The Dev GenAI model name (e.g., 'dev-genai-chat', 'dev-genai-text')."
    )
    temperature: float = Field(
        default=0.0, 
        ge=0.0, 
        le=2.0, 
        description="Sampling temperature in [0, 2]."
    )
    max_tokens: PositiveInt = Field(
        default=1024, 
        description="Maximum number of tokens to generate."
    )
    timeout: int = Field(
        default=60, 
        ge=1, 
        description="Request timeout in seconds."
    )
    max_retries: int = Field(
        default=3, 
        ge=0, 
        description="Maximum number of retry attempts."
    )
    streaming: bool = Field(
        default=False, 
        description="Enable streaming responses."
    )
    top_p: float = Field(
        default=1.0, 
        ge=0.0, 
        le=1.0, 
        description="Top-p for nucleus sampling."
    )
    frequency_penalty: float = Field(
        default=0.0, 
        ge=-2.0, 
        le=2.0, 
        description="Frequency penalty for token repetition."
    )
    presence_penalty: float = Field(
        default=0.0, 
        ge=-2.0, 
        le=2.0, 
        description="Presence penalty for token usage."
    )
    stop: list[str] | None = Field(
        default=None, 
        description="List of stop sequences to terminate generation."
    )


@register_llm_provider(config_type=DevGenAIModelConfig)
async def dev_genai_llm(config: DevGenAIModelConfig, builder: Builder):
    """Register Dev GenAI LLM provider."""
    
    yield LLMProviderInfo(
        config=config, 
        description="A Dev GenAI model for use with an LLM client."
    )