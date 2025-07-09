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

from pydantic import AliasChoices
from pydantic import ConfigDict
from pydantic import Field

from aiq.builder.builder import Builder
from aiq.builder.embedder import EmbedderProviderInfo
from aiq.cli.register_workflow import register_embedder_provider
from aiq.data_models.embedder import EmbedderBaseConfig


class DevGenAIEmbedderModelConfig(EmbedderBaseConfig, name="dev_genai"):
    """A Dev GenAI embedder provider to be used with an embedder client."""

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
        default="dev-genai-embed",
        validation_alias=AliasChoices("model_name", "model"),
        serialization_alias="model",
        description="The Dev GenAI embedding model name (e.g., 'dev-genai-embed', 'dev-genai-embed-large')."
    )
    dimensions: int | None = Field(
        default=None,
        description="Number of dimensions for the embedding vectors. If not specified, uses model default."
    )
    max_retries: int = Field(
        default=3, 
        ge=0, 
        description="Maximum number of retry attempts."
    )
    timeout: int = Field(
        default=60, 
        ge=1, 
        description="Request timeout in seconds."
    )
    batch_size: int = Field(
        default=100,
        ge=1,
        description="Maximum number of texts to process in a single batch."
    )
    encoding_format: str = Field(
        default="float",
        description="Encoding format for the embeddings (e.g., 'float', 'base64')."
    )
    truncate_input: bool = Field(
        default=True,
        description="Whether to truncate input text that exceeds the model's maximum length."
    )


@register_embedder_provider(config_type=DevGenAIEmbedderModelConfig)
async def dev_genai_embedder(config: DevGenAIEmbedderModelConfig, builder: Builder):
    """Register Dev GenAI embedder provider."""
    
    yield EmbedderProviderInfo(
        config=config, 
        description="A Dev GenAI model for use with an embedder client."
    )