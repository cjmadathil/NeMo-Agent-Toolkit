<!--
SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Dev GenAI Integration

The Agent Intelligence Toolkit supports integration with Dev GenAI models through the LangChain dev-genai integration. This documentation provides a comprehensive guide on how to integrate Dev GenAI models into your AIQ Toolkit workflow. To view the full list of supported LLM providers, run `aiq info components -t llm_provider`.

## Prerequisites

Before integrating Dev GenAI, ensure you have:

### 1. Install LangChain Dev GenAI Package
```bash
pip install langchain-dev-genai
```

### 2. API Key Setup
Obtain your Dev GenAI API key and set it as an environment variable:

```bash
export DEV_GENAI_API_KEY="your-api-key-here"
```

For detailed setup instructions, refer to the [Dev GenAI LangChain integration documentation](https://github.com/cjmadathil/langchain/tree/feature/dev-genai-integration/libs/partners/dev-genai).

## Configuration

### Basic Configuration
Add the Dev GenAI LLM configuration to your workflow config file:

```yaml
llms:
  dev_genai_llm:
    _type: dev_genai
    api_key: "${DEV_GENAI_API_KEY}"
    model_name: dev-genai-chat
    temperature: 0.7
    max_tokens: 1024
```

### Advanced Configuration
For more advanced use cases, you can configure additional parameters:

```yaml
llms:
  advanced_dev_genai:
    _type: dev_genai
    api_key: "${DEV_GENAI_API_KEY}"
    base_url: "https://api.dev-genai.com/v1"
    model_name: dev-genai-text
    temperature: 0.8
    max_tokens: 2048
    timeout: 30
    max_retries: 5
    streaming: true
    top_p: 0.9
    frequency_penalty: 0.1
    presence_penalty: 0.2
    stop: ["<end>", "<stop>"]
```

### Configurable Options

#### Required Parameters
* `_type`: Must be set to `"dev_genai"` to use the Dev GenAI provider
* `api_key`: Your Dev GenAI API key (can be set via `DEV_GENAI_API_KEY` environment variable)

#### Optional Parameters
* `base_url`: API endpoint URL (default: `"https://api.dev-genai.com/v1"`)
* `model_name`: Model name (default: `"dev-genai-chat"`)
  - Available models: `dev-genai-chat`, `dev-genai-text`, and other custom models
* `temperature`: Sampling temperature 0.0-2.0 (default: `0.0`)
* `max_tokens`: Maximum number of tokens to generate (must be > 0, default: `1024`)
* `timeout`: Request timeout in seconds (must be ≥ 1, default: `60`)
* `max_retries`: Maximum retry attempts (must be ≥ 0, default: `3`)
* `streaming`: Enable streaming responses (default: `false`)
* `top_p`: Top-p nucleus sampling 0.0-1.0 (default: `1.0`)
* `frequency_penalty`: Frequency penalty -2.0 to 2.0 (default: `0.0`)
* `presence_penalty`: Presence penalty -2.0 to 2.0 (default: `0.0`)
* `stop`: List of stop sequences to terminate generation (default: `null`)

## Usage in Workflow

### Basic React Agent
Reference the Dev GenAI LLM in your workflow configuration:

```yaml
functions:
  wikipedia_search:
    _type: wiki_search
    max_results: 2

llms:
  dev_genai_model:
    _type: dev_genai
    api_key: "${DEV_GENAI_API_KEY}"
    model_name: dev-genai-chat
    temperature: 0.7

workflow:
  _type: react_agent
  llm_name: dev_genai_model
  tool_names: [wikipedia_search]
  verbose: true
  retry_parsing_errors: true
  max_retries: 3
```

### With Streaming Enabled
For real-time responses, enable streaming:

```yaml
llms:
  streaming_dev_genai:
    _type: dev_genai
    api_key: "${DEV_GENAI_API_KEY}"
    model_name: dev-genai-chat
    temperature: 0.7
    streaming: true
    max_tokens: 2048

workflow:
  _type: react_agent
  llm_name: streaming_dev_genai
  # ... other workflow configurations
```

## Embedding Configuration

The Dev GenAI embedder provider supports the following configuration parameters:

### Required Parameters
- `_type`: Must be set to `"dev_genai"` to use the Dev GenAI provider
- `api_key`: Your Dev GenAI API key (can be set via `DEV_GENAI_API_KEY` environment variable)

### Optional Parameters
- `base_url`: API endpoint URL (default: `"https://api.dev-genai.com/v1"`)
- `model_name`: Embedding model name (default: `"dev-genai-embed"`)
- `dimensions`: Number of dimensions for embedding vectors (default: model default)
- `max_retries`: Maximum retry attempts (default: `3`)
- `timeout`: Request timeout in seconds (default: `60`)
- `batch_size`: Maximum texts to process in a single batch (default: `100`)
- `encoding_format`: Encoding format for embeddings (default: `"float"`)
- `truncate_input`: Whether to truncate input text if too long (default: `true`)

### Example Embedding Configuration

```yaml
embedders:
  dev_genai_embedder:
    _type: dev_genai
    api_key: "${DEV_GENAI_API_KEY}"
    model_name: "dev-genai-embed"
    dimensions: 384
    batch_size: 50
    timeout: 30
    max_retries: 2
    encoding_format: "float"
    truncate_input: true

# Use with retrievers
retrievers:
  my_retriever:
    _type: milvus
    embedder_name: "dev_genai_embedder"
    collection_name: "my_documents"
```

## Running the Workflow

### Using the CLI
1. **Create a configuration file** (e.g., `dev_genai_config.yaml`) with your Dev GenAI settings.

2. **Run the workflow**:
   ```bash
   aiq start console --config_file dev_genai_config.yaml --input "Tell me about artificial intelligence"
   ```

### Environment Variables
You can set the following environment variables for easier configuration:

```bash
export DEV_GENAI_API_KEY="your-api-key"
```

## Available Models

The Dev GenAI integration supports various models including:

### Language Models (LLMs)
- **dev-genai-chat**: Optimized for conversational AI and chat applications
- **dev-genai-text**: Designed for text generation and completion tasks
- **Custom models**: Other models as supported by the Dev GenAI API

### Embedding Models
- **dev-genai-embed**: Standard embedding model for text similarity and retrieval
- **dev-genai-embed-large**: Larger embedding model with higher dimensionality
- **Custom embedding models**: Other embedding models as supported by the Dev GenAI API

## Error Handling and Troubleshooting

### Common Issues

1. **API Key Issues**
   - Ensure your `DEV_GENAI_API_KEY` is set correctly
   - Verify the API key is valid and has proper permissions

2. **Network Issues**
   - Check your internet connection and firewall settings
   - Adjust `timeout` and `max_retries` parameters as needed

3. **Model Availability**
   - Verify the model name is correct and available
   - Check the Dev GenAI API documentation for supported models

4. **Rate Limiting**
   - Implement appropriate `max_retries` and `timeout` values
   - Consider using exponential backoff for retries

### Configuration Validation
The integration includes comprehensive validation for:
- Parameter ranges (temperature, top_p, penalties)
- Positive integers for tokens and timeout
- Valid model names and API endpoints

## Integration with Other Components

The Dev GenAI LLM works seamlessly with:

- **All NeMo Agent Toolkit agents**: react_agent, reasoning_agent, rewoo_agent, tool_calling_agent
- **Custom tools and functions**: Any registered function can be used with Dev GenAI
- **Evaluation framework**: Support for workflow evaluation and benchmarking
- **Observability tools**: Full integration with profiling and tracing systems
- **Memory systems**: Compatible with all supported memory backends
- **Embedding and retrieval**: Use Dev GenAI embeddings with any retrieval system

## Performance Considerations

### Optimization Tips
1. **Streaming**: Enable streaming for real-time applications
2. **Temperature**: Use lower temperatures (0.0-0.3) for deterministic outputs
3. **Token Limits**: Set appropriate `max_tokens` based on your use case
4. **Retry Strategy**: Configure `max_retries` and `timeout` for reliability

### Monitoring
- Use the built-in profiler to monitor token usage and response times
- Enable observability to track model performance across requests
- Monitor API rate limits and adjust request patterns accordingly

For more examples and advanced usage patterns, see the main NeMo Agent Toolkit documentation and the examples directory.