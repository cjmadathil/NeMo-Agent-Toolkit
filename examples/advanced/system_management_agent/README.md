# System Management Agent

A comprehensive system management agent built with the NeMo Agent Toolkit that integrates web search, PDF document analysis, and Redfish API capabilities for system administration tasks.

## Features

This agent provides three main capabilities:

### 1. Web Search Tool
- Search specific URLs and documentation sites
- Support for multiple domains (NVIDIA, Red Hat, VMware documentation)
- Concurrent URL processing
- Content extraction and cleanup

### 2. PDF Embeddings Tool
- Load and process PDF manuals and documentation
- Create semantic embeddings using sentence transformers
- Fast similarity search using FAISS
- Caching for improved performance

### 3. Redfish API Tool
- Query BMC (Baseboard Management Controller) endpoints
- Get system information, thermal status, and health data
- Support for multiple BMC endpoints
- Mock mode for testing without real hardware

## Prerequisites

1. **Python Environment**: Python 3.11 or 3.12
2. **Dev GenAI API Key**: Required for the LLM integration
3. **Dependencies**: All dependencies are listed in `pyproject.toml`

## Installation

1. **Navigate to the project directory**:
   ```bash
   cd examples/advanced/system_management_agent
   ```

2. **Install the package**:
   ```bash
   uv pip install -e .
   ```

3. **Set up your API key**:
   ```bash
   export DEV_GENAI_API_KEY="your-dev-genai-api-key"
   ```

## Configuration

The agent uses the configuration file `src/aiq_system_management_agent/configs/config.yml`. Key configurations include:

### Web Search Tool
```yaml
web_search_tool:
  allowed_urls:
    - "https://docs.nvidia.com"
    - "https://developer.nvidia.com"
    - "https://www.redhat.com"
  max_content_length: 10000
  timeout: 30
```

### PDF Embeddings Tool
```yaml
pdf_embeddings_tool:
  pdf_directory: "./data/pdfs"
  embeddings_model: "all-MiniLM-L6-v2"
  chunk_size: 1000
  cache_embeddings: true
```

### Redfish API Tool
```yaml
redfish_api_tool:
  bmc_endpoints:
    - "https://192.168.1.100"
  username: "admin"
  password: "admin"
  mock_mode: true  # Set to false for real BMCs
```

### Dev GenAI LLM
```yaml
dev_genai_llm:
  _type: dev_genai
  api_key: "${DEV_GENAI_API_KEY}"
  model_name: "dev-genai-chat"
  temperature: 0.3
```

## Usage

### Basic Usage

Run the agent with a system management query:

```bash
aiq start console --config_file src/aiq_system_management_agent/configs/config.yml --input "Check the thermal status of all managed systems"
```

### Example Queries

1. **System Health Check**:
   ```
   "Perform a comprehensive health check of all managed systems including thermal status, power state, and hardware information"
   ```

2. **Documentation Search**:
   ```
   "Search for NVIDIA GPU thermal management best practices and correlate with current system temperatures"
   ```

3. **Troubleshooting**:
   ```
   "The server fans are running at high speed. Help me investigate the cause and find solutions"
   ```

4. **PDF Manual Search**:
   ```
   "Search the loaded manuals for information about memory configuration and optimization"
   ```

## Data Preparation

### PDF Documents
1. Create the PDF directory:
   ```bash
   mkdir -p data/pdfs
   ```

2. Add your PDF manuals and documentation to this directory.

3. The agent will automatically process and index these PDFs on startup.

### BMC Configuration
For real BMC endpoints:
1. Update the `bmc_endpoints` in the configuration
2. Set the correct `username` and `password`
3. Set `mock_mode: false`
4. Ensure network connectivity to the BMCs

## Available Tools

### Web Search Tools
- `search_url(url)`: Search a specific URL
- `search_multiple_urls(urls)`: Search multiple URLs
- `search_documentation(query)`: Search system documentation

### PDF Tools
- `load_pdf_embeddings()`: Load and process PDFs
- `search_pdf_content(query, max_results)`: Search PDF content
- `list_loaded_pdfs()`: List loaded PDF files

### Redfish API Tools
- `get_system_info()`: Get hardware information
- `get_thermal_status()`: Get temperature and fan data
- `get_manager_info()`: Get BMC information
- `get_power_status()`: Get power status
- `get_health_status()`: Get overall health status

## Architecture

The system management agent is built using:

- **LangGraph**: For agent orchestration and tool management
- **Dev GenAI**: As the language model provider
- **Sentence Transformers**: For PDF embeddings
- **FAISS**: For fast similarity search
- **httpx**: For async HTTP requests
- **BeautifulSoup**: For web content parsing
- **PyPDF2**: for PDF text extraction

## Development

### Project Structure
```
system_management_agent/
├── src/aiq_system_management_agent/
│   ├── __init__.py
│   ├── register.py              # Main workflow registration
│   ├── prompts.py               # Agent prompts
│   ├── web_search_tool.py       # Web search implementation
│   ├── pdf_embeddings_tool.py   # PDF processing tool
│   ├── redfish_api_tool.py      # Redfish API tool
│   └── configs/
│       └── config.yml           # Configuration file
├── data/
│   ├── pdfs/                    # PDF documents directory
│   └── embeddings_cache/        # Cached embeddings
├── tests/                       # Test files
├── pyproject.toml              # Project configuration
└── README.md                   # This file
```

### Running Tests
```bash
pytest tests/ -v
```

## Security Considerations

1. **API Keys**: Store API keys as environment variables
2. **BMC Access**: Use secure credentials for BMC access
3. **Network Security**: Ensure BMC endpoints are on secure networks
4. **SSL Verification**: Enable SSL verification for production BMCs
5. **Access Control**: Limit web search to trusted domains

## Troubleshooting

### Common Issues

1. **Dev GenAI API Key**: Ensure `DEV_GENAI_API_KEY` is set correctly
2. **PDF Loading**: Check that PDF files are in the correct directory
3. **BMC Connection**: Verify network connectivity and credentials
4. **Memory Usage**: Large PDF collections may require more memory

### Debug Mode
Enable verbose logging by setting `verbose: true` in the configuration.

## Contributing

To extend the agent:

1. **Add new tools**: Create new tool files following the existing patterns
2. **Update prompts**: Modify `prompts.py` to include new capabilities
3. **Add configurations**: Update the config file for new tools
4. **Write tests**: Add comprehensive tests for new functionality

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

## Support

For issues and questions:
1. Check the NeMo Agent Toolkit documentation
2. Review the troubleshooting section above
3. File an issue on the GitHub repository