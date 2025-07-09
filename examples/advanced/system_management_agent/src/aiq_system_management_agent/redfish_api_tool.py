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
import base64
import json
import logging
from typing import Dict, List, Optional

import httpx
from langchain_core.tools import tool
from pydantic import Field

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig
from aiq.profiler.decorators.function_tracking import track_function

logger = logging.getLogger(__name__)


class RedfishAPIToolConfig(FunctionBaseConfig, name="redfish_api_tool"):
    """Configuration for the Redfish API Tool."""
    
    bmc_endpoints: List[str] = Field(
        default_factory=lambda: [
            "https://192.168.1.100",  # Example BMC IP
            "https://10.0.0.100",     # Example BMC IP
        ],
        description="List of BMC (Baseboard Management Controller) endpoints"
    )
    username: str = Field(
        default="admin",
        description="Username for BMC authentication"
    )
    password: str = Field(
        default="admin",
        description="Password for BMC authentication"
    )
    timeout: int = Field(
        default=30,
        description="Request timeout in seconds"
    )
    verify_ssl: bool = Field(
        default=False,
        description="Whether to verify SSL certificates"
    )
    max_retries: int = Field(
        default=3,
        description="Maximum number of retry attempts"
    )
    mock_mode: bool = Field(
        default=True,
        description="Whether to use mock data instead of real API calls"
    )


@register_function(config_type=RedfishAPIToolConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def redfish_api_tool(config: RedfishAPIToolConfig, builder: Builder):
    """
    Redfish API tool that can fetch data from BMC endpoints using Redfish APIs.
    """
    
    # Mock data for testing when real BMC endpoints are not available
    MOCK_DATA = {
        "systems": {
            "Members": [
                {
                    "@odata.id": "/redfish/v1/Systems/1",
                    "Id": "1",
                    "Name": "Computer System",
                    "SystemType": "Physical",
                    "Manufacturer": "Dell Inc.",
                    "Model": "PowerEdge R740",
                    "SerialNumber": "1234567",
                    "Status": {
                        "State": "Enabled",
                        "Health": "OK"
                    },
                    "PowerState": "On",
                    "ProcessorSummary": {
                        "Count": 2,
                        "Model": "Intel(R) Xeon(R) Gold 6248 CPU @ 2.50GHz"
                    },
                    "MemorySummary": {
                        "TotalSystemMemoryGiB": 256,
                        "Status": {"State": "Enabled", "Health": "OK"}
                    }
                }
            ]
        },
        "chassis": {
            "Members": [
                {
                    "@odata.id": "/redfish/v1/Chassis/1",
                    "Id": "1",
                    "Name": "Computer System Chassis",
                    "ChassisType": "RackMount",
                    "Manufacturer": "Dell Inc.",
                    "Model": "PowerEdge R740",
                    "SerialNumber": "1234567",
                    "Status": {
                        "State": "Enabled",
                        "Health": "OK"
                    },
                    "PowerState": "On",
                    "Thermal": {
                        "Temperatures": [
                            {
                                "Name": "Inlet Temperature",
                                "ReadingCelsius": 23.0,
                                "Status": {"State": "Enabled", "Health": "OK"}
                            },
                            {
                                "Name": "CPU1 Temperature",
                                "ReadingCelsius": 45.0,
                                "Status": {"State": "Enabled", "Health": "OK"}
                            }
                        ],
                        "Fans": [
                            {
                                "Name": "Fan1",
                                "Reading": 3200,
                                "Status": {"State": "Enabled", "Health": "OK"}
                            },
                            {
                                "Name": "Fan2",
                                "Reading": 3150,
                                "Status": {"State": "Enabled", "Health": "OK"}
                            }
                        ]
                    }
                }
            ]
        },
        "managers": {
            "Members": [
                {
                    "@odata.id": "/redfish/v1/Managers/1",
                    "Id": "1",
                    "Name": "Manager",
                    "ManagerType": "BMC",
                    "Status": {
                        "State": "Enabled",
                        "Health": "OK"
                    },
                    "FirmwareVersion": "4.40.00.00",
                    "NetworkProtocol": {
                        "Status": {"State": "Enabled", "Health": "OK"}
                    }
                }
            ]
        }
    }
    
    def _create_auth_header(username: str, password: str) -> str:
        """Create basic authentication header."""
        credentials = f"{username}:{password}"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()
        return f"Basic {encoded_credentials}"
    
    @track_function()
    async def _make_redfish_request(endpoint: str, path: str) -> Dict:
        """Make a Redfish API request to a BMC endpoint."""
        if config.mock_mode:
            # Return mock data based on the path
            if "Systems" in path:
                return MOCK_DATA["systems"]
            elif "Chassis" in path:
                return MOCK_DATA["chassis"]
            elif "Managers" in path:
                return MOCK_DATA["managers"]
            else:
                return {"error": f"Mock data not available for path: {path}"}
        
        try:
            url = f"{endpoint.rstrip('/')}/redfish/v1/{path.lstrip('/')}"
            headers = {
                "Authorization": _create_auth_header(config.username, config.password),
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            
            async with httpx.AsyncClient(
                timeout=config.timeout,
                verify=config.verify_ssl
            ) as client:
                for attempt in range(config.max_retries):
                    try:
                        response = await client.get(url, headers=headers)
                        response.raise_for_status()
                        return response.json()
                    except httpx.RequestError as e:
                        if attempt == config.max_retries - 1:
                            return {"error": f"Request failed after {config.max_retries} attempts: {str(e)}"}
                        await asyncio.sleep(2 ** attempt)
                    except httpx.HTTPStatusError as e:
                        return {"error": f"HTTP {e.response.status_code}: {e.response.text}"}
        
        except Exception as e:
            logger.error(f"Error making Redfish request to {endpoint}/{path}: {str(e)}")
            return {"error": f"Request failed: {str(e)}"}
    
    @track_function()
    async def _query_all_endpoints(path: str) -> Dict:
        """Query all configured BMC endpoints for a specific path."""
        results = {}
        tasks = []
        
        for endpoint in config.bmc_endpoints:
            tasks.append(_make_redfish_request(endpoint, path))
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, response in enumerate(responses):
            endpoint = config.bmc_endpoints[i]
            if isinstance(response, Exception):
                results[endpoint] = {"error": str(response)}
            else:
                results[endpoint] = response
        
        return results
    
    @track_function()
    def _format_system_info(data: Dict) -> str:
        """Format system information from Redfish response."""
        formatted = []
        
        for endpoint, response in data.items():
            if "error" in response:
                formatted.append(f"**{endpoint}**: Error - {response['error']}")
                continue
            
            formatted.append(f"**{endpoint}**:")
            
            if "Members" in response:
                for member in response["Members"]:
                    formatted.append(f"  - System: {member.get('Name', 'N/A')}")
                    formatted.append(f"    Manufacturer: {member.get('Manufacturer', 'N/A')}")
                    formatted.append(f"    Model: {member.get('Model', 'N/A')}")
                    formatted.append(f"    Serial: {member.get('SerialNumber', 'N/A')}")
                    formatted.append(f"    Power State: {member.get('PowerState', 'N/A')}")
                    
                    if "Status" in member:
                        status = member["Status"]
                        formatted.append(f"    Status: {status.get('State', 'N/A')} / {status.get('Health', 'N/A')}")
                    
                    if "ProcessorSummary" in member:
                        proc = member["ProcessorSummary"]
                        formatted.append(f"    Processors: {proc.get('Count', 'N/A')} x {proc.get('Model', 'N/A')}")
                    
                    if "MemorySummary" in member:
                        mem = member["MemorySummary"]
                        formatted.append(f"    Memory: {mem.get('TotalSystemMemoryGiB', 'N/A')} GiB")
        
        return "\n".join(formatted)
    
    @track_function()
    def _format_thermal_info(data: Dict) -> str:
        """Format thermal information from Redfish response."""
        formatted = []
        
        for endpoint, response in data.items():
            if "error" in response:
                formatted.append(f"**{endpoint}**: Error - {response['error']}")
                continue
            
            formatted.append(f"**{endpoint}**:")
            
            if "Members" in response:
                for member in response["Members"]:
                    formatted.append(f"  - Chassis: {member.get('Name', 'N/A')}")
                    
                    if "Thermal" in member:
                        thermal = member["Thermal"]
                        
                        if "Temperatures" in thermal:
                            formatted.append("    Temperatures:")
                            for temp in thermal["Temperatures"]:
                                formatted.append(f"      - {temp.get('Name', 'N/A')}: {temp.get('ReadingCelsius', 'N/A')}°C")
                        
                        if "Fans" in thermal:
                            formatted.append("    Fans:")
                            for fan in thermal["Fans"]:
                                formatted.append(f"      - {fan.get('Name', 'N/A')}: {fan.get('Reading', 'N/A')} RPM")
        
        return "\n".join(formatted)
    
    @tool
    async def get_system_info() -> str:
        """
        Get system information from all configured BMC endpoints.
        
        Returns:
            Formatted system information including hardware details, power state, and status
        """
        try:
            data = await _query_all_endpoints("Systems")
            return _format_system_info(data)
        except Exception as e:
            logger.error(f"Error getting system info: {str(e)}")
            return f"Error getting system info: {str(e)}"
    
    @tool
    async def get_thermal_status() -> str:
        """
        Get thermal status including temperature and fan information.
        
        Returns:
            Formatted thermal information including temperatures and fan speeds
        """
        try:
            data = await _query_all_endpoints("Chassis")
            return _format_thermal_info(data)
        except Exception as e:
            logger.error(f"Error getting thermal status: {str(e)}")
            return f"Error getting thermal status: {str(e)}"
    
    @tool
    async def get_manager_info() -> str:
        """
        Get BMC manager information.
        
        Returns:
            Information about the BMC managers and their status
        """
        try:
            data = await _query_all_endpoints("Managers")
            
            formatted = []
            for endpoint, response in data.items():
                if "error" in response:
                    formatted.append(f"**{endpoint}**: Error - {response['error']}")
                    continue
                
                formatted.append(f"**{endpoint}**:")
                
                if "Members" in response:
                    for member in response["Members"]:
                        formatted.append(f"  - Manager: {member.get('Name', 'N/A')}")
                        formatted.append(f"    Type: {member.get('ManagerType', 'N/A')}")
                        formatted.append(f"    Firmware: {member.get('FirmwareVersion', 'N/A')}")
                        
                        if "Status" in member:
                            status = member["Status"]
                            formatted.append(f"    Status: {status.get('State', 'N/A')} / {status.get('Health', 'N/A')}")
            
            return "\n".join(formatted)
        except Exception as e:
            logger.error(f"Error getting manager info: {str(e)}")
            return f"Error getting manager info: {str(e)}"
    
    @tool
    async def get_power_status() -> str:
        """
        Get power status of all managed systems.
        
        Returns:
            Power status information for all systems
        """
        try:
            data = await _query_all_endpoints("Systems")
            
            formatted = []
            for endpoint, response in data.items():
                if "error" in response:
                    formatted.append(f"**{endpoint}**: Error - {response['error']}")
                    continue
                
                formatted.append(f"**{endpoint}**:")
                
                if "Members" in response:
                    for member in response["Members"]:
                        system_name = member.get('Name', 'N/A')
                        power_state = member.get('PowerState', 'N/A')
                        formatted.append(f"  - {system_name}: {power_state}")
            
            return "\n".join(formatted)
        except Exception as e:
            logger.error(f"Error getting power status: {str(e)}")
            return f"Error getting power status: {str(e)}"
    
    @tool
    async def get_health_status() -> str:
        """
        Get overall health status of all managed systems.
        
        Returns:
            Health status summary for all systems
        """
        try:
            # Get system health
            system_data = await _query_all_endpoints("Systems")
            # Get chassis health
            chassis_data = await _query_all_endpoints("Chassis")
            
            formatted = ["# System Health Status"]
            
            # Process system health
            formatted.append("\n## System Health:")
            for endpoint, response in system_data.items():
                if "error" in response:
                    formatted.append(f"**{endpoint}**: Error - {response['error']}")
                    continue
                
                formatted.append(f"**{endpoint}**:")
                if "Members" in response:
                    for member in response["Members"]:
                        name = member.get('Name', 'N/A')
                        status = member.get('Status', {})
                        health = status.get('Health', 'N/A')
                        state = status.get('State', 'N/A')
                        formatted.append(f"  - {name}: {state} / {health}")
            
            # Process chassis health
            formatted.append("\n## Chassis Health:")
            for endpoint, response in chassis_data.items():
                if "error" in response:
                    formatted.append(f"**{endpoint}**: Error - {response['error']}")
                    continue
                
                formatted.append(f"**{endpoint}**:")
                if "Members" in response:
                    for member in response["Members"]:
                        name = member.get('Name', 'N/A')
                        status = member.get('Status', {})
                        health = status.get('Health', 'N/A')
                        state = status.get('State', 'N/A')
                        formatted.append(f"  - {name}: {state} / {health}")
            
            return "\n".join(formatted)
        except Exception as e:
            logger.error(f"Error getting health status: {str(e)}")
            return f"Error getting health status: {str(e)}"
    
    # Return the tools
    yield [get_system_info, get_thermal_status, get_manager_info, get_power_status, get_health_status]