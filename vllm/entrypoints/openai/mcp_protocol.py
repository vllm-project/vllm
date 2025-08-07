"""
Model Context Protocol (MCP) implementation for vLLM.
Provides a standard interface for external tools and resources.
"""
import json
import asyncio
from typing import Any, Dict, List, Optional, Union
from abc import ABC, abstractmethod

from vllm.logger import init_logger

logger = init_logger(__name__)


class MCPResource:
    """Represents an MCP resource."""
    
    def __init__(self, uri: str, name: str, description: str, mime_type: str):
        self.uri = uri
        self.name = name
        self.description = description
        self.mime_type = mime_type


class MCPTool:
    """Represents an MCP tool."""
    
    def __init__(self, name: str, description: str, input_schema: Dict[str, Any]):
        self.name = name
        self.description = description
        self.input_schema = input_schema


class MCPResult:
    """Result from an MCP operation."""
    
    def __init__(self, content: Union[str, Dict[str, Any]], is_error: bool = False):
        self.content = content
        self.is_error = is_error


class MCPServer(ABC):
    """Abstract base class for MCP servers."""
    
    @abstractmethod
    async def initialize(self) -> Dict[str, Any]:
        """Initialize the server and return capabilities."""
        pass
    
    @abstractmethod
    async def list_resources(self) -> List[MCPResource]:
        """List available resources."""
        pass
    
    @abstractmethod
    async def list_tools(self) -> List[MCPTool]:
        """List available tools."""
        pass
    
    @abstractmethod
    async def read_resource(self, uri: str) -> MCPResult:
        """Read a resource."""
        pass
    
    @abstractmethod
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> MCPResult:
        """Call a tool."""
        pass


class BasicMCPServer(MCPServer):
    """Basic MCP server implementation."""
    
    def __init__(self):
        self.resources: List[MCPResource] = []
        self.tools: List[MCPTool] = []
        self._initialized = False
    
    async def initialize(self) -> Dict[str, Any]:
        """Initialize the server."""
        if self._initialized:
            return self._get_capabilities()
        
        # Add default tools
        self.tools.extend([
            MCPTool(
                name="echo",
                description="Echo back the input text",
                input_schema={
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Text to echo back"
                        }
                    },
                    "required": ["text"]
                }
            ),
            MCPTool(
                name="calculate",
                description="Perform basic mathematical calculations",
                input_schema={
                    "type": "object", 
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Mathematical expression to evaluate"
                        }
                    },
                    "required": ["expression"]
                }
            ),
            MCPTool(
                name="get_time",
                description="Get the current time",
                input_schema={
                    "type": "object",
                    "properties": {
                        "timezone": {
                            "type": "string",
                            "description": "Timezone (optional)",
                            "default": "UTC"
                        }
                    }
                }
            )
        ])
        
        # Add default resources
        self.resources.extend([
            MCPResource(
                uri="memory://system_info",
                name="System Information",
                description="Basic system information",
                mime_type="application/json"
            )
        ])
        
        self._initialized = True
        return self._get_capabilities()
    
    def _get_capabilities(self) -> Dict[str, Any]:
        """Get server capabilities."""
        return {
            "resources": {"subscribe": False, "listChanged": False},
            "tools": {"listChanged": False},
            "logging": {}
        }
    
    async def list_resources(self) -> List[MCPResource]:
        """List available resources."""
        return self.resources
    
    async def list_tools(self) -> List[MCPTool]:
        """List available tools."""
        return self.tools
    
    async def read_resource(self, uri: str) -> MCPResult:
        """Read a resource."""
        if uri == "memory://system_info":
            import platform
            import sys
            
            info = {
                "platform": platform.platform(),
                "python_version": sys.version,
                "architecture": platform.architecture(),
                "processor": platform.processor()
            }
            return MCPResult(content=json.dumps(info, indent=2))
        
        return MCPResult(
            content=f"Resource not found: {uri}",
            is_error=True
        )
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> MCPResult:
        """Call a tool."""
        try:
            if name == "echo":
                text = arguments.get("text", "")
                return MCPResult(content=f"Echo: {text}")
            
            elif name == "calculate":
                expression = arguments.get("expression", "")
                # Basic safety check
                if any(char in expression for char in ["import", "exec", "eval", "__"]):
                    return MCPResult(
                        content="Security error: Invalid characters in expression",
                        is_error=True
                    )
                
                try:
                    # Safe evaluation of basic math expressions
                    result = eval(expression, {"__builtins__": {}}, {
                        "abs": abs, "round": round, "min": min, "max": max,
                        "sum": sum, "pow": pow, "len": len
                    })
                    return MCPResult(content=f"Result: {result}")
                except Exception as e:
                    return MCPResult(
                        content=f"Calculation error: {str(e)}",
                        is_error=True
                    )
            
            elif name == "get_time":
                import datetime
                timezone = arguments.get("timezone", "UTC")
                try:
                    if timezone == "UTC":
                        current_time = datetime.datetime.utcnow()
                    else:
                        # Basic timezone support
                        current_time = datetime.datetime.now()
                    
                    return MCPResult(content=f"Current time: {current_time.isoformat()}")
                except Exception as e:
                    return MCPResult(
                        content=f"Time error: {str(e)}",
                        is_error=True
                    )
            
            else:
                return MCPResult(
                    content=f"Unknown tool: {name}",
                    is_error=True
                )
        
        except Exception as e:
            return MCPResult(
                content=f"Tool execution error: {str(e)}",
                is_error=True
            )


class MCPRequestHandler:
    """Handles MCP requests for the vLLM server."""
    
    def __init__(self, server: MCPServer):
        self.server = server
        self._initialized = False
    
    async def initialize(self):
        """Initialize the MCP handler."""
        if not self._initialized:
            await self.server.initialize()
            self._initialized = True
    
    async def handle_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an MCP request."""
        await self.initialize()
        
        try:
            if method == "resources/list":
                resources = await self.server.list_resources()
                return {
                    "resources": [
                        {
                            "uri": r.uri,
                            "name": r.name,
                            "description": r.description,
                            "mimeType": r.mime_type
                        }
                        for r in resources
                    ]
                }
            
            elif method == "resources/read":
                uri = params.get("uri")
                if not uri:
                    raise ValueError("Missing required parameter: uri")
                
                result = await self.server.read_resource(uri)
                if result.is_error:
                    raise ValueError(result.content)
                
                return {
                    "contents": [
                        {
                            "uri": uri,
                            "mimeType": "text/plain",
                            "text": result.content
                        }
                    ]
                }
            
            elif method == "tools/list":
                tools = await self.server.list_tools()
                return {
                    "tools": [
                        {
                            "name": t.name,
                            "description": t.description,
                            "inputSchema": t.input_schema
                        }
                        for t in tools
                    ]
                }
            
            elif method == "tools/call":
                name = params.get("name")
                arguments = params.get("arguments", {})
                
                if not name:
                    raise ValueError("Missing required parameter: name")
                
                result = await self.server.call_tool(name, arguments)
                
                if result.is_error:
                    return {
                        "content": [
                            {
                                "type": "text", 
                                "text": f"Error: {result.content}"
                            }
                        ],
                        "isError": True
                    }
                else:
                    return {
                        "content": [
                            {
                                "type": "text",
                                "text": str(result.content)
                            }
                        ]
                    }
            
            else:
                raise ValueError(f"Unknown method: {method}")
        
        except Exception as e:
            logger.error(f"MCP request error: {str(e)}")
            return {
                "error": {
                    "code": -1,
                    "message": str(e)
                }
            }


# Default MCP server instance
_default_mcp_server = None


def get_default_mcp_server() -> MCPServer:
    """Get the default MCP server instance."""
    global _default_mcp_server
    if _default_mcp_server is None:
        _default_mcp_server = BasicMCPServer()
    return _default_mcp_server


def create_mcp_handler(server: Optional[MCPServer] = None) -> MCPRequestHandler:
    """Create an MCP request handler."""
    if server is None:
        server = get_default_mcp_server()
    return MCPRequestHandler(server)
