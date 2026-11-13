
import json
import os
import sys
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPClientManager:

    def __init__(self):

        self.exit_stack = AsyncExitStack()
        self.sessions = {}
        self.tool_registry = {}

        # Raspberry Pi SSH configuration
        self.pi_ssh_target = "kentjustincanja@raspberrypi.local"
        self.pi_project_root = "~/full_duplex_voice_agent"
        self.pi_python = ".venv/bin/python"

        # Required by Windows OpenSSH when launched
        # with the MCP SDK's filtered environment.
        self.ssh_environment = {
            "PROGRAMDATA": os.environ["PROGRAMDATA"],
        }

        # Windows-local Music server and Raspberry Pi remote servers
        self.server_configs = {

            "music": {
                "command": sys.executable,
                "args": [
                    "-m",
                    "src.mcp_servers.music.server",
                ],
            },

            "system": {
                "command": "ssh",
                "args": [
                    "-T",
                    "-o",
                    "BatchMode=yes",
                    self.pi_ssh_target,
                    f"cd {self.pi_project_root} && exec {self.pi_python} -m src.mcp_servers.system.server",
                ],
            },

            "information": {
                "command": "ssh",
                "args": [
                    "-T",
                    "-o",
                    "BatchMode=yes",
                    self.pi_ssh_target,
                    f"cd {self.pi_project_root} && exec {self.pi_python} -m src.mcp_servers.information.server",
                ],
            },

            "productivity": {
                "command": "ssh",
                "args": [
                    "-T",
                    "-o",
                    "BatchMode=yes",
                    self.pi_ssh_target,
                    f"cd {self.pi_project_root} && exec {self.pi_python} -m src.mcp_servers.productivity.server",
                ],
            },

            "website": {
                "command": "ssh",
                "args": [
                    "-T",
                    "-o",
                    "BatchMode=yes",
                    self.pi_ssh_target,
                    f"cd {self.pi_project_root} && exec {self.pi_python} -m src.mcp_servers.website.server",
                ],
            },

            "smart_home": {
                "command": "ssh",
                "args": [
                    "-T",
                    "-o",
                    "BatchMode=yes",
                    self.pi_ssh_target,
                    f"cd {self.pi_project_root} && exec {self.pi_python} -m src.mcp_servers.smart_home.server",
                ],
            },
        }

    async def connect_server(self, server_name):

        if server_name not in self.server_configs:
            raise ValueError(
                f"Unknown MCP server: {server_name}"
            )

        if server_name in self.sessions:
            print(f"[MCP] Already connected: {server_name}")
            return

        config = self.server_configs[server_name]

        server_parameters = StdioServerParameters(
            command=config["command"],
            args=config["args"],
            env=(
                self.ssh_environment
                if server_name != "music"
                else None
            ),
        )

        read_stream, write_stream = await self.exit_stack.enter_async_context(
            stdio_client(
                server_parameters,
                errlog=sys.stderr,
            )
        )

        session = await self.exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )

        await session.initialize()

        self.sessions[server_name] = session

        print(f"[MCP] Connected to {server_name} server.")

    async def connect_all(self):

        for server_name in self.server_configs:
            await self.connect_server(server_name)

    async def discover_tools(self):

        self.tool_registry.clear()

        for server_name, session in self.sessions.items():

            result = await session.list_tools()

            for tool in result.tools:

                if tool.name in self.tool_registry:
                    raise RuntimeError(
                        f"Duplicate MCP tool name: {tool.name}"
                    )

                self.tool_registry[tool.name] = {
                    "server": server_name,
                    "tool": tool,
                }

        print(
            f"[MCP] Discovered {len(self.tool_registry)} tools "
            f"across {len(self.sessions)} servers."
        )

        return self.tool_registry

    @staticmethod
    def _clean_json_schema(schema):

        if isinstance(schema, dict):

            cleaned = {}

            for key, value in schema.items():

                if key == "default":
                    continue

                if key == "properties" and isinstance(value, dict):

                    cleaned["properties"] = {
                        property_name:
                            MCPClientManager._clean_json_schema(
                                property_schema
                            )
                        for property_name, property_schema
                        in value.items()
                    }

                    continue

                if key == "title":
                    continue

                cleaned[key] = (
                    MCPClientManager._clean_json_schema(value)
                )

            return cleaned

        if isinstance(schema, list):

            return [
                MCPClientManager._clean_json_schema(item)
                for item in schema
            ]

        return schema

    def get_gemini_tools(self):

        if not self.tool_registry:
            raise RuntimeError(
                "No MCP tools discovered. "
                "Call discover_tools() first."
            )

        gemini_tools = []

        for tool_name, tool_info in self.tool_registry.items():

            tool = tool_info["tool"]

            parameters = self._clean_json_schema(
                tool.input_schema
            )

            if not isinstance(parameters, dict):
                raise TypeError(
                    f"Invalid input schema for tool: {tool_name}"
                )

            gemini_tools.append({
                "type": "function",
                "name": tool.name,
                "description": tool.description or "",
                "parameters": parameters,
            })

        print(
            f"[MCP] Converted {len(gemini_tools)} tools "
            "into Gemini function declarations."
        )

        return gemini_tools

    @staticmethod
    def _normalize_value(value):

        if isinstance(value, str):

            try:
                decoded = json.loads(value)
            except json.JSONDecodeError:
                return value

            return MCPClientManager._normalize_value(decoded)

        if isinstance(value, dict):

            return {
                key: MCPClientManager._normalize_value(item)
                for key, item in value.items()
            }

        if isinstance(value, list):

            return [
                MCPClientManager._normalize_value(item)
                for item in value
            ]

        return value

    @staticmethod
    def _normalize_content(content):

        if isinstance(content, list):

            normalized = []

            for item in content:

                text_value = getattr(item, "text", None)

                if text_value is not None:
                    normalized.append(
                        MCPClientManager._normalize_value(text_value)
                    )

                elif hasattr(item, "model_dump"):
                    normalized.append(item.model_dump())

                else:
                    normalized.append(str(item))

            if len(normalized) == 1:
                return normalized[0]

            return normalized

        return MCPClientManager._normalize_value(content)

    async def call_tool(self, tool_name, arguments=None):

        if tool_name not in self.tool_registry:
            raise ValueError(
                f"Unknown MCP tool: {tool_name}"
            )

        if arguments is None:
            arguments = {}

        if not isinstance(arguments, dict):
            raise TypeError(
                "Tool arguments must be a dictionary."
            )

        tool_info = self.tool_registry[tool_name]
        server_name = tool_info["server"]

        if server_name not in self.sessions:
            raise RuntimeError(
                f"MCP server is not connected: {server_name}"
            )

        session = self.sessions[server_name]

        print(
            f"[MCP] Routing '{tool_name}' "
            f"to '{server_name}' server."
        )

        result = await session.call_tool(
            name=tool_name,
            arguments=arguments,
        )

        if result.is_error:

            return {
                "success": False,
                "tool": tool_name,
                "server": server_name,
                "error": self._normalize_content(result.content),
            }

        if result.content:
            normalized_result = self._normalize_content(
                result.content
            )

        else:
            structured_content = getattr(
                result,
                "structured_content",
                None,
            )

            if structured_content is None:
                structured_content = getattr(
                    result,
                    "structuredContent",
                    None,
                )

            normalized_result = self._normalize_value(
                structured_content
            )

        return {
            "success": True,
            "tool": tool_name,
            "server": server_name,
            "result": normalized_result,
        }

    async def close(self):

        await self.exit_stack.aclose()

        self.sessions.clear()
        self.tool_registry.clear()

        print("[MCP] All server connections closed.")
