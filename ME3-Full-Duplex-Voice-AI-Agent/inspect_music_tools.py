
import asyncio
import json
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "src.mcp_servers.system.server"]
    )

    async with stdio_client(params, errlog=sys.stderr) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()

            for tool in result.tools:
                print("\n" + "=" * 60)
                print("NAME:", tool.name)
                print("DESCRIPTION:", tool.description)
                print("INPUT SCHEMA:")
                print(json.dumps(tool.input_schema, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
