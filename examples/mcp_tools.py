from genai_sdk.tools.mcp import MCPToolset


class DemoMCPClient:
    async def list_tools(self):
        return [
            {
                "name": "search_docs",
                "description": "Search docs",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }
        ]

    async def call_tool(self, name, arguments):
        return f"{name} -> {arguments}"


async def main():
    toolset = MCPToolset(DemoMCPClient())
    tools = await toolset.load()
    out = await tools[0].call({"query": "agent loop"}, ctx=None)
    print(out)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
