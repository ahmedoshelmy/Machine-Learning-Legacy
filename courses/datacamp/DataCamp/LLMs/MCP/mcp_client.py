from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def get_tools_from_mcp():
    # Define the server parameters
    params = StdioServerParameters(
        command=sys.executable,
        args=["currency_server.py"],
    )

    # Connect to the MCP server and open a session
    async with stdio_client(params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            # Initialize the session
            await session.initialize()

            # Ask the server what tools it provides
            response = await session.list_tools()

            # Display the available tools
            print("Connected to MCP server!")
            print("Available tools:")
            for tool in response.tools:
                print(f" - {tool.name}: {tool.description}")
                
            return response.tools

asyncio.run(get_tools_from_mcp())


from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def call_mcp_tool(tool_name: str, arguments: dict) -> str:
    params = StdioServerParameters(
        command=sys.executable,
        args=["currency_server.py"],
    )

    async with stdio_client(params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()

            # Call the currency conversion tool
            result = await session.call_tool(tool_name, arguments)

            # Extract and print the text content of the server response
            text_content = result.content[0].text

            print(f"Conversion Result: {text_content}")
            return text_content

# Run the "convert_currency" tool
asyncio.run(
    call_mcp_tool("convert_currency",
                  {"amount": 250.0, "from_currency": "USD", "to_currency": "EUR"})
)