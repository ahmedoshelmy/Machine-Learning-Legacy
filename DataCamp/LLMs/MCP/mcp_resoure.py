from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Currency Converter")

# Define a resource for the currencies file
@mcp.resource("file://currencies.txt")
def get_currencies() -> str:
    """
    Get the list of currency names published by the European Central Bank for currency conversion.

    Returns:
        Contents of the currencies.txt file with currency names
    """
    # Open currencies.txt and read the data
    try:
        with open('currencies.txt', 'r') as f:
            content = f.read()
        return content
    except FileNotFoundError:
        return "currencies.txt file not found"

# Test the resource function
print(get_currencies())

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def list_resources():
    """List all available resources from the MCP server."""
    params = StdioServerParameters(command=sys.executable, args=["currency_server.py"])

    async with stdio_client(params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            # Get the list of resources
            response = await session.list_resources()

            print("Available resources:")
            # Print each resource's URI, name, and description
            for resource in response.resources:
                print(f" - {resource.uri}")
                print(f"   Name: {resource.name}")
                print(f"   Description: {resource.description}")

            return response.resources

asyncio.run(list_resources())


from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Define an async function for reading MCP resources
async def read_resource(resource_uri: str):
    """Read a specific resource by URI."""
    params = StdioServerParameters(
        command=sys.executable,
        args=["currency_server.py"],
    )

    async with stdio_client(params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()

            print(f"Reading resource: {resource_uri}")
            # Read the resource from the session context
            resource_content = await session.read_resource(resource_uri)

            # Print the contents of each resource
            for content in resource_content.contents:
                print(f"\nContent ({content.mimeType}):")
                print(content.text)

            return resource_content

asyncio.run(read_resource("file://currencies.txt"))