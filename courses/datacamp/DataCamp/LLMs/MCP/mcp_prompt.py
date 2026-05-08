from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def list_prompts():
    """List all available prompts from the MCP server."""
    params = StdioServerParameters(command=sys.executable, args=["currency_server.py"])

    async with stdio_client(params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()

            # List available prompts
            prompts = await session.list_prompts()
            print(f"Available prompts: {[p.name for p in prompts.prompts]}")

            return prompts.prompts

asyncio.run(list_prompts())

# Define a prompt for currency conversion
@mcp.prompt(title="Currency Conversion")
def currency_request(currency_request: str) -> str:
    return f"""You are a currency conversion assistant.

Your task is to:
1. Extract the amount and source currency from the user's natural language input.
2. Identify the target currency.
3. Use the conversion tool to convert the amount.

Rules:
- If the amount or currencies are ambiguous or missing, ask the user for clarification.
- Use only supported currency codes (e.g., USD, EUR, GBP).

User's currency conversion request: {currency_request}"""

# Test the prompt function
print(currency_request("100 USD to EUR"))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def read_prompt(user_input: str = "How much is 50 GBP in euros?", prompt_name: str = "convert_currency_prompt") -> str:
    """Retrieve a prompt from the MCP server with user input."""
    params = StdioServerParameters(command=sys.executable, args=["currency_server.py"])

    async with stdio_client(params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()

            # Retrieve the prompt with the user's input
            prompt = await session.get_prompt(prompt_name, arguments={"currency_request": user_input})

            # Print the full prompt text (template + user request)
            text = prompt.messages[0].content.text
            print(text)
            return text

asyncio.run(read_prompt(user_input="How much is 50 GBP in euros?"))