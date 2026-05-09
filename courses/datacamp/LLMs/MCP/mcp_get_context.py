from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def get_context_from_mcp(user_query: str) -> tuple[str, str]:
    """Fetch resource content and prompt text from the MCP server."""
    params = StdioServerParameters(command=sys.executable, args=["currency_server.py"])

    async with stdio_client(params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()

            # Read the resource (supported currencies)
            resource_result = await session.read_resource("file://currencies.txt")
            resource_text = resource_result.contents[0].text

            # Get the prompt with the user's query
            prompt_result = await session.get_prompt("convert_currency_prompt",
                arguments={"currency_request": user_query})
            prompt_text = prompt_result.messages[0].content.text

            return resource_text, prompt_text

print(asyncio.run(get_context_from_mcp("How much is 50 GBP in euros?")))


from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def get_context_from_mcp(user_query: str) -> tuple[str, str]:
    params = StdioServerParameters(command=sys.executable, args=["currency_server.py"])
    async with stdio_client(params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            resource_result = await session.read_resource("file://currencies.txt")
            resource_text = resource_result.contents[0].text
            prompt_result = await session.get_prompt("convert_currency_prompt",
                arguments={"currency_request": user_query})
            prompt_text = prompt_result.messages[0].content.text
            return resource_text, prompt_text

async def get_tools_from_mcp():
    params = StdioServerParameters(command=sys.executable, args=["currency_server.py"])
    async with stdio_client(params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            response = await session.list_tools()
            return response.tools

async def call_mcp_tool(tool_name: str, arguments: dict) -> str:
    params = StdioServerParameters(command=sys.executable, args=["currency_server.py"])
    async with stdio_client(params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)
            return str(result.content[0].text)

async def call_llm_with_context(user_query: str):
    """Call the LLM with resource and prompt context from MCP."""
    resource_text, prompt_text = await get_context_from_mcp(user_query)

    # Combine the resource and prompt text
    full_prompt = prompt_text + "\n\nSupported currencies:\n" + resource_text

    client = AsyncOpenAI(api_key="<OPENAI_API_TOKEN>")
    mcp_tools = await get_tools_from_mcp()
    openai_tools = [{"type": "function", "name": t.name, "description": t.description or "", "parameters": t.inputSchema} for t in mcp_tools]

    # Send full_prompt and the tools list to the model
    response = await client.responses.create(
        model="gpt-4o-mini",
        input=full_prompt,
        tools=openai_tools,
    )

    output = response.output[0]

    # Return the text response
    if output.type == "message":
        print(f"\nAssistant: {output.content[0].text}")
        return str(output.content[0].text)

    # Call the tool requested in the LLM's function call
    if output.type == "function_call":
        args = json.loads(output.arguments)
        result = await call_mcp_tool(output.name, args)
        followup = await client.responses.create(
            model="gpt-4o-mini",
            input=[
                {"role": "user", "content": user_query},
                output,
                {"type": "function_call_output", "call_id": output.call_id, "output": result},
            ],
        )
        if followup.output and followup.output[0].type == "message":
            print(f"\nAssistant: {followup.output[0].content[0].text}")
            return str(followup.output[0].content[0].text)

print("=== Ambiguous request (prompt asks for clarification) ===")
asyncio.run(call_llm_with_context("Convert some euros to dollars"))
print("\n=== Unambiguous request (model calls tool) ===")
asyncio.run(call_llm_with_context("How much is 50 GBP in euros?"))