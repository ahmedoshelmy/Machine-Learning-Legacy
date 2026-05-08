from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def call_openai_llm(user_query: str):
    """Call OpenAI LLM with MCP tools."""
    
    print(f"\nUser: {user_query}\n")

    mcp_tools = await get_tools_from_mcp()
    
    openai_tools = []
    for tool in mcp_tools:
        openai_tool = {
            "type": "function",
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema,
        }
        openai_tools.append(openai_tool)
    
    # Send the user query and formatted tools to the LLM
    client = AsyncOpenAI(api_key="<OPENAI_API_TOKEN>")

    response = await client.responses.create(
        model="gpt-4o-mini",
        input=user_query,
        tools=openai_tools,
    )

    output = response.output[0]

    if output.type == "function_call":
        args = json.loads(output.arguments)
        name = output.name

        print(f"Model decided to call: {name}")
        print(f"Arguments: {args}\n")

        # Call the MCP tool
        result = await call_mcp_tool(name, args)

        # Send the result back to OpenAI for final response
        followup = await client.responses.create(
            model="gpt-4o-mini",
            input=[
                {"role": "user", "content": user_query},
                output,
                {
                    "type": "function_call_output",
                    "call_id": output.call_id,
                    "output": result,
                },
            ],
        )

        if followup.output and followup.output[0].type == "message":
            print(f"\nAssistant: {followup.output[0].content[0].text}")
            return str(followup.output[0].content[0].text)
        else:
            print("No follow-up message from model.")

    elif output.type == "message":
        print(f"\nAssistant: {output.content[0].text}")
        return str(output.content[0].text)
    else:
        print(f"Unhandled output type: {output.type}")


if __name__ == "__main__":
    asyncio.run(call_openai_llm("How much is 250 US dollars in euros?"))