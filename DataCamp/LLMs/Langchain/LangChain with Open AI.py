import langchain_core.prompts
from langchain.memory import ChatMessageHistory
# Define the LLM
llm = OpenAI(model="gpt-3.5-turbo-instruct", api_key="<OPENAI_API_TOKEN>")

# Predict the words following the text in question
question = 'Whatever you do, take care of your shoes'
output = llm.invoke(question)

print(output)
# Define an OpenAI chat model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key='<OPENAI_API_TOKEN>')		

# Create a chat prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "Respond to question: {question}")
    ]
)

# Chain the prompt template and model, and invoke the chain
llm_chain = prompt_template | llm
response = llm_chain.invoke({"question": "How can I retain learning?"})
print(response.content)


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key='<OPENAI_API_TOKEN>')

# Create the conversation history and add the first AI message
history = ChatMessageHistory()
history.add_ai_message("Hello! Ask me anything about Python programming!")

# Add the user message to the history
history.add_user_message("What is a list comprehension?")

# Add another user message and call the model
history.add_user_message("Describe the same in fewer words")
response = llm.invoke(history.messages)
print(response.content)