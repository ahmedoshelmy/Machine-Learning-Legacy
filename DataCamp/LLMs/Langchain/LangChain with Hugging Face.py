from langchain_huggingface import HuggingFaceEndpoint

# Set your Hugging Face API token 
huggingfacehub_api_token = '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'

# Define the LLM
llm = HuggingFaceEndpoint(repo_id='tiiuae/falcon-7b-instruct', huggingfacehub_api_token=huggingfacehub_api_token)

# Predict the words following the text in question
question = 'Whatever you do, take care of your shoes'
output = llm.invoke(question)

print(output)

# Set your Hugging Face API token
huggingfacehub_api_token = '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'

# Create a prompt template from the template string
template = "You are an artificial intelligence assistant, answer the question. {question}"
prompt = PromptTemplate(template=template,input_variables=['question'])

# Create a chain to integrate the prompt template and LLM
llm = HuggingFaceEndpoint(repo_id='tiiuae/falcon-7b-instruct', huggingfacehub_api_token=huggingfacehub_api_token)
llm_chain = prompt | llm 

question = "How does LangChain make LLM application development easier?"
print(llm_chain.invoke({"question": question}))