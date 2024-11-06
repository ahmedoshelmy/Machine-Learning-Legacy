# Define the LLM
llm = ChatOpenAI(api_key="<OPENAI_API_TOKEN>", model="gpt-4o-mini", temperature=0)

# Instantiate the LLM graph transformer
llm_transformer = LLMGraphTransformer(llm=llm)

# Convert the text documents to graph documents
graph_documents = llm_transformer.convert_to_graph_documents(docs)
print(f"Derived Nodes:\n{graph_documents[0].nodes}\n")
print(f"Derived Edges:\n{graph_documents[0].relationships}")

# Instantiate the Neo4j graph
graph = Neo4jGraph(url=url, username=user, password=password)

# Add the graph documents, sources, and include entity labels
graph.add_graph_documents(graph_documents, 
include_source=True,
baseEntityLabel=True
)

# Refresh the graph schema
graph.refresh_schema()

# Print the graph schema
print(graph.get_schema)

# Print the graph schema
print(graph.get_schema)

# Query the graph
results = graph.query("""
MATCH (relativity:Concept {id: "Theory Of Relativity"}) <-[:KNOWN_FOR]- (scientist)
RETURN scientist
""")

print(results[0])

# Create the Graph Cypher QA chain
graph_qa_chain = GraphCypherQAChain.from_llm(
    llm=ChatOpenAI(api_key="<OPENAI_API_TOKEN>", temperature=0), graph=graph, verbose=True
)

# Invoke the chain with the input provided
result = graph_qa_chain.invoke({"query": "Who discovered the element Radium?"})
print(result)
# Print the result text
print(f"Final answer: {result['result']}")

# Create the graph QA chain excluding Concept
graph_qa_chain = GraphCypherQAChain.from_llm(
    graph=graph, 
    llm=llm,
    exclude_types=['Concept'],
    verbose=True 
)

# Invoke the chain with the input provided
result = graph_qa_chain.invoke({"query": "Who was Marie Curie married to?"})
print(f"Final answer: {result['result']}")

# Create the graph QA chain excluding Concept
graph_qa_chain = GraphCypherQAChain.from_llm(
    graph=graph, 
    llm=llm,
    validate_cypher=True,
    verbose=True 
)

# Invoke the chain with the input provided
result = graph_qa_chain.invoke({"query": "Who won the Nobel Prize In Physics?"})
print(f"Final answer: {result['result']}")


# Create an example prompt template
example_prompt = PromptTemplate.from_template(
    "User input: {question}\nCypher query: {query}"
)

# Create the few-shot prompt template
cypher_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="You are a Neo4j expert. Given an input question, create a syntactically correct Cypher query to run.\n\nHere is the schema information\n{schema}.\n\nBelow are a number of examples of questions and their corresponding Cypher queries.",
    suffix="User input: {question}\nCypher query: ",
    input_variables=["question"]
)

# Create the graph Cypher QA chain
graph_qa_chain = GraphCypherQAChain.from_llm(
    llm=llm,
    verbose=True,
    graph=graph,
    validate_cypher=True
)

# Invoke the chain with the input provided
result = graph_qa_chain.invoke({"query": "Which scientist proposed the Theory Of Relativity?"})
print(f"Final answer: {result['result']}")