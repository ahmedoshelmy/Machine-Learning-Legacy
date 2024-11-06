# Import libraries for document loaders
from langchain_community.document_loaders import PyPDFLoader, UnstructuredHTMLLoader
from langchain_community.document_loaders.csv_loader import CSVLoader

# Import libraries for text splitters
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

# Import libraries for embeddings and vector stores
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# Import libraries for prompt templates and chains
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RunnablePassthrough

# Import libraries for markdown and Python loaders
from langchain_community.document_loaders import UnstructuredMarkdownLoader, PythonLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import Language

# Import libraries for semantic chunking
from langchain_text_splitters import SemanticChunker

# Import libraries for token text splitting
import tiktoken
from langchain_text_splitters import TokenTextSplitter

# Import BM25 retriever
from langchain.retrievers import BM25Retriever

# Import evaluation metrics
from ragas.metrics import context_precision, faithfulness
from ragas.evaluation import EvaluatorChain, LangChainStringEvaluator

# Create a document loader for rag_vs_fine_tuning.pdf
loader = PyPDFLoader('rag_vs_fine_tuning.pdf')

# Load the document
data = loader.load()
print(data[0])

# Import library
from langchain_community.document_loaders.csv_loader import CSVLoader

# Create a document loader for fifa_countries_audience.csv
loader = CSVLoader('fifa_countries_audience.csv')

# Load the document
data = loader.load()
print(data[0])

from langchain_community.document_loaders import UnstructuredHTMLLoader

# Create a document loader for unstructured HTML
loader = UnstructuredHTMLLoader('white_house_executive_order_nov_2023.html')

# Load the document
data = loader.load()

# Print the first document
print(data[0])

# Print the first document's metadata
print(data[0].metadata)

# Import the character splitter
from langchain_text_splitters import CharacterTextSplitter

quote = 'Words are flowing out like endless rain into a paper cup,\nthey slither while they pass,\nthey slip away across the universe.'
chunk_size = 24
chunk_overlap = 10

# Create an instance of the splitter class
splitter = CharacterTextSplitter(
    separator = '\n',
    chunk_size = 24 , 
    chunk_overlap = 10 
)

# Split the string and print the chunks
docs = splitter.split_text(quote)
print(docs)
print([len(doc) for doc in docs])

# Import the recursive character splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

quote = 'Words are flowing out like endless rain into a paper cup,\nthey slither while they pass,\nthey slip away across the universe.'
chunk_size = 24
chunk_overlap = 10

# Create an instance of the splitter class
splitter = RecursiveCharacterTextSplitter(
    separators = ['\n'," ",""],
    chunk_size = 24 , 
    chunk_overlap = 10 
)

# Split the document and print the chunks
docs = splitter.split_text(quote)
print(docs)
print([len(doc) for doc in docs])

# Load the HTML document into memory
loader = UnstructuredHTMLLoader('white_house_executive_order_nov_2023.html')
data = loader.load()

# Define variables
chunk_size = 300
chunk_overlap = 100

# Split the HTML
splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=["."])

docs = splitter.split_documents(data)
print(docs)

loader = PyPDFLoader('rag_vs_fine_tuning.pdf')
data = loader.load()

# Split the document using RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_overlap=50,chunk_size=300)
docs = splitter.split_documents(data) 

# Embed the documents in a persistent Chroma vector database
embedding_function = OpenAIEmbeddings(api_key='<OPENAI_API_TOKEN>', model='text-embedding-3-small')
vectorstore = Chroma.from_documents(
    docs,
    embedding=embedding_function,
    persist_directory=os.getcwd()
)

# Configure the vector store as a retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k":3}
)

# Add placeholders to the message string
message = """
Answer the following question using the context provided:

Context:
{context}

Question:
{question}

Answer:
"""

# Create a chat prompt template from the message string
prompt_template = ChatPromptTemplate.from_messages([("human", message)])


vectorstore = Chroma.from_documents(
    docs,
    embedding=OpenAIEmbeddings(api_key='<OPENAI_API_TOKEN>', model='text-embedding-3-small'),
    persist_directory=os.getcwd()
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# Create a chain to link retriever, prompt_template, and llm
rag_chain = ({"context": retriever, "question": RunnablePassthrough()}
            | prompt_template
            | llm)

# Invoke the chain
response = rag_chain.invoke("Which popular LLMs were considered in the paper?")
print(response.content)


# Create a document loader for README.md and load it
loader = UnstructuredMarkdownLoader('README.md')

markdown_data = loader.load()
print(markdown_data[0])

# Create a document loader for rag.py and load it
loader = PythonLoader('rag.py')

python_data = loader.load()
print(python_data[0])

# Create a Python-aware recursive character splitter
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=300, chunk_overlap=100
)

# Split the Python content into chunks
chunks = python_splitter.split_documents(python_data)

for i, chunk in enumerate(chunks[:3]):
    print(f"Chunk {i+1}:\n{chunk.page_content}\n")


# Instantiate an OpenAI embeddings model
embedding_model = OpenAIEmbeddings(api_key="<OPENAI_API_TOKEN>", model='text-embedding-3-small')

# Create the semantic text splitter with desired parameters
semantic_splitter = SemanticChunker(
    embeddings=embedding_model, breakpoint_threshold_type="gradient", breakpoint_threshold_amount=0.8
)

# Split the document
chunks = semantic_splitter.split_documents(document)
print(chunks[0])

# Get the encoding for gpt-4o-mini
encoding = tiktoken.encoding_for_model('gpt-4o-mini')

# Create a token text splitter
token_splitter = TokenTextSplitter(encoding_name=encoding.name, chunk_size=100, chunk_overlap=10)

# Split the PDF into chunks
chunks = token_splitter.split_documents(document)

for i, chunk in enumerate(chunks[:3]):
    print(f"Chunk {i+1}:\nNo. tokens: {len(encoding.encode(chunk.page_content))}\n{chunk}\n")


chunks = [
    "RAG stands for Retrieval Augmented Generation.",
    "Graph Retrieval Augmented Generation uses graphs to store and utilize relationships between documents in the retrieval process.",
    "There are different types of RAG architectures; for example, Graph RAG."
]

# Initialize the BM25 retriever
bm25_retriever = BM25Retriever.from_texts(chunks,k=3)

# Invoke the retriever
results = bm25_retriever.invoke("Graph RAG")

# Extract the page content from the first result
print("Most Relevant Document:")
print(results[0].page_content)

# Create a BM25 retriever from chunks
retriever = BM25Retriever.from_documents(chunks,k=5)

# Create the LCEL retrieval chain
chain = ({"context": retriever, "question": RunnablePassthrough()}
         | prompt
         | llm
         | StrOutputParser()
)

print(chain.invoke("What are knowledge-intensive tasks?"))

############# Evaluating RAG ##########################
from ragas.metrics import context_precision

# Define the context precision chain
context_precision_chain = EvaluatorChain(metric=context_precision, llm=llm, embeddings=embeddings)

# Evaluate the context precision of the RAG chain
eval_result = context_precision_chain({
  "question": "How does RAG enable AI applications?",
  "ground_truth": "RAG enables AI applications by integrating external data in generative models.",
  "contexts": [
    "RAG enables AI applications by integrating external data in generative models.",
    "RAG enables AI applications such as semantic search engines, recommendation systems, and context-aware chatbots."
  ]
})

print(f"Context Precision: {eval_result['context_precision']}")

# Create the QA string evaluator
qa_evaluator = LangChainStringEvaluator(
    "qa",
    config={
        "llm": eval_llm,
        "prompt": prompt_template
    }
)

query = "How does RAG improve question answering with LLMs?"

# Evaluate the RAG output by evaluating strings
score = qa_evaluator.evaluator.evaluate_strings(
    prediction=predicted_answer,
    reference=ref_answer,
    input=query
)

print(f"Score: {score}")


from ragas.metrics import faithfulness

# Query the retriever using the query and extract the document text
query = "How does RAG improve question answering with LLMs?"
retrieved_docs = [doc.page_content for doc in retriever.invoke(query)]

# Define the faithfulness chain
faithfulness_chain = EvaluatorChain(metric=faithfulness, llm=llm, embeddings=embeddings)

# Evaluate the faithfulness of the RAG chain
eval_result = faithfulness_chain({
  "question": query,
  "answer": chain.invoke(query),
  "contexts": retrieved_docs
})

print(eval_result)