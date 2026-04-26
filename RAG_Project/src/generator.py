from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

def generate_response(query):
    # Initialize ChromaDB retriever
    chroma_db_path = "../embeddings/chromadb"
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma(persist_directory=chroma_db_path, embedding_function=embedding_model)
    retriever = vector_store.as_retriever()

    # Initialize Hugging Face LLM
    generator = pipeline("text-generation", model="gpt2")
    llm = HuggingFacePipeline(pipeline=generator)

    # Create RAG chain
    qa_chain = RetrievalQA(llm=llm, retriever=retriever)

    # Generate response
    response = qa_chain.run(query)
    return response

if __name__ == "__main__":
    query = input("Enter your query: ")
    response = generate_response(query)
    print(f"Response: {response}")
