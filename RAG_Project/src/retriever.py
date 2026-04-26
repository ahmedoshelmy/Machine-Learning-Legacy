from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

def retrieve_documents(query):
    # Initialize ChromaDB
    chroma_db_path = "../embeddings/chromadb"
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma(persist_directory=chroma_db_path, embedding_function=embedding_model)

    # Retrieve relevant documents
    results = vector_store.similarity_search(query, k=5)
    return results

if __name__ == "__main__":
    query = input("Enter your query: ")
    results = retrieve_documents(query)
    for i, result in enumerate(results):
        print(f"Result {i+1}: {result.page_content}\n")
