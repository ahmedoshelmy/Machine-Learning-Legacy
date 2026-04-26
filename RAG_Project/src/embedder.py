import os
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

def embed_documents():
    # Load documents
    documents_path = "../data/documents"
    documents = []
    for filename in os.listdir(documents_path):
        with open(os.path.join(documents_path, filename), 'r', encoding='utf-8') as file:
            documents.append(file.read())

    # Initialize Hugging Face embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Initialize ChromaDB
    chroma_db_path = "../embeddings/chromadb"
    vector_store = Chroma(persist_directory=chroma_db_path, embedding_function=embedding_model)

    # Add documents to ChromaDB
    vector_store.add_texts(documents)
    vector_store.persist()

if __name__ == "__main__":
    embed_documents()
