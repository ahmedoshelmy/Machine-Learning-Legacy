import os
from embedder import embed_documents
from retriever import retrieve_documents
from generator import generate_response

def main():
    # Step 1: Embed documents
    if not os.path.exists("../embeddings/chromadb/chroma.sqlite3"):
        print("Embedding documents...")
        embed_documents()

    # Step 2: Query and generate response
    while True:
        query = input("Enter your query (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break

        print("Retrieving relevant documents...")
        results = retrieve_documents(query)

        print("Generating response...")
        response = generate_response(query)
        print(f"Response: {response}\n")

if __name__ == "__main__":
    main()
