import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Paths
FAISS_DIR = "Data/faiss"
FAISS_INDEX_PATH = os.path.join(FAISS_DIR, "faiss_index_flatl2.index")
METADATA_PATH = os.path.join(FAISS_DIR, "metadata.json")

# Load embedding model
print("🔄 Loading embedding model for retrieval...")
model = SentenceTransformer("BAAI/bge-large-en")

def search(query, top_k=5):
    """
    Searches the FAISS index for the most relevant chunks to the query.

    Args:
        query (str): The search query.
        top_k (int): The number of top results to retrieve.

    Returns:
        list: A list of dictionaries, where each dictionary contains the
              content and metadata of a retrieved chunk.
    """
    try:
        # Load FAISS index
        index = faiss.read_index(FAISS_INDEX_PATH)

        # Load metadata
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # Encode the query
        query_embedding = model.encode([query], normalize_embeddings=True).astype("float32")

        # Search the index
        distances, indices = index.search(query_embedding, top_k)

        # Retrieve results
        results = []
        for i in range(len(indices[0])):
            result_index = indices[0][i]
            if result_index < len(metadata):
                results.append(metadata[result_index])
            else:
                print(f"Warning: Index {result_index} out of bounds in metadata.")

        return results

    except FileNotFoundError:
        print("Error: FAISS index or metadata file not found. Please run 'embed_and_index.py' first.")
        return []

if __name__ == "__main__":
    query = "What are the academic programs offered?"
    results = search(query)
    print(f"\n🔍 Query: {query}\n")
    for i, result in enumerate(results):
        print(f"--- Result {i+1} ---")
        print(f"Content: {result['content'][:200]}...") # Display first 200 characters
        print(f"Source: {result['source']}, Page: {result['page']}")
        print("-" * 20)