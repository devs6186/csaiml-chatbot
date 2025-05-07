import os
import json
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Paths
CLEANED_CHUNKS_PATH = "Data/cleaned_chunks.json"
FAISS_DIR = "Data/faiss"  # New directory for FAISS files
FAISS_INDEX_PATH = os.path.join(FAISS_DIR, "faiss_index_flatl2.index")
METADATA_PATH = os.path.join(FAISS_DIR, "metadata.json")

# Create the faiss directory if it doesn't exist
os.makedirs(FAISS_DIR, exist_ok=True)

# Add this before generating embeddings
def preprocess_content(chunks):
    for chunk in chunks:
        # Enhance course-related content structure
        if chunk.get("is_structured"):
            content = chunk["content"]
            # Convert table rows to natural language
            if "|" in content:
                parts = [p.strip() for p in content.split("|") if p.strip()]
                if len(parts) >= 2:
                    chunk["content"] = f"{parts[0]} includes: {', '.join(parts[1:])}"
    return chunks


# Load data
with open(CLEANED_CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Apply preprocessing to enhance course-related content
chunks = preprocess_content(chunks)  # Apply preprocessing

# Load embedding model
print("🔄 Loading embedding model...")
model = SentenceTransformer("BAAI/bge-large-en")  # or another BGE variant

# Generate embeddings
texts = [chunk["content"] for chunk in chunks]
print("🧠 Generating embeddings...")
embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
embeddings = np.array(embeddings).astype("float32")

# Create FAISS FlatL2 index
print("📦 Creating FAISS FlatL2 index...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save index
faiss.write_index(index, FAISS_INDEX_PATH)
print(f"✅ FAISS index saved to: {FAISS_INDEX_PATH}")

# Save metadata (for source tracing during retrieval)
metadata = [
    {
        "source": chunk["source"],
        "page": chunk["page"],
        "content": chunk["content"]
    }
    for chunk in chunks
]

with open(METADATA_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"✅ Metadata saved to: {METADATA_PATH}")
