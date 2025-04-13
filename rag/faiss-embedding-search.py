import faiss
import pickle
import numpy as np
from langchain_ollama import OllamaEmbeddings

# --- Config ---
FAISS_INDEX_PATH = "faiss_index.index"
METADATA_PATH = "metadata.pkl"
MODEL_NAME = "nomic-embed-text"
TOP_K = 5

# --- Load Model, Index, Metadata ---
print("🔄 Loading FAISS index and metadata...")
embedding = OllamaEmbeddings(model=MODEL_NAME)
index = faiss.read_index(FAISS_INDEX_PATH)

with open(METADATA_PATH, "rb") as f:
    meta = pickle.load(f)
    ids = meta["ids"]
    metadata = meta["metadata"]

# --- Get Query ---
query = input("\n🔍 Enter your search query: ")

# --- Embed Query ---
query_vector = embedding.embed_query(query)
query_vector = np.array(query_vector, dtype="float32").reshape(1, -1)

# --- Search ---
D, I = index.search(query_vector, TOP_K)

# --- Show Results ---
print(f"\n📄 Top {TOP_K} Results for: '{query}'\n")
for rank, idx in enumerate(I[0]):
    doc_id = ids[idx]
    doc_meta = metadata[idx]
    print(f"[{rank + 1}] ID: {doc_id} | Source: {doc_meta['id']}")
    print("-" * 60)
