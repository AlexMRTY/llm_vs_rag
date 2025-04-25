import json
import time
from datetime import timedelta
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
import faiss
import numpy as np
import pickle
from tqdm import tqdm

# --- Config ---
TOTAL_DOCS_TO_PROCESS = 2_001_000
MODEL_NAME = "nomic-embed-text"
# DOCUMENTS_FILE = "final-document-chunks.jsonl"
DOCUMENTS_FILE = "refined-web-2m-updated.jsonl"
BATCH_SIZE = 64
FAISS_INDEX_PATH = "faiss_index-updated.index"
METADATA_PATH = "metadata-updated.pkl"

# --- Embedding Model ---
embedding = OllamaEmbeddings(model=MODEL_NAME)

# --- Storage ---
all_vectors = []
all_metadata = [] 
all_ids = []
dimension = None
doc_count = 0

# --- Streaming JSONL ---
def stream_documents(file_path, limit=None):
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            yield json.loads(line.strip())

try:
    count = 0
    report_every = 100
    start_time = time.time()
    batch = []
    metadata_batch = []

    for raw_doc in stream_documents(DOCUMENTS_FILE):
        content = raw_doc["content"]
        metadata = {"source": raw_doc.get("source", "unknown"), "id": raw_doc.get("id")}

        batch.append(content)
        metadata_batch.append(metadata)

        if len(batch) >= BATCH_SIZE:
            embeddings = embedding.embed_documents(batch)  # returns list of vectors
            if dimension is None:
                dimension = len(embeddings[0])

            all_vectors.append(np.array(embeddings, dtype="float32"))
            all_metadata.extend(metadata_batch)
            all_ids.extend([f"doc_{count + i}" for i in range(len(batch))])

            count += len(batch)
            batch = []
            metadata_batch = []

            if count % report_every == 0:
                elapsed_time = time.time() - start_time
                docs_per_second = count / elapsed_time  # Calculate documents per second
                remaining_docs = TOTAL_DOCS_TO_PROCESS - count
                eta_seconds = remaining_docs / docs_per_second if docs_per_second > 0 else float('inf')  # Avoid division by zero
                eta_hours = eta_seconds // 3600
                eta_minutes = (eta_seconds % 3600) // 60
                # By what time the process will be done
                done_by = time.strftime("%H:%M:%S", time.localtime(time.time() + eta_seconds))
                print(f"Processed {count} | Rate: {docs_per_second:.2f} Docs/sec | ETA: {int(eta_hours)}h {int(eta_minutes)}m | Done By: {done_by}")

    # Final batch
    if batch:
        embeddings = embedding.embed_documents(batch)
        all_vectors.append(np.array(embeddings, dtype="float32"))
        all_metadata.extend(metadata_batch)
        all_ids.extend([f"doc_{count + i}" for i in range(len(batch))])
        count += len(batch)

    print("🔄 Building FAISS index...")

    # Combine vectors and build FAISS index
    full_vector_array = np.vstack(all_vectors)
    index = faiss.IndexFlatL2(dimension)
    index.add(full_vector_array)

    # Save index and metadata
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump({"ids": all_ids, "metadata": all_metadata}, f)

    print(f"✅ Indexing complete. {count} documents embedded and saved.")
    print(f"📦 Index path: {FAISS_INDEX_PATH}")
    print(f"📝 Metadata path: {METADATA_PATH}")

except KeyboardInterrupt:
    elapsed = time.time() - start_time
    print(f"\n⛔ Interrupted at {count} documents. Time elapsed: {timedelta(seconds=int(elapsed))}")
