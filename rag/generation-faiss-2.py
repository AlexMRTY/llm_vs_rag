import json
import time
from datetime import timedelta
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain.vectorstores import FAISS
import os
from tqdm import tqdm

# --- Config ---
TOTAL_DOCS_TO_PROCESS = 101_000
MODEL_NAME = "nomic-embed-text"
DOCUMENTS_FILE = "data/refined-web-100k-updated.jsonl"
BATCH_SIZE = 64
SAVE_PATH = "data/101k-test/faiss_index-101k"

# --- Embedding Model ---
embedding = OllamaEmbeddings(model=MODEL_NAME)

# --- Streaming JSONL ---
def stream_documents(file_path, limit=None):
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            yield json.loads(line.strip())

# --- Processing & Indexing ---
try:
    print("🚀 Starting embedding and indexing process...")
    start_time = time.time()
    all_docs = []
    batch = []
    count = 0

    for raw_doc in tqdm(stream_documents(DOCUMENTS_FILE, TOTAL_DOCS_TO_PROCESS), desc="Processing documents"):
        content = raw_doc.get("content", "").strip()
        if not content:
            continue

        metadata = {
            "source": raw_doc.get("source", "unknown"),
            "id": raw_doc.get("id", f"doc_{count}")
        }

        doc = Document(page_content=content, metadata=metadata)
        batch.append(doc)
        count += 1

        if len(batch) >= BATCH_SIZE:
            all_docs.extend(batch)
            batch = []

    if batch:
        all_docs.extend(batch)

    print(f"🔢 Total documents processed: {len(all_docs)}")

    # Embed and build FAISS index
    print("🔄 Embedding and building FAISS index...")
    faiss_index = FAISS.from_documents(all_docs, embedding)

    # Save the index and metadata
    os.makedirs(SAVE_PATH, exist_ok=True)
    faiss_index.save_local(SAVE_PATH)

    elapsed = time.time() - start_time
    print(f"✅ Done! {count} documents indexed.")
    print(f"📦 FAISS index saved at: {SAVE_PATH}")
    print(f"⏱️ Time elapsed: {timedelta(seconds=int(elapsed))}")

except KeyboardInterrupt:
    elapsed = time.time() - start_time
    print(f"\n⛔ Interrupted at {count} documents. Time elapsed: {timedelta(seconds=int(elapsed))}")
