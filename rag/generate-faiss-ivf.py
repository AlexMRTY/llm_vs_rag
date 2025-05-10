import json
import time
from datetime import timedelta

from langchain_community.docstore import InMemoryDocstore
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain.vectorstores import FAISS
import os
from tqdm import tqdm
import numpy as np
import faiss

# --- Config ---
TOTAL_DOCS_TO_PROCESS = 101_000
MODEL_NAME = "nomic-embed-text"
DOCUMENTS_FILE = "data/refined-web-100k-updated.jsonl"
BATCH_SIZE = 64
SAVE_PATH = "data/101k-test/faiss_index-101k-ivf"
N_LIST = 400  # Number of clusters for IVF
N_PROBE = 64  # Number of clusters to search at query time

# --- Embedding Model ---
embedding_model = OllamaEmbeddings(model=MODEL_NAME)

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
    texts = []
    count = 0

    for raw_doc in tqdm(stream_documents(DOCUMENTS_FILE, TOTAL_DOCS_TO_PROCESS), desc="Processing documents"):
        # if count >= 1000: break
        content = raw_doc.get("content", "").strip()
        if not content:
            continue

        metadata = {
            "source": raw_doc.get("source", "unknown"),
            "id": raw_doc.get("id", f"doc_{count}")
        }

        doc = Document(page_content=content, metadata=metadata)
        all_docs.append(doc)
        texts.append(content)
        count += 1

    print(f"🔢 Total documents processed: {len(all_docs)}")

    # --- Embedding ---
    print("🔄 Embedding documents...")
    embedded = embedding_model.embed_documents(texts)
    embeddings = np.array(embedded).astype("float32")

    # --- Build IVF index ---
    dim = embeddings.shape[1]
    quantizer = faiss.IndexFlatL2(dim)
    index_ivf = faiss.IndexIVFFlat(quantizer, dim, N_LIST)
    print("🧠 Training FAISS IVF index...")
    index_ivf.train(embeddings)
    index_ivf.add(embeddings)
    index_ivf.nprobe = N_PROBE

    # --- Wrap with LangChain FAISS ---
    # Build mapping of ID -> Document
    index_to_docstore_id = {i: str(i) for i in range(len(all_docs))}
    docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(all_docs)})

    # Build FAISS vector store
    vectorstore = FAISS(
        index=index_ivf,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
        embedding_function=embedding_model,
    )

    # --- Save index ---
    os.makedirs(SAVE_PATH, exist_ok=True)
    vectorstore.save_local(SAVE_PATH)

    elapsed = time.time() - start_time
    print(f"✅ Done! {count} documents indexed.")
    print(f"📦 FAISS IVF index saved at: {SAVE_PATH}")
    print(f"⏱️ Time elapsed: {timedelta(seconds=int(elapsed))}")

except KeyboardInterrupt:
    elapsed = time.time() - start_time
    print(f"\n⛔ Interrupted at {count} documents. Time elapsed: {timedelta(seconds=int(elapsed))}")
