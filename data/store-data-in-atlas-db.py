from langchain_astradb import AstraDBVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
import json
import time
from datetime import timedelta

TOTAL_DOCS = 1000000  # Total number of documents to upload

MODEL_NAME = "nomic-embed-text"
DOCUMENTS_FILE = "final-document-chunks.jsonl"



embedding = OllamaEmbeddings(
    model=MODEL_NAME,
)

astra_vectorstore = AstraDBVectorStore(
    collection_name="rag_documents",
    embedding=embedding,
    token=ASTRA_DB_APPLICATION_TOKEN,
    api_endpoint=ASTRA_DB_API_ENDPOINT,
)

def stream_documents(file_path, limit=None):
    """
    Stream documents from a JSONL file.

    Args:
        file_path (str): Path to the JSONL file.
        limit (int, optional): Maximum number of lines to stream. If None, stream all lines.

    Yields:
        dict: A document as a dictionary.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            yield json.loads(line.strip())


try:
    count = 0
    report_every = 10  # Report every N documents
    start_time = time.time()

    for raw_doc in stream_documents(DOCUMENTS_FILE):
        content = raw_doc["content"]
        metadata = {"source": raw_doc.get("source", "unknown")}
        doc = Document(page_content=content, metadata=metadata)

        astra_vectorstore.add_documents([doc])
        count += 1

        if count % report_every == 0:
            elapsed_time = time.time() - start_time
            avg_time_per_doc = elapsed_time / count
            remaining_docs = TOTAL_DOCS - count
            eta_seconds = remaining_docs * avg_time_per_doc
            eta = timedelta(seconds=int(eta_seconds))
            print(f"{count} documents uploaded... ETA to 1M: {eta}")

    print(f"Upload complete. Total documents uploaded: {count}")

except KeyboardInterrupt:
    elapsed_time = time.time() - start_time
    print(f"\nUpload interrupted by user. Documents uploaded: {count}")
    print(f"Elapsed time: {timedelta(seconds=int(elapsed_time))}")


