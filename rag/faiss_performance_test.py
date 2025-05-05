import pandas as pd

import faiss
import json
import numpy as np
import pickle

from langchain_ollama import OllamaEmbeddings

from tqdm import tqdm  # For progress reporting

def load_documents(file_path):
    print("🔄 Loading documents...")
    docs = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            docs[data["id"]] = data["content"]
    return docs

def load_QA_pairs(file_path):
    print("🔄 Loading QA pairs...")
    docs = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            docs.append({
                "id": data["id"],
                "content": data["document"],
                "question": data["question"],
                "answer": data["answer"],
            })
    return docs

def get_context(question_vector, docs, k, index, metadata):
    # Embed the question
    # question_vector = embedding.embed_query(question)
    question_vector = np.array(question_vector, dtype="float32").reshape(1, -1)

    # Search the index
    D, I = index.search(question_vector, k)

    # Retrieve the top K documents
    counter = 0
    contexts = []
    for idx in I[0]:
        doc_meta = metadata[idx]
        doc_content = docs[doc_meta["id"]]
        contexts.append(doc_content)
        counter += 1

    return contexts

def run_test(qa_pairs, index, metadata, embedding, documents) -> (pd.DataFrame, dict):
    stats = {
        "id": [],
        "correct_retrieval": []
    }
    summary = {
        "correct": 0,
        "incorrect": 0,
    }
    with tqdm(total=len(qa_pairs), desc="Testing: ") as pbar:
        for i, pair in enumerate(qa_pairs, start=1):
            query_embedding = embedding.embed_query(pair["question"])
            context = get_context(query_embedding, documents, 3, index, metadata)

            stats["id"].append(pair["id"])
            if pair["content"] in context:
                stats["correct_retrieval"].append(True)
                summary["correct"] += 1
            else:
                stats["correct_retrieval"].append(False)
                summary["incorrect"] += 1
            pbar.update(1)

    return pd.DataFrame(stats), summary

def main():

    embedding = OllamaEmbeddings(model="nomic-embed-text")
    faiss_index = faiss.read_index("data/101k-test/faiss_index-101k.index")

    with open("data/101k-test/metadata-101k.pkl", "rb") as f:
        meta = pickle.load(f)
        ids = meta["ids"]
        faiss_metadata = meta["metadata"]

    qa_pairs = load_QA_pairs("data/QA-pair-1000-huggingface.jsonl")
    docs = load_documents("data/refined-web-100k-updated.jsonl")

    stats, summary = run_test(qa_pairs, faiss_index, faiss_metadata, embedding, docs)

    stats.to_csv("data/101k-test/faiss-stats-1.csv", index=False)
    print(summary)


if __name__ == '__main__':
    main()