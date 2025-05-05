import chromadb
import ollama
import pprint
import json

import pandas as pd
from langchain_ollama import OllamaEmbeddings
from tqdm import tqdm

def load_QA_pairs(file_path):
    print("🔄 Loading QA pairs...")
    pairs = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            pairs.append({
                "id": data["id"],
                "content": data["document"],
                "question": data["question"],
                "answer": data["answer"],
                })
    return pairs

def run_test(qa_pairs, collection, embedding) -> (pd.DataFrame, dict):
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
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=3
            )
            stats["id"].append(pair["id"])
            if pair["content"] in results['documents'][0]:
                stats["correct_retrieval"].append(True)
                summary["correct"] += 1
            else:
                stats["correct_retrieval"].append(False)
                summary["incorrect"] += 1
            pbar.update(1)
    return pd.DataFrame(stats), summary

def main():
    embedding = OllamaEmbeddings(model="nomic-embed-text")
    client = chromadb.PersistentClient("data/101k-test/chroma")
    collection = client.get_or_create_collection(name="101k_performance_test")

    qa_pairs = load_QA_pairs("data/QA-pair-1000-huggingface.jsonl")

    stats, summary = run_test(qa_pairs, collection, embedding)

    stats.to_csv("data/101k-test/chromadb-stats-1.csv", index=False)
    print(summary)

if __name__ == '__main__':
    main()