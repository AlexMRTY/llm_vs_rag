
import pprint
import json
import sys
import pandas as pd
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.retrievers.multi_query import MultiQueryRetriever
from tqdm import tqdm

def load_documents(file_path):
    print("🔄 Loading documents...")
    docs = {}
    # counter = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            # if counter == 1000:
            #     break
            # counter += 1
            data = json.loads(line.strip())
            docs[data["id"]] = data["content"]
    return docs

def load_QA_pairs(file_path):
    print("🔄 Loading QA pairs...")
    pairs = []
    # counter = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            # if counter == 10:
            #     break
            # counter += 1
            data = json.loads(line.strip())
            pairs.append({
                "id": data["id"],
                "content": data["document"],
                "question": data["question"],
                "answer": data["answer"],
                })
    return pairs

def contains_doc(doc, docs) -> bool:
    for d in docs:
        if doc["content"] in d.page_content:
            return True
    return False

def run_test(qa_pairs, retriever: MultiQueryRetriever) -> (pd.DataFrame, dict):
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
            query = pair["question"]
            results = retriever.get_relevant_documents(query)
            stats["id"].append(pair["id"])
            if contains_doc(pair, results):
                stats["correct_retrieval"].append(True)
                summary["correct"] += 1
            else:
                stats["correct_retrieval"].append(False)
                summary["incorrect"] += 1
            pbar.update(1)

    return pd.DataFrame(stats), summary



def main(output_path):
    embedding = OllamaEmbeddings(model="nomic-embed-text")
    # vector_store = Chroma(
    #     collection_name="101k_performance_test",
    #     embedding_function=embedding,
    #     persist_directory="data/101k-test/chroma"
    # )
    vector_store = FAISS.load_local(
        folder_path="data/101k-test/faiss_index-101k",
        embeddings=embedding,
        allow_dangerous_deserialization=True
    )

    llm = OllamaLLM(model="qwen2.5:14b-instruct")
    retriever = MultiQueryRetriever.from_llm(llm=llm, retriever=vector_store.as_retriever())

    qa_pairs = load_QA_pairs("data/QA-pair-1000-huggingface.jsonl")

    stats, summary = run_test(qa_pairs, retriever)

    stats.to_csv(output_path, index=False)
    pprint.pp(summary)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python multi-query-retrieval-test.py <path-for-output>")
        sys.exit(1)
    output = sys.argv[1]
    main(output)