import json




# Load Documents
import re

def split_documents(contexts):
    # context_text = contexts["context"]
    # Use regex to split while keeping the delimiter (Document x:)
    parts = re.split(r'(Document \d+:)', contexts)
    
    # Reconstruct documents using the marker and its following content
    documents = []
    for i in range(1, len(parts), 2):  # skip non-matching start if any
        doc_header = parts[i].strip()
        doc_body = parts[i+1].strip() if i+1 < len(parts) else ''
        documents.append(f"{doc_header} {doc_body}\n")
    
    return documents

QA_PAIR_PATH = "data/rag_test_result.jsonl"
documents = []
with open(QA_PAIR_PATH, "r", encoding="utf-8") as f:
    count = 0
    for line in f:
        if count > 3: break;
        data = json.loads(line.strip())
        if (data.get("response")):
            continue
        documents.append({
            "id": data["id"], 
            "question": data["question"], 
            "answer": data["answer"],
            "ground_truth": data["expected_answer"],
            "contexts": split_documents(data["context"]),
          })
        count += 1





from ragas.metrics import (
    faithfulness
)

from langchain_community.chat_models import ChatOllama
from ragas import evaluate
from langchain_community.embeddings import OllamaEmbeddings

langchain_llm = ChatOllama(model="qwen2.5:14b-instruct")
langchain_embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

result = evaluate(documents,
                  metrics=[faithfulness], llm=langchain_llm,embeddings=langchain_embeddings)