import json
from langchain_ollama import OllamaEmbeddings


def load_documents(file_path):
    print("🔄 Loading documents...")
    docs = {}
    counter = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if counter == 10:
                break
            counter += 1
            data = json.loads(line.strip())
            docs[data["id"]] = data["content"]
    return docs

def main():
    embedding = OllamaEmbeddings(model="nomic-embed-text")

    documents = load_documents("data/refined-web-100k-updated.jsonl")
    # documents = load_documents("data/QA-pair-1000-huggingface.jsonl")

    for doc in documents.keys():
        query_embedding = embedding.embed_query(documents[doc])
        print(len(query_embedding))

if __name__ == '__main__':
    main()