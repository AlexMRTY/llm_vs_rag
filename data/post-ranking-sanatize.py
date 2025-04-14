import json
from tqdm import tqdm


DOCUMENT_PATH = "data/refined-web-50k-post-ranking.jsonl"
OUTPUT_PATH = "data/refined-web-50k-post-ranking-sanatized.jsonl"


def load_documents(file_path):
    documents = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            documents.append({"id": data["id"], "content": data["content"], "score": data["score"]})
    return documents


documents = load_documents(DOCUMENT_PATH)
bad_doc_count = 0;
with tqdm(total=len(documents), desc="Sanatizing Documents", unit="docs") as pbar:
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
      for doc in documents:
          # Remove the score from the document
          if not isinstance(doc.get("score"), int):
              bad_doc_count += 1
              continue
          
          # Write the sanatized document to the output file
          f.write(json.dumps(doc) + "\n")
          pbar.update(1)

print(f"Total bad documents: {bad_doc_count}")
