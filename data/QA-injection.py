from datasets import load_dataset
import json
import random
from tqdm import tqdm



DOCUMENT_PATH = "data/refined-web-2m.jsonl"
OUTPUT_PATH_REFINED_WEB = "data/refined-web-2m-updated.jsonl"
OUTPUT_QA_PAIRS = "data/QA-pair-1000-huggingface.jsonl"

huggingface_QA_pairs = load_dataset("pinecone/refinedweb-generated-questions", split="train")


documents = []
with open(DOCUMENT_PATH, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line.strip())
        if (data.get("response")):
            continue
        documents.append({
            "id": data["id"], 
            "content": data["content"],
          })
        
print (f"Loaded {len(documents)} documents from {DOCUMENT_PATH}")

# Add content from Huggingface questions to the documents
with tqdm(total=len(huggingface_QA_pairs), unit="docs") as pbar:
    for pair in huggingface_QA_pairs:
        random_index = random.randint(0, len(documents))  # Generate a random index
        documents.insert(random_index, {
            "id": pair["document_id"],
            "content": pair["documnet_text"]
        })
        pbar.update(1)

print (f"Added {len(huggingface_QA_pairs)} QA pairs to the documents. Total documents: {len(documents)}")
# Write the updated documents to the output file
with open(OUTPUT_PATH_REFINED_WEB, "w", encoding="utf-8") as f:
    for doc in documents:
        f.write(json.dumps(doc) + "\n")
print(f"Results have been written to {OUTPUT_PATH_REFINED_WEB}")


with open(OUTPUT_QA_PAIRS, "w", encoding="utf-8") as f:
    for pair in huggingface_QA_pairs:
        f.write(json.dumps({
            "id": pair["document_id"],
            "document": pair["documnet_text"],
            "question": pair["generated_question"],
            "answer": pair["generated_answer"],
            "score": 5,
            "suitable": True,
        }) + "\n")
print(f"QA pairs have been written to {OUTPUT_QA_PAIRS}")

