import json
from tqdm import tqdm


DOCUMENT_PATH = "data/QA-pair-first-iteration.jsonl"
OUTPUT_PATH = "data/QA-pair-first-iteration-sanatized.jsonl"


def load_documents(file_path):
    documents = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            if (data.get("response")):
                continue
            documents.append({
                "id": data["id"], 
                "content": data["document"], 
                "score": data["score"], 
                "question": data["question"], 
                "answer": data["answer"]
              })
    return documents

def count_words(text):
    """
    Count the number of words in a string.
    """
    return len(text.split())

def include_word_purpose(text):
    """
    Check if the text contains the word "purpose".
    """
    return "purpose" in text.lower()

documents = load_documents(DOCUMENT_PATH)
doc_over_10 = 0
doc_with_purpose = 0

with tqdm(total=len(documents), unit="docs") as pbar:
    with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
        for doc in documents:
            if count_words(doc.get("answer")) > 10:
                doc_over_10 += 1
                continue
            elif include_word_purpose(doc.get("answer")):
                doc_with_purpose += 1
                continue
                

            # Write the sanatized document to the output file
            f.write(json.dumps(doc) + "\n")
            pbar.update(1)
print(f"Total documents with more than 10 words in answer: {doc_over_10}")
print(f"Total documents with the word 'purpose' in answer: {doc_with_purpose}")
print(f"Results have been written to {OUTPUT_PATH}")