import matplotlib.pyplot as plt
import numpy as np
import json

import pandas as pd


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
            docs[data["id"]] = data["document"]
    return docs

def filter_away_true_or_false(docs, filter_away):
    counter = 0
    # stats = pd.read_csv("data/101k-test/chromadb-multistep-stats.csv")
    stats = pd.read_csv("data/101k-test/faiss-stats-k1000.csv")

    stats = stats.reset_index()
    for index, row in stats.iterrows():
        if row["correct_retrieval"] == filter_away:
            docs.pop(row["id"])
            counter += 1
    print(f"Removed {counter} documents from the dataset")
    return docs


# documents = load_documents("data/refined-web-100k-updated.jsonl")
documents = filter_away_true_or_false(load_documents("data/QA-pair-1000-huggingface.jsonl"), True)

# Choose whether to measure length in characters or words
measure_by_words = True

# Compute lengths
lengths = [
    len(content.split()) if measure_by_words else len(content)
    for content in documents.values()
]

# Create the histogram
plt.figure(figsize=(10, 6))
counts, bins, patches = plt.hist(lengths, bins=50, edgecolor='black')

# Add labels on top of bars
for count, patch in zip(counts, patches):
    if count > 0:
        x = patch.get_x() + patch.get_width() / 2
        y = patch.get_height()
        plt.text(x, y + 0.5, str(int(count)), ha='center', va='bottom', fontsize=9)

# Styling
plt.title('Document Length Distribution')
plt.xlabel('Length (in {})'.format('words' if measure_by_words else 'characters'))
plt.ylabel('Number of Documents')
plt.grid(True)
plt.tight_layout()
plt.show()