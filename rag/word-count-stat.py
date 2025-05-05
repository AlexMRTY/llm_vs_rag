import matplotlib.pyplot as plt
import numpy as np
import json

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

# documents = load_documents("data/refined-web-100k-updated.jsonl")
documents = load_documents("data/QA-pair-1000-huggingface.jsonl")

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