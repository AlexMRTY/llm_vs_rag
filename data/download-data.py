import time
import json
from datasets import load_dataset
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import uuid

# # Load Falcon tokenizer
# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
#
# # Token count helper
# def count_tokens(text: str) -> int:
#     return len(tokenizer.encode(text))
#
# # Langchain splitter (512-token chunks)
# splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
#     tokenizer=tokenizer,
#     chunk_size=512,
#     chunk_overlap=0,
# )

# def stream_dataset(dataset_name, output_path="data/refined-web-8m.jsonl"):
#     print(f"Streaming dataset: {dataset_name}")
#     dataset = load_dataset(dataset_name, streaming=True, split="train")
#
#     total_examples = 0
#     total_chunks = 0
#     start_time = time.time()
#
#     try:
#         with open(output_path, "a", encoding="utf-8") as f:
#             for example in dataset:
#                 text = example.get("content") or ""
#                 token_count = count_tokens(text)
#
#                 # Filter by token count
#                 if 20 <= token_count <= 4096:
#                     chunks = splitter.split_text(text)
#                     for chunk in chunks:
#                         # Add a unique ID for each chunk
#                         unique_id = str(uuid.uuid4())
#                         f.write(json.dumps({"id": unique_id, "content": chunk}) + "\n")
#                     total_chunks += len(chunks)
#
#                 total_examples += 1
#                 if total_examples % 1000 == 0:
#                     elapsed = time.time() - start_time
#                     throughput = total_examples / elapsed
#                     file_size = os.path.getsize(output_path) / (1024 * 1024)  # Convert to MB
#                     print(f"Processed {total_examples} examples, {total_chunks} chunks. Throughput: {throughput:.2f} ex/s. File size: {file_size:.2f} MB")
#     except KeyboardInterrupt:
#         print("\n🛑 Interrupted by user. Saving progress...")
#
#     elapsed = time.time() - start_time
#     print(f"\nDone. {total_examples} examples → {total_chunks} chunks in {elapsed:.2f}s")
#     print(f"Avg throughput: {total_examples / elapsed:.2f} ex/s")

def download_dataset(dataset_name, output_path="data/refined-web-8m.jsonl"):
    print(f"Downloading dataset: {dataset_name}")
    dataset = load_dataset("AlexMRTY/refinedWeb-subset")

    dataset.save_to_disk(output_path)

    print(f"Dataset saved to {output_path}")

if __name__ == "__main__":
    download_dataset("AlexMRTY/refinedWeb-subset")

