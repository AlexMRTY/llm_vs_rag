
import json
import random


def stream_documents(file_path, limit=None):
    """
    Stream documents from a JSONL file.

    Args:
        file_path (str): Path to the JSONL file.
        limit (int, optional): Maximum number of lines to stream. If None, stream all lines.

    Yields:
        dict: A document as a dictionary.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            yield json.loads(line.strip())



def pick_random_subset(input_file, output_file, num_samples):
    """
    Pick a random subset of documents from a JSONL file and save them to a new file.

    Args:
        input_file (str): Path to the input JSONL file.
        output_file (str): Path to the output JSONL file.
        num_samples (int): Number of random samples to pick.
    """

    # Read all documents from the input file
    documents = list(stream_documents(input_file))

    # Randomly sample the specified number of documents
    sampled_documents = random.sample(documents, num_samples)

    # Write the sampled documents to the output file
    with open(output_file, "w", encoding="utf-8") as f:
        for doc in sampled_documents:
            f.write(json.dumps(doc) + "\n")

if __name__ == "__main__":
    input_file = "data/refined-web-2m.jsonl"
    output_file = "data/refined-web-50k-random.jsonl"
    num_samples = 50_000

    pick_random_subset(input_file, output_file, num_samples)