# from langchain_astradb import AstraDBVectorStore
# from langchain_ollama import OllamaEmbeddings
# from langchain_core.documents import Document
import os
import json
# from transformers import AutoTokenizer
from tqdm import tqdm  # For progress reporting


import json

# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")



MODEL_NAME = "nomic-embed-text"
DOCUMENTS_FILE = "refined-web-2m.jsonl"




# embedding = OllamaEmbeddings(
#     model=MODEL_NAME,
# )

# astra_vectorstore = AstraDBVectorStore(
#     collection_name="rag_documents",
#     embedding=embedding,
#     token=ASTRA_DB_APPLICATION_TOKEN,
#     api_endpoint=ASTRA_DB_API_ENDPOINT,
# )

# results = astra_vectorstore.similarity_search("I'm thinking a lot about how we can design accessible tools to help people unlock their potential — not just for productivity, but for growth, empathy, and connection. Technology shouldn't be reserved for the privileged. What if we built systems that supported every human, no matter their background? What kind of future are we building, and who gets left behind if we don’t rethink it?", k=3)
# for res in results:
#     print(f"* {res.page_content} [{res.metadata}]")


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


def count_tokens_in_document(document):
    """
    Count the total number of tokens in a document with progress reporting.

    Args:
        document (str): Path to the JSONL file containing documents.

    Returns:
        int: Total token count.
    """
    token_count = 0
    total_lines = 2_000_000  # Get the total number of lines for progress tracking

    with tqdm(total=total_lines, desc="Processing documents") as pbar:
        for doc in stream_documents(document):
            content = doc["content"]
            tokens = tokenizer.encode(content)
            token_count += len(tokens)
            pbar.update(1)  # Update progress bar for each document processed

    return token_count


# Example usage
# total_tokens = count_tokens_in_document("refined-web-2m.jsonl")
# print(f"Total tokens in document: {total_tokens}")
        




def extract_subset_of_documents(file_path, output_file, limit=None):
    """
    Extract a subset of documents from a JSONL file and save them to a new file.

    Args:
        file_path (str): Path to the input JSONL file.
        output_file (str): Path to the output JSONL file.
        limit (int, optional): Maximum number of lines to extract. If None, extract all lines.
    """
    with open(file_path, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
        for i, line in enumerate(f_in):
            if limit is not None and i >= limit:
                break
            f_out.write(line)

# Example usage
# extract_subset_of_documents(
#     file_path=DOCUMENTS_FILE,
#     output_file="refined-web-2m.jsonl",
#     limit=2_000_000,  # Adjust the limit as needed
# )


def get_last_line(file_path):
    """
    Get the last line of a file.

    Args:
        file_path (str): Path to the file.

    Returns:
        str: The last line of the file.
    """
    with open(file_path, "rb") as f:
        f.seek(-2, os.SEEK_END)  # Jump to the second last byte
        while f.read(1) != b"\n":  # Until we find a newline
            f.seek(-2, os.SEEK_CUR)  # Jump back two bytes
        return f.readline().decode("utf-8").strip()
    
# Example usage\
# last_line = get_last_line(DOCUMENTS_FILE)
# print(last_line)


# DOCUMENTS_FILE = "final-document-chunks.jsonl"
# output_file = "sample_documents.jsonl"
# # Stream 100 documents and save them to a new file
# with open(output_file, "w", encoding="utf-8") as f_out:
#     for doc in stream_documents(DOCUMENTS_FILE, limit=1000):
#         f_out.write(json.dumps(doc) + "\n")


def count_lines(file_path):
    """
    Count the number of lines in a file.

    Args:
        file_path (str): Path to the file.

    Returns:
        int: The total number of lines in the file.
    """
    line_count = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for _ in f:
            line_count += 1
    return line_count

# Example usage
DOCUMENTS_FILE = "final-document-chunks.jsonl"
# total_lines = count_lines(DOCUMENTS_FILE)
# print(f"The file '{DOCUMENTS_FILE}' contains {total_lines} lines.")


documents = []
with open("data/refined-web-2m-updated.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line.strip())
        if (data.get("response")):
            continue
        documents.append({
            "id": data["id"], 
            "content": data["content"],
          })
        
print (f"Loaded {len(documents)} documents from data/refined-web-2m-updated.jsonl")

random_docs = [
    """"Certificate Management with kubeadm
Kubernetes v1.15 [stable]
Client certificates generated by kubeadm expire after 1 year. This page explains how to manage certificate renewals with kubeadm. alpha. alpha certs renew command.
This command performs the renewal using CA (or front-proxy-CA) certificate and key stored in
/etc/kubernetes/pki.
Warning: If you are running an HA cluster, this command needs to be executed on all the control-plane nodes.
Note:
alpha certs renewuses the existing certificates as the authoritative source for attributes (Common Name, Organization, SAN, etc.) instead of the kubeadm-config ConfigMap. It is strongly recommended to keep them both in sync.
kubeadm alpha.2 kind: ClusterConfiguration controllerManager: extraArgs: cluster-signing-cert-file: /etc/kubernetes/pki/ca.crt cluster-signing-key-file: /etc/kubernetes/pki/ca.key
Create certificate signing requests (CSR)
You can create the certificate signing requests for the Kubernetes certificates API with
kubeadm alpha certs renew --use-api. &
The output is similar to this:
[1] 2890 [certs] certificate request "kubeadm-cert-kube-apiserver-ld526" created
Approve certificate signing requests (CSR)
If you set up an external signer, certificate signing requests (CSRs) are automatically approved.
Otherwise, you must manually approve certificates with the
kubectl certificate command. e.g.
kubectl certificate approve kubeadm-cert-kube-apiserver-ld526
The output is similar to this:
certificatesigningrequest.certificates.k8s.io/kubeadm-cert-kube-apiserver-ld526 approved
You can view a list of pending certificates with
kubectl get csr. alpha alpha."""
]

random_ids = [
     "4f464273-14ad-5980-b717-8d6665177e6b_0"
]

# check if the random_docs are in the documents
# count = 0
# for rand_id in random_ids:
#     for document in documents:
#         if rand_id in document["id"]:
#             print(document["content"])
#             # print(f"Found document: {document['id']}")
# print(f"Found {count} documents in the list.")

# check for duplicate ids
# id_set = set()
# for doc in documents:
#     if doc["id"] in id_set:
#         print(f"Duplicate id found: {doc['id']}")
#     else:
#         id_set.add(doc["id"])
