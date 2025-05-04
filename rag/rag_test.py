import faiss
import pickle
import json
import numpy as np
import sys
import re
from langchain_ollama import OllamaEmbeddings

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from tqdm import tqdm  # For progress reporting


def load_documents(file_path):
    print("🔄 Loading documents...")
    docs = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            docs[data["id"]] = data["content"]
            # documents.append({
            #     data["id"]: data["content"],
            #     })
    return docs


def load_QA_pairs(file_path):
    print("🔄 Loading QA pairs...")
    docs = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            docs.append({
                "id": data["id"],
                "content": data["document"],
                "question": data["question"],
                "answer": data["answer"],
            })
    return docs


def get_context(question, docs, k, index, metadata):
    # Embed the question
    question_vector = embedding.embed_query(question)
    question_vector = np.array(question_vector, dtype="float32").reshape(1, -1)

    # Search the index
    D, I = index.search(question_vector, k)

    # Retrieve the top K documents
    counter = 0
    contexts = []
    for idx in I[0]:
        doc_meta = metadata[idx]
        doc_content = docs[doc_meta["id"]]
        contexts.append(f"Document {counter}: {doc_content}")
        counter += 1

    return "\n".join(contexts)


# Test for different values of k (1, 2, 3, 4, 5)

def remove_think_block(text):
    # This pattern matches the <think>...</think> block, including line breaks
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)


def run_k(k, questions, docs, llm, model_name, index, metadata):
    with tqdm(total=len(questions), unit="doc") as pbar:
        with open(f"data/{model_name}_k{k}.jsonl", "w", encoding="utf-8") as f:
            nr_of_bad_docs = 0

            for i, qa in enumerate(questions, start=1):
                # Get the context for the current question
                context = get_context(qa["question"], docs, k, index, metadata)
                relevant_doc_in_context = True if qa["content"] in context else False
                response = chain.invoke({"context": context, "question": qa["question"]})
                parsed_response = remove_think_block(response)
                try:
                    # Extract the answer from the response
                    lines = parsed_response.split("\n")
                    answer = lines[0].split(": ", 1)[1]  # Extract the part after "Answer: "

                except IndexError or ValueError:
                    # Handle cases where the response format is unexpected
                    nr_of_bad_docs += 1
                    answer = parsed_response

                f.write(json.dumps({
                    "id": qa["id"],
                    "content": qa["content"],
                    "question": qa["question"],
                    "expected_answer": qa["answer"],
                    "relevant_doc_in_context": relevant_doc_in_context,
                    "answer": answer,
                    "context": context,
                }) + "\n")
                # Update progress bar
                pbar.set_description(f"K: {k}. Processed {i} docs")
                pbar.update(1)
            print(f"\n\n🔄 Processed {len(questions)} documents with {k} context documents. Bad docs: {nr_of_bad_docs}")


# --- Config ---
FAISS_INDEX_PATH = "faiss/faiss_index-updated.index"
METADATA_PATH = "faiss/metadata-updated.pkl"
DOCUMENTS_PATH = "data/refined-web-2m-updated.jsonl"
QA_PAIRS_PATH = "data/QA-pair-1000-huggingface.jsonl"
EMBEDDING_MODEL_NAME = "nomic-embed-text"
LLM_MODEL_NAME = "llama3.1:8b-instruct-fp16"
TOP_K = 3  # Default

if __name__ == "__main__":

    if len(sys.argv) < 4:
        print("Usage: python rag_test.py <model_name> <embedding_model_name> <k-value>")
        sys.exit(1)
    llm_model_name = sys.argv[1]
    embedding_model_name = sys.argv[2]
    k_value = int(sys.argv[3]) if len(sys.argv) > 3 else TOP_K
    print(f"Running RAG test with LLM: {llm_model_name} and Embedding Model: {embedding_model_name}...")

    # --- Load Model, Index, Metadata ---
    print("🔄 Loading FAISS index and metadata...")
    embedding = OllamaEmbeddings(model=embedding_model_name)
    model = OllamaLLM(model=llm_model_name)
    faiss_index = faiss.read_index(FAISS_INDEX_PATH)

    with open(METADATA_PATH, "rb") as f:
        meta = pickle.load(f)
        ids = meta["ids"]
        faiss_metadata = meta["metadata"]

    template = """
    Use the following pieces of context to answer the user question. This context retrieved from a knowledge base and you should use only the facts from the context to answer, even if the context contains wrong information.
    Your answer must be based on the context. If the context does not contain the answer, just say that 'I don't know', don't try to make up an answer, use the context.
    Don't address the context directly, but use it to answer the user question like it's your own knowledge.
    Answer in short, use up to 10 words.
    
    Context:
    {context}

    Question: {question}

    It's critical that the answer follows the output format exactly as specified below. And do not include any additional text or explanations.
    Output format:
    Answer: <answer>

    
    """

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model

    documents = load_documents(DOCUMENTS_PATH)
    qa_pairs = load_QA_pairs(QA_PAIRS_PATH)

    # --- Run the test ---
    run_k(k_value, qa_pairs, documents, model, llm_model_name, faiss_index, faiss_metadata)

    # for k in [1, 2, 3, 4, 5]:
    #     run_k(k, qa_pairs, documents, model, faiss_index, faiss_metadata)
