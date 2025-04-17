import faiss
import pickle
import json
import numpy as np
from langchain_ollama import OllamaEmbeddings

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from tqdm import tqdm  # For progress reporting

# --- Config ---
FAISS_INDEX_PATH = "faiss/faiss_index.index"
METADATA_PATH = "faiss/metadata.pkl"
DOCUMENTS_PATH = "data/refined-web-2m.jsonl"
QA_PAIRS_PATH = "data/QA-pair-1000.jsonl"
EMBEDDING_MODEL_NAME = "nomic-embed-text"
LLM_MODEL_NAME = "llama3.1:8b-instruct-fp16"
TOP_K = 5

def load_documents(file_path):

    print("🔄 Loading documents...")
    documents = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            documents[data["id"]] = data["content"]
            # documents.append({
            #     data["id"]: data["content"],
            #     })
    return documents

def load_QA_pairs(file_path):

    print("🔄 Loading QA pairs...")
    documents = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            documents.append({
                "id": data["id"],
                "content": data["document"],
                "question": data["question"],
                "answer": data["answer"],
                })
    return documents

# --- Load Model, Index, Metadata ---
print("🔄 Loading FAISS index and metadata...")
embedding = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
model = OllamaLLM(model=LLM_MODEL_NAME)
index = faiss.read_index(FAISS_INDEX_PATH)

with open(METADATA_PATH, "rb") as f:
    meta = pickle.load(f)
    ids = meta["ids"]
    metadata = meta["metadata"]


documents = load_documents(DOCUMENTS_PATH)
qa_pairs = load_QA_pairs(QA_PAIRS_PATH)

template = """
Use the following pieces of context to answer the user question. This context retrieved from a knowledge base and you should use only the facts from the context to answer, even if the context contains wrong information.
Your answer must be based on the context. If the context does not contain the answer, just say that 'I don't know', don't try to make up an answer, use the context.
Don't address the context directly, but use it to answer the user question like it's your own knowledge.
Answer in short, use up to 10 words.

It's critical that the answer follows the output format exactly as specified below. And do not include any additional text or explanations.
Output format:
Answer: <answer>

Context:
{context}

Question: {question}
"""

def get_context(question):
    # Embed the question
    question_vector = embedding.embed_query(question)
    question_vector = np.array(question_vector, dtype="float32").reshape(1, -1)

    # Search the index
    D, I = index.search(question_vector, TOP_K)

    # Retrieve the top K documents
    counter = 0
    contexts = []
    for idx in I[0]:
        doc_meta = metadata[idx]
        doc_content = documents[doc_meta["id"]]
        contexts.append(f"Document {counter}: {doc_content}")
        counter += 1
    
    return "\n".join(contexts)


prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

with tqdm(total=len(qa_pairs), unit="doc") as pbar:
    with open("data/rag_test_result.jsonl", "w", encoding="utf-8") as f:
        nr_of_bad_docs = 0
        
        for i, qa in enumerate(qa_pairs, start=1):
            # Get the context for the current question
            context = get_context(qa["question"])
            response = chain.invoke({"context": context, "question": qa["question"]})
            try:
                # Extract the answer from the response
                lines = response.split("\n")
                answer = lines[0].split(": ", 1)[1]  # Extract the part after "Answer: "
                
            except IndexError or ValueError:
                # Handle cases where the response format is unexpected
                nr_of_bad_docs += 1
                answer = response
            
            f.write(json.dumps({
                "id": qa["id"],
                "content": qa["content"],
                "question": qa["question"],
                "expected_answer": qa["answer"],
                "answer": answer,
                "context": context,
            }) + "\n")
            # Update progress bar
            pbar.set_description(f"Processed {i} docs")
            pbar.update(1)

# # --- Get Query ---
# query = input("\n🔍 Enter your search query: ")

# # --- Embed Query ---
# query_vector = embedding.embed_query(query)
# query_vector = np.array(query_vector, dtype="float32").reshape(1, -1)

# # --- Search ---
# D, I = index.search(query_vector, TOP_K)

# # --- Show Results ---
# print(f"\n📄 Top {TOP_K} Results for: '{query}'\n")
# for rank, idx in enumerate(I[0]):
#     doc_id = ids[idx]
#     doc_meta = metadata[idx]
#     print(f"[{rank + 1}] ID: {doc_id} | Source: {doc_meta['id']}")
#     print("-" * 60)
