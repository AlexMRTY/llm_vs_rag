
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import json
from tqdm import tqdm


MODEL_NAME = "llama3.3:70b-instruct-q8_0"
DOCUMENT_PATH = "data/QA-pair-first-iteration-sanatized.jsonl"
OUTPUT_PATH = "data/QA-pair-second-filter.jsonl"



def load_documents(file_path):
    documents = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            # if (data.get("response")):
            #     continue
            documents.append({
                "id": data["id"], 
                "content": data["document"], 
                "score": data["score"], 
                "question": data["question"], 
                "answer": data["answer"],
                "response": data["response"]
              })
    return documents


template = """
Given a question, classify it as either "suitable" or "unsuitable" for factual, open-domain question answering. A suitable question should be general-purpose, fact-seeking, and answerable based on verifiable or informative content. An unsuitable question may be overly subjective, vague, opinion-based, hypothetical, or not appropriate for factual QA.

It is critical to follow the output format exactly as specified below. And do not include any additional text or explanations.
Output Format:
Label: <suitable/unsuitable>

Question: {question}
"""

# model = OllamaLLM(model=MODEL_NAME)

def categorize_QA(documents):
    results = []
    try:
        # Initialize tqdm progress bar
        with tqdm(total=len(documents), unit="doc") as pbar:
            for i, doc in enumerate(documents, start=1):
                prompt = ChatPromptTemplate.from_template(template)
                chain = prompt | model
                response = chain.invoke({"question": doc["question"]})
              
                results.append({
                    "id": doc["id"],
                    "document": doc["content"],
                    "score": doc["score"],
                    "question": doc["question"],
                    "answer": doc["answer"],
                    "response": response,
                })
                
                # Update progress bar
                pbar.set_description(f"Processed {i} docs ")
                pbar.update(1)
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user. Saving progress...")

    return results

def post_process(documents):
    results = []
    try:
        # Initialize tqdm progress bar
        with tqdm(total=len(documents), unit="doc") as pbar:
            for i, doc in enumerate(documents, start=1):
                if "unsuitable" in doc["response"]: continue
                results.append({
                    "id": doc["id"],
                    "document": doc["content"],
                    "score": doc["score"],
                    "question": doc["question"],
                    "answer": doc["answer"],
                    "suitable": True,
                })
                # Update progress bar
                pbar.set_description(f"Processed {i} docs ")
                pbar.update(1)
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user. Saving progress...")

    return results

# Load documents 
# documents = load_documents(DOCUMENT_PATH)

# Run the categorizing
# result = categorize_QA(documents)

documents = load_documents("data/QA-pair-second-filter.jsonl")
result = post_process(documents)


with open("data/QA-pair-suitable.jsonl", "w", encoding="utf-8") as f:
    for res in result:
        f.write(json.dumps(res) + "\n")

print(f"Results have been written to {OUTPUT_PATH}")