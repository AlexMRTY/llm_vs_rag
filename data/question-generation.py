from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from tqdm import tqdm  # For progress reporting
import json

MODEL_NAME = "llama3.1:70b"
DOCUMENTS_FILE = "data/refined-web-50k-post-ranking-sanatized.jsonl"
OUTPUT_FILE = "data/QA-pair-first-iteration.jsonl"




# Define the instruction prompt
# It is creitical to follow the output format exactly as specified below. And do not include any additional text or explanations.
template = """
Given a document, generate one pair consisting of a factual-style question and its corresponding answer. The question must follow open-domain question answering conventions — it should appear factual and general-purpose. The answer must be consistent with the content of the document, regardless of whether the document itself is factually accurate.

It is critical to follow the output format exactly as specified below. And do not include any additional text or explanations.
Output Format:
Q: <question>
A: <answer>

Content: {content}
"""

def load_documents(file_path):
    documents = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            documents.append({"id": data["id"], "content": data["content"], "score": data["score"]})
    return documents

model = OllamaLLM(model=MODEL_NAME)

def generate_QA_pairs(documents):
    results = []
    nr_of_bad_docs = 0
    try:
        # Initialize tqdm progress bar
        with tqdm(total=len(documents), unit="doc") as pbar:
            for i, doc in enumerate(documents, start=1):
                prompt = ChatPromptTemplate.from_template(template)
                chain = prompt | model
                response = chain.invoke({"content": doc["content"]})
                try:
                    # Extract the question and answer from the response
                    lines = response.split("\n")
                    question = lines[0].split(": ", 1)[1]  # Extract the part after "Q: "
                    answer = lines[1].split(": ", 1)[1]    # Extract the part after "A: "

                    results.append({
                        "id": doc["id"],
                        "document": doc["content"],
                        "score": doc["score"],
                        "question": question,
                        "answer": answer
                    })
                except IndexError or ValueError:
                    # Handle cases where the response format is unexpected
                    results.append({
                        "id": doc["id"],
                        "document": doc["content"],
                        "score": doc["score"],
                        "response": response,
                    })
                    nr_of_bad_docs += 1
                # Update progress bar
                pbar.set_description(f"Processed {i} docs, Bad docs: {nr_of_bad_docs}")
                pbar.update(1)
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user. Saving progress...")

    return results

# Load documents from chunks.jsonl
documents = load_documents(DOCUMENTS_FILE)

# Run the ranking
document_with_QA_pair = generate_QA_pairs(documents)


with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for res in document_with_QA_pair:
        f.write(json.dumps(res) + "\n")

print(f"Results have been written to {OUTPUT_FILE}")