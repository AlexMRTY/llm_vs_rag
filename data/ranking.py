from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import json
from datasets import load_dataset
from tqdm import tqdm  # For progress reporting

# Define the instruction prompt
# It is creitical to follow the output format exactly as specified below. And do not include any additional text or explanations.
template = """
You will evaluate and assign a score to each document on a scale of 1 to 5. 
The goal is to assess how suitable the document is for use in an open-domain question-answering (QA) system, 
based on its factual content and appropriateness for such tasks.

Scoring Criteria:
- 1 - Very Poor: The document contains little to no factual information, is highly subjective, opinion-based, or irrelevant to QA tasks.
- 2 - Poor: Contains minimal factual content, or the information is vague, outdated, or difficult to verify. Not useful for QA.
- 3 - Moderate: Contains some factual information, but may include noise, inconsistencies, or limited depth. Partially useful for QA.
- 4 - Good: Mostly factual, relevant, and clear. Provides useful content that can contribute to answering a range of questions.
- 5 - Excellent: Highly factual, well-structured, and rich in verifiable information. Ideal for QA use.

Additional Instructions:
- Focus on the quality, clarity, and density of factual information.
- Ignore formatting issues or grammatical mistakes unless they affect comprehension.
- Do not summarize the document; only assign a score.
- Assume the QA system is open-domain and not restricted to any specific topic.

Output Format:
It is critical to follow the output format exactly as specified below. And do not include any additional text or explanations.
S: <1–5>


Content: {content}
"""

def load_documents(file_path):
    documents = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            documents.append({"id": data["id"], "content": data["content"]})
    return documents

# Choose your ollama model (e.g., 'llama2', 'mistral', etc.)
MODEL_NAME = "llama3.1:8b-instruct-fp16"
model = OllamaLLM(model=MODEL_NAME)

# Function to process and rank each document
def rank_documents(documents):
    results = []
    try:
        with tqdm(total=len(documents), desc="Ranking Documents", unit="doc") as pbar:
            for i, doc in enumerate(documents, start=1):
                prompt = ChatPromptTemplate.from_template(template)
                chain = prompt | model
                response = chain.invoke({"content": doc["content"]})

                # Extract the score from the response as int
                try:
                    score = int(response[-1])
                except ValueError:
                    score = response  # Assign the whole response if int conversion fails
                
                if (not isinstance(score, int)):
                    results.append({"id": doc["id"], "content": doc["content"], "score": score})
                elif (score >= 4):
                    results.append({"id": doc["id"], "content": doc["content"], "score": score})
                

                # Update progress bar
                pbar.update(1)
    except KeyboardInterrupt:
        print("\nInterrupted! Returning results collected so far...")
    return results

# Load documents from chunks.jsonl
DOCUMENTS_FILE = "data/refined-web-50k-random.jsonl"
documents = load_documents(DOCUMENTS_FILE)
# documents = load_dataset("AlexMRTY/refined-web-50k-random", split="train")

# Run the ranking
ranked_docs = rank_documents(documents)

OUTPUT_FILE = "data/refined-web-50k-post-ranking.jsonl"

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for res in ranked_docs:
        # Parse the result into a JSON object
        # document_id, score = res.split("\n")[0].split(": ")[1], res.split("\n")[1].split(": ")[1]
        # json_line = {
        #     # "id": document_id,
        #     # "score": int(score),
        #     "full_response": res  # Save the entire response
        # }
        f.write(json.dumps(res) + "\n")

print(f"Results have been written to {OUTPUT_FILE}")