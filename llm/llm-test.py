import json
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from tqdm import tqdm  # For progress reporting

# --- Config ---
QA_PAIRS_PATH = "data/QA-pair-1000-huggingface.jsonl"
LLM_MODEL_NAME = "llama3.1:8b-instruct-fp16"


def load_QA_pairs(file_path):
    """
    Load QA pairs from a JSONL file.
    """
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


def run_llm_only(questions, llm, chain):
    """
    Run the LLM to answer questions without retrieval.
    """
    with tqdm(total=len(questions), unit="doc") as pbar:
        with open(f"data/{llm}_llm_only.jsonl", "w", encoding="utf-8") as f:
            for i, qa in enumerate(questions, start=1):
                # Directly invoke the LLM with the question
                response = chain.invoke({"question": qa["question"]})
                try:
                    # Extract the answer from the response
                    lines = response.split("\n")
                    answer = lines[0].split(": ", 1)[1]  # Extract the part after "Answer: "
                except (IndexError, ValueError):
                    # Handle cases where the response format is unexpected
                    answer = response

                f.write(json.dumps({
                    "id": qa["id"],
                    "content": qa["content"],
                    "question": qa["question"],
                    "expected_answer": qa["answer"],
                    "answer": answer,
                }) + "\n")
                # Update progress bar
                pbar.set_description(f"Processed {i} docs")
                pbar.update(1)


if __name__ == "__main__":
    # --- Load Model ---
    print("🔄 Loading LLM...")
    model = OllamaLLM(model=LLM_MODEL_NAME)

    # --- Define Template ---
    template = """
    Answer the following question concisely and accurately. If you don't know the answer, respond with 'I don't know.'

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model

    # --- Load QA Pairs ---
    qa_pairs = load_QA_pairs(QA_PAIRS_PATH)

    # --- Run the test ---
    run_llm_only(qa_pairs, model, chain)