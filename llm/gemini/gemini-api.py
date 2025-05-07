import os
import json
import argparse
from tqdm import tqdm
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

API_KEY = os.getenv("API_KEY")

def load_QA_pairs():
    """
    Load question-answer pairs from a JSONL file.

    :return: List of dictionaries containing QA pairs with id, content, question and answer fields
    """
    print("🔄 Loading QA pairs...")
    pairs = []
    with open("../data/QA-pair-1000-huggingface.jsonl", "r", encoding="utf-8") as f:
        counter = 0
        for line in f:
            if counter >= 5: break
            counter += 1

            data = json.loads(line.strip())
            pairs.append({
                "id": data["id"],
                "content": data["document"],
                "question": data["question"],
                "answer": data["answer"],
                })
    return pairs


def run_test(client, model, qa_pairs) -> list[dict]:
    response_list = []
    instructions = """You are an expert in answering Open domain questions. Use the knowledge you have to answer the question. \nYour response should only include the answer to the question (max 30 words) without any extra explanation or reasoning. \n"""

    with tqdm(total=len(qa_pairs)) as pbar:
        for pair in qa_pairs:
            response = client.models.generate_content(
                model=model,
                config=types.GenerateContentConfig(
                    system_instruction=instructions
                ),
                contents=pair["question"],
            )
            response_list.append({
                "id": pair["id"],
                "content": pair["print(response.text)content"],
                "question": pair["question"],
                "answer": response.text,
                "expected_answer": pair["answer"],
            })

            pbar.update(1)

    return response_list


def main(model):
    client = genai.Client(api_key=API_KEY)

    responses = run_test(client, model, load_QA_pairs())
    print("Done!")

    with open(f"results/{model}.jsonl", "w", encoding="utf-8") as f:
        for response in responses:
            f.write(json.dumps(response) + "\n")
    print(f"Saved responses to results/{model}.jsonl")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run test on gemini models")
    parser.add_argument("-m", "--model", required=True, help="The model to use")
    args = parser.parse_args()

    main(args.model)
