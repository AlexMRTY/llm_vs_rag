import argparse
import json
import sys


def load_QA_pairs():
    """
    Load question-answer pairs from a JSONL file.

    :return: List of dictionaries containing QA pairs with id, content, question and answer fields
    """
    print("🔄 Loading QA pairs...")
    pairs = []
    with open("../data/QA-pair-1000-huggingface.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            pairs.append({
                "id": data["id"],
                "content": data["document"],
                "question": data["question"],
                "answer": data["answer"],
                })
    return pairs

def generate_batch(qa_pairs, output_name, model_name, endpoint):
    """
    Structure of the generated output JSONL file:
    {
        "custom_id": "request-1",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-3.5-turbo-0125",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": "Hello world!"
                }
            ],
            "max_tokens": 1000
        }
    }
    :param qa_pairs:
    :param output_name:
    :param model_name:
    :param endpoint:
    :return:
    """
    print("🔄 Generating batch...")
    with open(f"../data/{output_name}", "w", encoding="utf-8") as f:
        for index, pair in enumerate(qa_pairs):
            f.write(json.dumps({
                "custom_id": f"request-{index}",
                "method": "POST",
                "url": endpoint,
                "body": {
                    "model": model_name,
                    "messages": [
                        {
                            "role": "system",
                            "content": """You are an expert in answering Open domain questions. Use the knowledge you have to answer the question. \n
                                       Your response should only include the answer to the question (max 30 words) without any extra explanation or reasoning. \n"""
                        },
                        {
                            "role": "user",
                            "content": pair["question"]
                        }
                    ],
                    "max_tokens": 500
                }
            }, ensure_ascii=False) + "\n")


def main(
        model_name
):
    qa_pairs = load_QA_pairs()

    generate_batch(
        qa_pairs=qa_pairs,
        output_name=f"batch_{model_name}.jsonl",
        model_name=model_name,
        endpoint="/v1/chat/completions"
    )

    print("✅ Batch generation completed!")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create a batch using the OpenAI API.")
    parser.add_argument("-m", "--model_name", type=str, help="Name of the model to use.")

    args = parser.parse_args()

    main(
        model_name=args.model_name
    )