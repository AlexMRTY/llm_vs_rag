import json

def load_openai_batch(path):

    batch = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            batch[data['custom_id']] = data["body"]["messages"][1]["content"]
    return batch

def load_qa_pairs():
    with open("../data/QA-pair-1000-huggingface.jsonl", "r", encoding="utf-8") as f:
        qa_pairs = {}
        for line in f:
            data = json.loads(line.strip())
            qa_pairs[data["question"]] = {
                "id": data["id"],
                "expected_answer": data["answer"]
            }
    return qa_pairs

def openai_response_to_eval(file_names: list[list[str]]):
    """
    Load the OpenAI response from the specified JSONL file.
    """
    result_collection = {}
    for name in file_names:
        qa_pairs = load_qa_pairs()
        batch = load_openai_batch(name[0])
        results = []
        with open(name[1], "r", encoding="utf-8") as f:
            count = 0
            for line in f:
                # if count > 3: break;
                data = json.loads(line.strip())
                results.append({
                    "id": qa_pairs[batch[data['custom_id']]]["id"],
                    "question": batch[data['custom_id']],
                    "answer": data["response"]["body"]["choices"][0]["message"]["content"],
                    "expected_answer": qa_pairs[batch[data['custom_id']]]["expected_answer"],
                })
                count += 1
        result_collection[name[1]] = results
    return result_collection


def main():
    qa_openai_paths = [
        ["../data/batch_gpt-4.1-2025-04-14.jsonl", "results/batch_gpt-4.1-2025-04-14_results.jsonl"],
        ["../data/batch_gpt-4o-2024-11-20.jsonl", "results/batch_gpt-4o-2024-11-20_results.jsonl"]
    ]

    data = openai_response_to_eval(qa_openai_paths)
    for file_name, results in data.items():
        with open(f"../data/{file_name.replace("results/", "")}", "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()