from datasets import load_dataset, Dataset


def remove_docs_under_150_words(dataset) -> Dataset:
    return dataset.filter(lambda x: len(x["content"].split()) > 150)

def main():
    dataset = load_dataset("data/refined-web-8m.jsonl", split="train")
    print(f"Dataset Length (stage 0): {dataset.__len__()}")

    stage_1 = remove_docs_under_150_words(dataset)
    print(f"Dataset Length (stage 1): {stage_1.__len__()}")



if __name__ == "__main__":
    main()