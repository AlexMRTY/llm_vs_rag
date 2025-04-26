import json




# Load Documents
import re
from ragas import SingleTurnSample, EvaluationDataset

def split_documents(contexts):
    # Use regex to split while keeping the delimiter (Document x:)
    parts = re.split(r'(Document \d+:)', contexts)
    
    # Reconstruct documents using the marker and its following content
    documents = []
    for i in range(1, len(parts), 2):  # skip non-matching start if any
        doc_header = parts[i].strip()
        doc_body = parts[i+1].strip() if i+1 < len(parts) else ''
        documents.append(f"{doc_header} {doc_body}\n")
    
    return documents

# QA_PAIR_PATH = "results/llama3.1:8b-instruct-fp16_k1.jsonl"
qa_results = [
    "results/llama3.1:8b-instruct-fp16_k1.jsonl",
    "results/llama3.1:8b-instruct-fp16_k2.jsonl",
    "results/llama3.1:8b-instruct-fp16_k3.jsonl",
    "results/llama3.1:8b-instruct-fp16_k4.jsonl",
    "results/llama3.1:8b-instruct-fp16_k5.jsonl",
]
samples_collection = {}
for path in qa_results:
    with open(path, "r", encoding="utf-8") as f:
        samples = {}
        # count = 0
        for line in f:
            # if count > 3: break;
            data = json.loads(line.strip())

            # Null control: Ensure required keys exist and are not None
            if not all(key in data and data[key] is not None for key in ["question", "answer", "expected_answer"]):
                continue

            samples[data['id']] = SingleTurnSample(
                user_input=data["question"],
                response=data["answer"],
                reference=data["expected_answer"]
            )
            # count += 1
        samples_collection[path] = samples
    





from ragas.metrics import (
    faithfulness,
    AnswerAccuracy,
)

# from langchain_community.chat_models import ChatOllama
from langchain_ollama import (
    OllamaLLM, 
    OllamaEmbeddings
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas import evaluate
import pandas as pd
from tqdm import tqdm

llm = LangchainLLMWrapper(OllamaLLM(model="qwen2.5:14b-instruct"))
embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="nomic-embed-text:latest"))

"""
    Measures answer accuracy compared to ground truth given a user_input.
    This metric averages two distinct judge prompts to evaluate.

    Top10, Zero-shoot LLM-as-a-Judge Leaderboard:
    1)- mistralai/mixtral-8x22b-instruct-v0.1
    2)- mistralai/mixtral-8x7b-instruct-v0.1
    3)- meta/llama-3.1-70b-instruct
    4)- meta/llama-3.3-70b-instruct
    5)- meta/llama-3.1-405b-instruct
    6)- mistralai/mistral-nemo-12b-instruct
    7)- nvidia/llama-3.1-nemotron-70b-instruct
    8)- meta/llama-3.1-8b-instruct
    9)- google/gemma-2-2b-it
    10)- nvidia/nemotron-mini-4b-instruct
    The top1 LB model have high correlation with human judges (~0.90).

    Attributes
    ----------
    name: string
        The name of the metrics

    answer_accuracy:
        The AnswerAccuracy object
    """
import asyncio
async def eval_answer_accuracy(dataset) -> tuple:
    scorer = AnswerAccuracy(llm=llm) # evaluator_llm wrapped with ragas LLM Wrapper
    scores = []
    errors = []
    with tqdm(total=len(dataset.keys()), unit="question") as pbar:
        for sampleKey in dataset.keys():
            sample = dataset[sampleKey]
            score = await scorer.single_turn_ascore(sample)
            try:
                scores.append({
                    "id": sampleKey,
                    "user_input": sample.user_input,
                    "response": sample.response,
                    "reference": sample.reference,
                    "score": score
                })
            except ValueError:
                errors.append(sampleKey)

            pbar.update(1)
    print(f"Processed {len(dataset.keys())} samples. Errors: {len(errors)}")
    
    return (scores, errors)


import os

def extract_file_name(path: str) -> str:
    filename = path.split("/")[-1]
    return filename.removesuffix(".jsonl")

for sample_name in samples_collection.keys():
    samples = samples_collection[sample_name]
    result, errors = asyncio.run(eval_answer_accuracy(samples))

    # Ensure the directory exists
    output_dir = "results/evaluations"
    os.makedirs(output_dir, exist_ok=True)

    # Save Results to csv
    df_results = pd.DataFrame(result)
    df_results.to_csv(f"{output_dir}/{extract_file_name(sample_name)}_answer_accuracy.csv", index=False)
    print(f"Results for {sample_name} saved to CSV.")

    # Save Errors to csv
    df_errors = pd.DataFrame(errors)
    df_errors.to_csv(f"{output_dir}/{extract_file_name(sample_name)}_answer_accuracy_errors.csv", index=False)
    print(f"Errors for {sample_name} saved to CSV.")
