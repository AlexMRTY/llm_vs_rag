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
            if (data.get("response")):
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
async def eval_answer_accuracy(dataset):
    scorer = AnswerAccuracy(llm=llm) # evaluator_llm wrapped with ragas LLM Wrapper
    scores = []
    with tqdm(total=len(dataset.keys()), unit="sample") as pbar:
        for sampleKey in dataset.keys():
            sample = dataset[sampleKey]
            score = await scorer.single_turn_ascore(sample)
            scores.append({
                "id": sampleKey,
                "user_input": sample.user_input,
                "response": sample.response,
                "reference": sample.reference,
                "score": score
            })
            
            pbar.update(1)
    
    return scores

result_collection = {}
for sample_name in samples_collection.keys():
    samples = samples_collection[sample_name]
    result_collection[sample_name] = asyncio.run(eval_answer_accuracy(samples))


# Save Results to csv
for sample_name in result_collection.keys():
    df = pd.DataFrame(result_collection[sample_name])
    df.to_csv(f"results/evaluations/{sample_name}_answer_accuracy.csv", index=False)
    print(f"Results for {sample_name} saved to CSV.")

