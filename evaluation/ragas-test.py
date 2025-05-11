import argparse
import json
import re
from ragas import SingleTurnSample, EvaluationDataset
from ragas.metrics import (
    faithfulness,
    AnswerAccuracy,
    FactualCorrectness
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
import asyncio
import os
import sys

# from dotenv import load_dotenv

# load_dotenv()
#
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# if not OPENAI_API_KEY:
#     print("Please set the OPENAI_API_KEY environment variable.")
#     sys.exit(1)
#
# OPENAI_ORG_ID = os.getenv("OPENAI_ORG_ID")
# if not OPENAI_ORG_ID:
#     print("Please set the OPENAI_ORG_ID environment variable.")
#     sys.exit(1)
#
# OPENAI_PROJECT_ID = os.getenv("OPENAI_PROJECT_ID")
# if not OPENAI_PROJECT_ID:
#     print("Please set the OPENAI_PROJECT_ID environment variable.")
#     sys.exit(1)


def split_documents(contexts):
    """
    Splits the input string into separate documents based on the pattern "Document x:".
    Each document is returned as a separate string in a list.
    """

    # Use regex to split while keeping the delimiter (Document x:)
    parts = re.split(r'(Document \d+:)', contexts)
    
    # Reconstruct documents using the marker and its following content
    documents = []
    for i in range(1, len(parts), 2):  # skip non-matching start if any
        doc_header = parts[i].strip()
        doc_body = parts[i+1].strip() if i+1 < len(parts) else ''
        documents.append(f"{doc_header} {doc_body}\n")
    
    return documents

def load_qa_results(qa_results_paths, base_path):
    """
    Load the QA pairs from the specified JSONL files.
    Each file is expected to contain a list of dictionaries with keys:
    - "question": The question asked.
    - "answer": The answer provided by the model.
    - "expected_answer": The expected answer for comparison.
    """


    samples_collection = {}
    for path in qa_results_paths:
        with open(f"{base_path}/{path}", "r", encoding="utf-8") as f:
            samples = {}
            count = 0
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
                count += 1
            samples_collection[path] = samples
    
    return samples_collection


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
async def eval_answer_accuracy(dataset, llm) -> tuple:
    answerAccuracy = AnswerAccuracy(llm=llm) 
    # factualCorrectness = FactualCorrectness(llm=llm)
    scores = []
    errors = []
    with tqdm(total=len(dataset.keys()), unit="question") as pbar:
        for sampleKey in dataset.keys():
            sample = dataset[sampleKey]
            answerAccuracyScore = await answerAccuracy.single_turn_ascore(sample)
            # factualCorrectnessScore = await factualCorrectness.single_turn_ascore(sample)
            try:
                scores.append({
                    "id": sampleKey,
                    "user_input": sample.user_input,
                    "response": sample.response,
                    "reference": sample.reference,
                    "answer_accuracy_score": answerAccuracyScore,
                    # "factual_correctness_score": factualCorrectnessScore
                })
            except ValueError:
                errors.append(sampleKey)

            pbar.update(1)
    print(f"Processed {len(dataset.keys())} samples. Errors: {len(errors)}")
    
    return (scores, errors)

def extract_file_name(path: str) -> str:
    filename = path.split("/")[-1]
    return filename.removesuffix(".jsonl")



def main():
    model_name = args.model_name
    llm = LangchainLLMWrapper(OllamaLLM(model=model_name))

    base_path = "results/all-k-done"
    file_names = [
        "gpt-3.5-turbo-instruct_k2_t2.0.jsonl",
        "gpt-3.5-turbo-instruct_k2_t3.0.jsonl",
        "gpt-3.5-turbo-instruct_k2_t4.0.jsonl",
        "gpt-3.5-turbo-instruct_k2_t5.0.jsonl",
        "gpt-3.5-turbo-instruct_k2_t6.0.jsonl",
        "gpt-3.5-turbo-instruct_k3_t2.0.jsonl",
        "gpt-3.5-turbo-instruct_k3_t3.0.jsonl",
        "gpt-3.5-turbo-instruct_k3_t4.0.jsonl",
        "gpt-3.5-turbo-instruct_k3_t5.0.jsonl",
        "gpt-3.5-turbo-instruct_k3_t6.0.jsonl",
        "gpt-3.5-turbo-instruct_k4_t2.0.jsonl",
        "gpt-3.5-turbo-instruct_k4_t3.0.jsonl",
        "gpt-3.5-turbo-instruct_k4_t4.0.jsonl",
        "gpt-3.5-turbo-instruct_k4_t5.0.jsonl",
        "gpt-3.5-turbo-instruct_k4_t6.0.jsonl",
        "gpt-3.5-turbo-instruct_k5_t2.0.jsonl",
        "gpt-3.5-turbo-instruct_k5_t3.0.jsonl",
        "gpt-3.5-turbo-instruct_k5_t4.0.jsonl",
        "gpt-3.5-turbo-instruct_k5_t5.0.jsonl",
    ]
    samples_collection = load_qa_results(file_names, base_path)

    for sample_name in samples_collection.keys():
        samples = samples_collection[sample_name]
        result, errors = asyncio.run(eval_answer_accuracy(samples, llm))

        # Ensure the directory exists
        output_dir = f"results/evaluations/ragas-runpod-{model_name}_all-k_amd"
        os.makedirs(output_dir, exist_ok=True)

        # Save Results to csv
        df_results = pd.DataFrame(result)
        df_results.to_csv(f"{output_dir}/{sample_name}_AA.csv", index=False)
        print(f"Results for {sample_name} saved to CSV.")

        # Save Errors to csv
        df_errors = pd.DataFrame(errors)
        df_errors.to_csv(f"{output_dir}/{sample_name}_AA_errors.csv", index=False)
        print(f"Errors for {sample_name} saved to CSV.")

if __name__ == "__main__":
    # Get LLM model from command line argument

    parser = argparse.ArgumentParser(description="Evaluate RAGAS on a given model.")
    parser.add_argument("-m", "--model_name", type=str, required=True, help="LLM model name (e.g., mistralai/mixtral-8x22b-instruct-v0.1)")
    args = parser.parse_args()

    main()
    # llm = LangchainLLMWrapper(ChatOpenAI(
    #     model="gpt-3.5-turbo-instruct",
    #     organization=OPENAI_ORG_ID,
    #     api_key=OPENAI_API_KEY,
    # ))
    #embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="nomic-embed-text:latest"))

    # qa_results_paths = [
    #     "results/qwen2.5:14b-instruct-q8_0_k3.jsonl",
    #     "results/gpt-3.5-turbo-0125_k3.jsonl",
    #     "results/gemma3:27b-it-q8_0_k3.jsonl",
    #     "results/gemma3:12b-it-q8_0_k3.jsonl",
    # ]
