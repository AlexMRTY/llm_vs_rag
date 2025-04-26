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

QA_PAIR_PATH = "data/llama3.1:8b-instruct-fp16_k1.jsonl"
samples = {}
with open(QA_PAIR_PATH, "r", encoding="utf-8") as f:
    count = 0
    for line in f:
        if count > 3: break;
        data = json.loads(line.strip())
        if (data.get("response")):
            continue
        samples[data['id']] = SingleTurnSample(
            user_input=data["question"],
            response=data["answer"],
            reference=data["expected_answer"]
        )
        count += 1





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

# dataset = EvaluationDataset(samples)

import asyncio
async def eval_answer_accuracy():
    scorer = AnswerAccuracy(llm=llm) # evaluator_llm wrapped with ragas LLM Wrapper
    scores = []
    with tqdm(total=len(samples.keys()), unit="sample") as pbar:
        for sampleKey in samples.keys():
            sample = samples[sampleKey]
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

result_answer_accuracy = asyncio.run(eval_answer_accuracy())
# print("Score: ", score)
# result = evaluate(dataset, metrics=[AnswerAccuracy], llm=llm,embeddings=embeddings)

# Save Results to csv
df = pd.DataFrame(result_answer_accuracy)
df.to_csv("data/ragas-result-test.csv", index=False)