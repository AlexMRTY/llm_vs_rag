import pandas as pd
from pandas import DataFrame

def check_for_nan(df: DataFrame):
    """
    Check for NaN scores.
    :param df:
    :return: number of NaN scores
    """
    return df["answer_accuracy_score"].isna().sum()

def print_nan_scores(df_collections: dict):
    """
    Print the number of NaN scores in a collection of DataFrames.
    :param df_collections:
    """
    for file_name, df in df_collections.items():
        print(f"File: {file_name}, Number of NaN scores: {check_for_nan(df)}")


def nan_example(df: DataFrame):
    """
    Id of a row with NaN score.
    :param df:
    :return:
    """
    # Get the first row with NaN in 'answer_accuracy_score' column
    nan_row = df[df["answer_accuracy_score"].isna()].iloc[0]
    # Get the index of that row
    nan_index = nan_row.name
    return nan_index

def main():

    files = {
        "mixtral-7b" : {
            "base": "results/evaluations/ragas-runpod-mistral-8x7b-instruct-v0.1",
            "file_names": [
                "qwen2.5:14b-instruct-q8_0_k3_FC.csv",
                "gpt-3.5-turbo-0125_k3_FC.csv",
                "gemma3:27b-it-q8_0_k3_FC.csv"
            ]
        },
        "mixtral-22b": {
            "base": "results/evaluations/ragas-runpod-mistral-8x22b-instruct-v0.1",
            "file_names": [
                "llama3.1_8b-instruct-fp16_k1_answer_accuracy.csv",
                "llama3.1_8b-instruct-fp16_k2_answer_accuracy.csv",
                "llama3.1_8b-instruct-fp16_k3_answer_accuracy.csv",
                "llama3.1_8b-instruct-fp16_k4_answer_accuracy.csv",
                "llama3.1_8b-instruct-fp16_k5_answer_accuracy.csv"
            ]
        }
    }

    df_collection = {}
    for eval_model_name in files.keys():
        for file_name in files[eval_model_name]["file_names"]:
            df = pd.read_csv(f"{files[eval_model_name]['base']}/{file_name}")
            df_collection[file_name] = df

    print_nan_scores(df_collection)


if __name__ == "__main__":
    main()