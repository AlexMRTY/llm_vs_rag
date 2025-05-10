import pandas as pd
from pandas import DataFrame
import pprint

def load_results(file_names: list[str], base_path: str) -> dict[str, DataFrame]:
    df_collection = {}
    for name in file_names:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(f"{base_path}/{name}")
        # Extract the model name from the filename
        model_name = name.replace("_FC.csv", "").replace("batch_", "")
        # Store the DataFrame in the dictionary with the model name as the key
        df_collection[model_name] = df

    return df_collection

def answer_accuracy_averages(df_collection: dict[str, DataFrame]) -> dict[str, float]:
    """
    Calculate the average accuracy scores for each model in the collection of DataFrames.
    :param df_collection:
    :return: {model_name: average_accuracy}
    """
    accuracy_averages = {}
    for model_name, df in df_collection.items():
        # Calculate the average accuracy for the current model
        avg_accuracy = df["answer_accuracy_score"].mean()
        # Store the average accuracy in the dictionary
        accuracy_averages[model_name] = avg_accuracy

    return accuracy_averages

def main():
    base_path = "results/evaluations/ragas-runpod-mixtral_8x7b-instruct-v0.1-q8_0_amd"
    file_names = [
        "batch_gpt-4.1-2025-04-14_results_FC.csv",
        "batch_gpt-4o-2024-11-20_results_FC.csv",
        "gemini-2.5-pro-preview-05-06_FC.csv",
        "gemma3_12b-it-q8_0_k3_FC.csv",
        "gemma3_27b-it-q8_0_k3_FC.csv",
        "gpt-3.5-turbo-0125_k3_FC.csv",
        "qwen2.5_14b-instruct-q8_0_k3_FC.csv"
    ]

    data_collection = load_results(file_names, base_path)
    answer_accuracy_score_averages = answer_accuracy_averages(data_collection)



if __name__ == "__main__":
    main()