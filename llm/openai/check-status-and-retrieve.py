import sys

from dotenv import load_dotenv
from openai import OpenAI
import os
import argparse

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("Please set the OPENAI_API_KEY environment variable.")
    sys.exit(1)

OPENAI_ORG_ID = os.getenv("OPENAI_ORG_ID")
if not OPENAI_ORG_ID:
    print("Please set the OPENAI_ORG_ID environment variable.")
    sys.exit(1)

OPENAI_PROJECT_ID = os.getenv("OPENAI_PROJECT_ID")
if not OPENAI_PROJECT_ID:
    print("Please set the OPENAI_PROJECT_ID environment variable.")
    sys.exit(1)

def main(batch_id, retrieve):
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        organization=OPENAI_ORG_ID,
        project=OPENAI_PROJECT_ID
    )

    batch = client.batches.retrieve(batch_id)
    print(batch.status)
    if batch.status != "completed": return

    if retrieve:
        # check if directory exists
        if not os.path.exists("results"):
            os.makedirs("results")
        if not os.path.exists("errors"):
            os.makedirs("errors")

        output_file = client.files.content(batch.output_file_id)
        output_file.write_to_file(f"results/{batch.metadata['batch_name'].replace(".jsonl", "")}_results.jsonl")
        # if batch.errors:
        print("Errors found in batch. Saving error file.")
        error_file = client.files.content(batch.error_file_id)
        error_file.write_to_file(f"errors/{batch.metadata['batch_name'].replace('.jsonl', '')}_errors.jsonl")
        print("Results retrieved successfully.")
    else:
        print("Batch retrieval cancelled.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieve batch results from OpenAI API.")
    parser.add_argument("-b", "--batch_id", type=str, help="The ID of the batch to retrieve.", required=True)
    parser.add_argument("-r", "--retrieve", type=bool, help="Whether to retrieve the results.", default=False, choices=[True, False])
    args = parser.parse_args()

    # print(args.batch_id)
    # print(args.retrieve)
    main(args.batch_id, args.retrieve)
