import argparse
from datetime import datetime

import sys
import os
from openai import OpenAI
from dotenv import load_dotenv

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

def main(batch):


    client = OpenAI(
        api_key=OPENAI_API_KEY,
        organization=OPENAI_ORG_ID,
        project=OPENAI_PROJECT_ID
    )

    uploaded_file = client.files.create(
        file=open(f"../data/{batch}", "rb"),
        purpose="batch"
    )
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    created_batch = client.batches.create(
        input_file_id=uploaded_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "batch_name": batch,
            "creation_time": current_time
        }
    )

    if created_batch.errors:
        print(f"Error creating batch {created_batch.id}, errors: {created_batch.errors}")
        return

    print(f"Batch created with ID: {created_batch.id}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create a batch using the OpenAI API.")
    parser.add_argument("-b", "--batch_input_file", type=str, help="The input file for the batch.")

    args = parser.parse_args()

    main(
        batch=args.batch_input_file
    )