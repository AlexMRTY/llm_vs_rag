import pprint
from openai import OpenAI
import sys
from dotenv import load_dotenv

import os

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

client = OpenAI(
    api_key=OPENAI_API_KEY,
    organization=OPENAI_ORG_ID,
    project=OPENAI_PROJECT_ID
)

for batch in client.batches.list(limit=30):
    # if batch.status == "failed":
        # print(batch.input_file_id)
        # print (batch.id)
    if not batch.status == "failed":
        print(f"-------------{batch.id}----------")
        print(f"Status: {batch.status}")
        print(f"Created: {batch.created_at}")
        print(f"Endpoint: {batch.endpoint}")
        print(f"Input file ID: {batch.input_file_id}")
        print(f"Output file ID: {batch.output_file_id}")
        print(f"Error file ID: {batch.error_file_id}")
        print(f"Metadata: {batch.metadata}")
        print(f"Errors: {batch.errors}")
        print("\n")

print("---------failed----------------")
for batch in client.batches.list(limit=30):

    if batch.id == "batch_681fa70231348190bbece69336588d38":
        print(batch.id)
        print(batch.status)
        print(batch.created_at)
        print(batch.errors)
        print(batch.error_file_id)
    # elif not batch.status == "completed" or not batch.status == "in_progress":
    #     print(batch.id)