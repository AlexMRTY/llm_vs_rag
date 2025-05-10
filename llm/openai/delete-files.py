import os
import sys
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

def delete_files(ids):
    from openai import OpenAI
    client = OpenAI()

    for file_id in file_ids:
        response = client.files.delete(file_id)
        print(f"File {file_id} deleted: {response.deleted}")

def cancel_batches(ids):
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        organization=OPENAI_ORG_ID,
        project=OPENAI_PROJECT_ID
    )

    for batch_id in ids:
        response = client.batches.cancel(batch_id)
        print(f"Batch {batch_id} cancel: {response.deleted}")

if __name__ == "__main__":
    file_ids = [
        "file-YQXNw9GCahYYmbuD6FXx7C",
        "file-9wDgVhqVtuyD7eUKrZoG9G",
        "file-LGqSHrLvHLLZSQUkufMji1",
        "file-SKW65SiS8m5KyZsrs3EDnt",
        "file-3v3qTDCtU5D3w7u3gkhnRD",
        "file-1Zvsud5qBh1mRkXuDoca5A",
        "file-XkEWgVB95YSCZYe5rm15B4",
        "file-Na4CCS1H97m9MQugxZzbiz",
        "file-NLNY9JVzpRaSZfyJ3qNtzr",
        "file-K5iCmrDYnNndUzWTxcr69B",
        "file-9Ff9D41yqUypYXXtuWXtga",
        "file-JsY1ydVnyRkrmkRj5w7zdQ",
        "file-XHVBaAUvoJJvTuW4WkktiE",
        "file-2qavicTD5qRz2rppt5tjzF",
        "file-5bovGWCoBRuqCL9CgMqUD2",
        "file-DGJGukdmdK3BxJhS8wYwzS",
        "file-BHYZKadJjxqXjz9sodtNVq",
        "file-2hgMEybCTubvUMnM8gPghM",
        "file-TWx1EuyWQTRsXP96HyEoSZ",
        "file-HeeegwMYvht5QS5b7sXAYv",
        "file-WCLASdC5nYFBue6LfXXoUw",
        "file-8w8vias4Afbhnp7EQXtKog",
        "file-BmbU7rC9U2E5h8T5MucnJx",
        "file-EHGskSutuc8JwijiNkmyUC",
        "file-GMSKEkTfzbpgJ6AjK4mXos"
    ]

    batch_ids = [
        # "batch_681f62380e8481908e3adb32bf9501c4",
        "batch_681f5d4cae648190b5665d0e91282d0d",
        "batch_681f5d4295908190968fdae942ec5923",
        "batch_681f5d3fd79c8190aa82ed05335da6a8",
        "batch_681f5d3db1a08190b54492117d7931b8",
        "batch_681f5d3b3f748190a913a485e5985b50",
        "batch_681f5d391eb481909365ec961f4ea962",
        "batch_681f5d36deb08190b9f9dd7cc9c2f887",
        "batch_681f5d34a84481908aae2cd7dc324aed",
        "batch_681f5d32b8a88190ae94f495f873f178",
        "batch_681f5d29a904819083622f8284cb5918",
        "batch_681f5d27c1188190942bc1b5246bb9ac",
        "batch_681f5d2053b48190a8cd77f592b8597b",
        "batch_681f5d17b5508190b2b4a6e0056d68f8",
        "batch_681f5d0e64ec8190ab9d041cb639c23f",
        "batch_681f5d0c4f648190be08a85d9c8edac7",
        "batch_681f5d0a3e7481908096bcf0d1943240",
        "batch_681f5d0876f08190bb88ef193ecec4d5",
        "batch_681f5d0705148190927827cddbfd8a00",
        "batch_681f5d0435708190a90ec045490b089f",
        "batch_681f5d021674819082cdf46e8a9efe30",
        "batch_681f5cf9cf708190b544354a9779e603",
        "batch_681f5cefc4cc8190bc2aa5e954b8679c",
        "batch_681f5cee72648190b5316a3153b2dec2",
        "batch_681f5cec7c9c819084ebb47dd3c7f339",
        "batch_681f5cea3bc08190babca8f36ee672fe",
        "batch_6819af712d448190ba0ffaf99e20d9d3",
        "batch_6819420362c8819082dc738a27a4daca"
    ]
    cancel_batches(batch_ids)