import json
import random
import argparse

def extract_random_subset(input_file, output_file, num_docs):
  with open(input_file, 'r') as infile:
    lines = infile.readlines()
  
  if num_docs > len(lines):
    raise ValueError("Number of documents to extract exceeds the total number of documents in the file.")
  
  random_subset = random.sample(lines, num_docs)
  
  with open(output_file, 'w') as outfile:
    outfile.writelines(random_subset)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Extract a random subset of documents from a .jsonl file.")
  parser.add_argument("input_file", help="Path to the input .jsonl file")
  parser.add_argument("output_file", help="Path to the output .jsonl file")
  parser.add_argument("num_docs", type=int, help="Number of documents to extract")
  
  args = parser.parse_args()
  
  try:
    extract_random_subset(args.input_file, args.output_file, args.num_docs)
    print(f"Successfully extracted {args.num_docs} documents to {args.output_file}")
  except Exception as e:
    print(f"Error: {e}")