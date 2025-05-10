import json
from tqdm import tqdm

             # Total number of JSON objects expected

def fix_jsonl(file_name, total_objects):
    with open(f"results/batches/{file_name}", 'r', encoding='utf-8') as f:
        data = f.read()

    decoder = json.JSONDecoder()
    pos = 0
    fixed_lines = []

    with tqdm(total=total_objects, desc="Fixing JSONL") as pbar:
        while pos < len(data) and len(fixed_lines) < total_objects:
            try:
                obj, idx = decoder.raw_decode(data[pos:])
                fixed_lines.append(json.dumps(obj))
                pos += idx
                pbar.update(1)
            except json.JSONDecodeError as e:
                print(f"\nError decoding JSON at position {pos}: {e}")
                break

    with open(f"results/batches-fixed/{file_name}", 'w', encoding='utf-8') as f:
        for line in fixed_lines:
            f.write(line + '\n')

    print(f"✅ Fixed JSONL file written to: 'results/batches-fixed/{file_name}'")


file_names = [
    "batch_gpt-3.5-turbo-instruct_k2_t0.2.jsonl",
    "batch_gpt-3.5-turbo-instruct_k2_t0.3.jsonl",
    "batch_gpt-3.5-turbo-instruct_k2_t0.4.jsonl",
    "batch_gpt-3.5-turbo-instruct_k2_t0.5.jsonl",
    "batch_gpt-3.5-turbo-instruct_k2_t0.6.jsonl",
    "batch_gpt-3.5-turbo-instruct_k3_t0.2.jsonl",
    "batch_gpt-3.5-turbo-instruct_k3_t0.3.jsonl",
    "batch_gpt-3.5-turbo-instruct_k3_t0.4.jsonl",
    "batch_gpt-3.5-turbo-instruct_k3_t0.5.jsonl",
    "batch_gpt-3.5-turbo-instruct_k3_t0.6.jsonl",
    "batch_gpt-3.5-turbo-instruct_k4_t0.2.jsonl",
    "batch_gpt-3.5-turbo-instruct_k4_t0.3.jsonl",
    "batch_gpt-3.5-turbo-instruct_k4_t0.4.jsonl",
    "batch_gpt-3.5-turbo-instruct_k4_t0.5.jsonl",
    "batch_gpt-3.5-turbo-instruct_k4_t0.6.jsonl",
    "batch_gpt-3.5-turbo-instruct_k5_t0.2.jsonl",
    "batch_gpt-3.5-turbo-instruct_k5_t0.3.jsonl",
    "batch_gpt-3.5-turbo-instruct_k5_t0.4.jsonl",
    "batch_gpt-3.5-turbo-instruct_k5_t0.5.jsonl",
    "batch_gpt-3.5-turbo-instruct_k5_t0.6.jsonl",
    "batch_gpt-3.5-turbo-instruct_k6_t0.2.jsonl",
    "batch_gpt-3.5-turbo-instruct_k6_t0.3.jsonl",
    "batch_gpt-3.5-turbo-instruct_k6_t0.4.jsonl",
    "batch_gpt-3.5-turbo-instruct_k6_t0.5.jsonl",
    "batch_gpt-3.5-turbo-instruct_k6_t0.6.jsonl",
]
num_objects = 999
for f in file_names:
    fix_jsonl(f, num_objects)
