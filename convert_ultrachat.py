"""Convert UltraChat 200k parquet files to Vidya training format.

Output: one conversation per line, with <|user|> and <|assistant|> markers.
Each line looks like:
  <|user|> question <|assistant|> answer <|user|> followup <|assistant|> reply

Reads from the existing download at ~/github/vidya/data/ultrachat/data/
"""

import pyarrow.parquet as pq
import glob
import random

input_dir = "/home/roland/github/vidya/data/ultrachat/data"
output_file = "/home/roland/github/vidya-v3/chat_input_ultrachat.txt"

files = sorted(glob.glob(f"{input_dir}/train_sft*.parquet"))

docs = []
for f in files:
    table = pq.read_table(f)
    rows = table.to_pydict()
    for messages in rows["messages"]:
        # Build one line per conversation
        parts = []
        for msg in messages:
            role = msg["role"]
            text = msg["content"].replace("\n", " ").strip()
            if not text:
                continue
            if role == "user":
                parts.append(f"<|user|> {text}")
            elif role == "assistant":
                parts.append(f"<|assistant|> {text}")
        if len(parts) >= 2:
            docs.append(" ".join(parts))

# Shuffle for training
random.seed(42)
random.shuffle(docs)

with open(output_file, "w") as f:
    for doc in docs:
        f.write(doc + "\n")

print(f"wrote {len(docs)} conversations to {output_file}")
