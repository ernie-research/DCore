"""Count perfectblend conversations whose full tokenized length < 3072.

Replicates the training preprocessing exactly (GeneralParser + qwen template):
  1. prepend system prompt "You are a helpful assistant." when absent
  2. tokenizer.apply_chat_template(..., tokenize=False, add_generation_prompt=False)
  3. tokenize with add_special_tokens=False (no truncation here; we measure)

Usage: python3 scripts/count_perfectblend_lengths.py
"""

import json
import os
import sys
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(
    ROOT, "cache/dataset/perfectblend/perfectblend_train_regen_qwen3_8b.jsonl"
)
TOKENIZER_PATH = os.path.join(ROOT, "cache/models/Qwen3-8B")
MAX_LEN = 3072
NUM_WORKERS = 96
CHUNKS = 384
SYSTEM_PROMPT = "You are a helpful assistant."

_tok = None


def get_tok():
    global _tok
    if _tok is None:
        from transformers import AutoTokenizer

        _tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    return _tok


def process_chunk(args):
    start, end = args
    tok = get_tok()
    n_total = n_below = n_err = 0
    # histogram buckets: <1024, <2048, <3072, <4096, <8192, >=8192
    edges = [1024, 2048, 3072, 4096, 8192]
    hist = [0] * (len(edges) + 1)
    sum_len = 0
    max_len_seen = 0
    with open(DATA, "rb") as f:
        f.seek(start)
        if start > 0:
            f.readline()  # skip partial line (owned by previous chunk)
        while f.tell() <= end:
            line = f.readline()
            if not line:
                break
            try:
                sample = json.loads(line)
                messages = sample["conversations"]
                if messages and messages[0].get("role") != "system":
                    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
                text = tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                n = len(tok(text, add_special_tokens=False).input_ids)
            except Exception:
                n_err += 1
                continue
            n_total += 1
            sum_len += n
            max_len_seen = max(max_len_seen, n)
            if n < MAX_LEN:
                n_below += 1
            for i, e in enumerate(edges):
                if n < e:
                    hist[i] += 1
                    break
            else:
                hist[-1] += 1
    return n_total, n_below, n_err, hist, sum_len, max_len_seen


def main():
    size = os.path.getsize(DATA)
    bounds = [size * i // CHUNKS for i in range(CHUNKS + 1)]
    jobs = list(zip(bounds[:-1], bounds[1:]))

    tot = below = err = 0
    hist = [0] * 6
    ssum = 0
    mx = 0
    done = 0
    with Pool(NUM_WORKERS) as pool:
        for r in pool.imap_unordered(process_chunk, jobs):
            t, b, e, h, s, m = r
            tot += t
            below += b
            err += e
            ssum += s
            mx = max(mx, m)
            hist = [a + x for a, x in zip(hist, h)]
            done += 1
            if done % 24 == 0 or done == CHUNKS:
                print(
                    f"progress {done}/{CHUNKS} chunks | so far total={tot:,} "
                    f"below3072={below:,} ({below / max(tot, 1) * 100:.2f}%)",
                    flush=True,
                )

    print("\n==== FINAL ====")
    print(f"total samples          : {tot:,}  (parse errors: {err})")
    print(f"len < 3072             : {below:,}  ({below / tot * 100:.3f}%)")
    print(f"len >= 3072 (truncated): {tot - below:,}  ({(tot - below) / tot * 100:.3f}%)")
    print(f"mean length            : {ssum / tot:.1f} tokens; max: {mx:,}")
    labels = ["<1024", "1024-2047", "2048-3071", "3072-4095", "4096-8191", ">=8192"]
    for lab, c in zip(labels, hist):
        print(f"  {lab:>10}: {c:>10,}  ({c / tot * 100:6.2f}%)")


if __name__ == "__main__":
    sys.exit(main())
