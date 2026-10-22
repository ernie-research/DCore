"""Baseline autoregressive generation for speedup comparison."""

import argparse
import json
import time
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "third_party", "Domino", "code"))

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from model import load_and_process_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--answer-file", type=str, required=True)
    args = parser.parse_args()

    attn_impl = "eager"
    print(f"Attention: {attn_impl}")

    device = torch.device("cuda:0")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, attn_implementation=attn_impl, dtype=torch.bfloat16
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    dataset = load_and_process_dataset(args.dataset)
    if args.max_samples and len(dataset) > args.max_samples:
        dataset = dataset.shuffle(seed=0).select(range(args.max_samples))

    answers = []
    total_tokens = 0
    total_time = 0.0

    def eval_sample(idx):
        instance = dataset[idx]
        messages = []
        choice = {"index": 0, "turns": [], "new_tokens": [], "wall_time": []}
        for user_content in instance["turns"]:
            messages.append({"role": "user", "content": user_content})
            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature if args.temperature > 0 else None,
                    do_sample=args.temperature > 0,
                    eos_token_id=tokenizer.eos_token_id,
                )
            torch.cuda.synchronize()
            wall = time.perf_counter() - t0

            gen_ids = output_ids[0, input_ids.shape[1]:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            choice["turns"].append(text)
            choice["new_tokens"].append(int(gen_ids.shape[0]))
            choice["wall_time"].append(wall)
            messages.append({"role": "assistant", "content": text})

        return {"question_id": idx, "choices": [choice], "tstamp": time.time()}

    # Run the first sample twice and discard the first (warmup) pass, so every
    # written sample is a hot measurement (matches the benchmark eval scripts).
    pbar = tqdm(range(len(dataset)), desc=args.dataset)
    for pos in pbar:
        if pos == 0:
            eval_sample(pos)  # warmup pass (discarded)
        ans = eval_sample(pos)
        answers.append(ans)
        ch = ans["choices"][0]
        # Turn-0 protocol, matching the benchmark eval scripts' token/s aggregation.
        total_tokens += ch["new_tokens"][0]
        total_time += ch["wall_time"][0]
        avg_tps = total_tokens / total_time if total_time > 0 else 0
        pbar.set_postfix({"avg_tok/s": f"{avg_tps:.1f}", "tokens": total_tokens})

    print(f"\nAvg token/s (excl. warmup): {total_tokens / total_time:.1f} ({total_tokens} tokens in {total_time:.1f}s)")

    os.makedirs(os.path.dirname(args.answer_file) or ".", exist_ok=True)
    with open(args.answer_file, "w") as f:
        for ans in answers:
            f.write(json.dumps(ans, ensure_ascii=False) + "\n")
    print(f"Saved {len(answers)} samples to {args.answer_file}")


if __name__ == "__main__":
    main()
