"""E1 traces: Qwen3.5 thinking mode, n samples per problem, one vLLM replica per GPU (shard = problem index mod nshards).
Writes one JSON line per sample as each chunk finishes (resumable: skips problem_ids already in --out)."""
import argparse, json, os, time
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."  # model card, math
# a problem row may carry its own "suffix" (abandon-pilot: Countdown, HumanEvalFix); math rows have none
# model card, "thinking mode for general tasks"
SAMPLING = dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0)

ap = argparse.ArgumentParser()
ap.add_argument("--problems", default="problems.jsonl")
ap.add_argument("--out", required=True)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--max-tokens", type=int, default=32768)
ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
ap.add_argument("--gpu-mem", type=float, default=0.90)
ap.add_argument("--chunk", type=int, default=0,
                help="problems per generate() call; 0 = the whole shard in one call (no per-chunk 32K-token tail)")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--base-seed", type=int, default=0)
a = ap.parse_args()

def main():  # vLLM starts its engine with spawn: keep module-level code import-safe
    probs = [json.loads(l) for l in open(a.problems)]
    probs = [(gi, p) for gi, p in enumerate(probs) if gi % a.nshards == a.shard]
    if a.limit:
        probs = probs[: a.limit]
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            done.add(json.loads(l)["problem_id"])
    probs = [(gi, p) for gi, p in probs if p["problem_id"] not in done]
    CH = a.chunk or len(probs) or 1
    print(f"shard {a.shard}/{a.nshards}: {len(probs)} problems to do ({len(done)} done)", flush=True)
    if not probs:
        raise SystemExit(0)

    def seed_of(gi):  # request seed; vLLM V1 parallel sampling gives child k seed + k, so space problems by 16
        return a.base_seed * 100003 + 16 * gi


    tok = AutoTokenizer.from_pretrained(a.model)
    # explicit stop set (the repo has no generation_config.json): same set as e3_branch.py / fta_common.TokBytes.eos
    STOP = sorted({tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")})
    llm = LLM(a.model, max_model_len=a.max_tokens + 2048, gpu_memory_utilization=a.gpu_mem, enable_prefix_caching=True,
              seed=a.base_seed, limit_mm_per_prompt={"image": 0, "video": 0}, max_num_seqs=256)
    for s in range(0, len(probs), CH):
        chunk = probs[s : s + CH]
        prompts, params = [], []
        for gi, p in chunk:
            ids = tok.apply_chat_template([{"role": "user", "content": p["problem"] + p.get("suffix", SUFFIX)}], add_generation_prompt=True,
                                          enable_thinking=True, tokenize=True)
            if not isinstance(ids, list):
                ids = ids["input_ids"]
            prompts.append(TokensPrompt(prompt_token_ids=ids))
            params.append(SamplingParams(n=a.n, max_tokens=a.max_tokens, seed=seed_of(gi), stop_token_ids=STOP, **SAMPLING))
        t0 = time.time()
        outs = llm.generate(prompts, params, use_tqdm=True)
        ntok = 0
        with open(a.out, "a") as f:
            for (gi, p), o in zip(chunk, outs):
                for k, c in enumerate(o.outputs):
                    ntok += len(c.token_ids)
                    f.write(json.dumps({"problem_id": p["problem_id"], "dataset": p["dataset"], "sample_idx": k, "reference": p["answer"],
                                        "prompt_ids": list(o.prompt_token_ids), "prompt_len": len(o.prompt_token_ids),
                                        "output_ids": list(c.token_ids), "text": c.text, "finish_reason": c.finish_reason,
                                        "n_tokens": len(c.token_ids), "seed": seed_of(gi)}) + "\n")
        dt = time.time() - t0
        dup = sum(len({tuple(c.token_ids[:64]) for c in o.outputs}) < len(o.outputs) for o in outs)
        print(f"chunk {s // CH}: {len(chunk)} problems, {ntok} tokens, {dt:.0f}s, {ntok / dt:.0f} tok/s, "
              f"problems with duplicate samples (first 64 tokens) {dup}", flush=True)


if __name__ == "__main__":
    main()
