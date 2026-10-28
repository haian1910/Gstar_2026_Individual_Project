# Fade, Don't Forget

GStar Bootcamp 2026 (LLM) individual project, Vu Hai An.

**Fade** is a training-free KV-cache policy for hybrid reasoning models. We test it on Qwen3.5-9B, which has 24 Gated DeltaNet layers and 8 full-attention layers.
- The prompt and the last ~2K generated tokens stay exact.
- Older full-attention K/V are stored at 2 or 4 bits (KIVI-style, blocks of 128 tokens).
- The Gated DeltaNet layers are not touched.

At 32K tokens, the full-attention KV cache is 1 GiB per sample, 95% of decode-state memory.

Results on 120 competition-math problems (AIME 2025, AIME 2026, HMMT 2025, MATH-500 level 5; 4 samples each):

| Cell | Accuracy | Memory vs full |
|---|---|---|
| Full attention | 65.8% | 1× |
| Fade-4 (4-bit) | 66.2% | 0.37× |
| Fade-2 (2-bit, Hadamard) | 60.2% | 0.27× |
| 8K window (prompt + last 8K tokens) | 56.5% | 0.39× |
| StreamingLLM-8K (2 samples) | 12.5% | 0.38× |

- **Primary, pre-registered:** Fade-2 vs the 8K window at matched memory is **+3.8 pp [+1.0, +6.7]**.
- **4-bit:** Fade-4 vs full attention shows no detectable difference (+0.4 pp [−2.7, +3.5]).

The paper is in [`paper/`](paper/) and covers the full set of contrasts and limitations.

## Layout

```
experiments/   decode loop, quantizer, generation, scoring, memory benchmark, KL / layer / head maps (GPU)
  tests/       exactness and unit tests
analysis/      pre-registered and exploratory statistics (stdlib; h1_buckets needs numpy)
plots/         figures for the paper and slides (matplotlib, python-pptx)
results/
  runs/        one row per run (no generated text): the inputs to every number in the paper
  stats/       JSON written by the analysis scripts
prereg/        pre-registrations with their change logs (prereg_fade.md; prereg_tl.md for the Think-Local baselines)
paper/         write-up source (LaTeX) and PDF
slides/        the 2 presentation slides
```

## Reproducing the analysis (CPU, a few seconds)

These commands rebuild `results/stats/` from `results/runs/`. We checked that they give identical JSON from this checkout.

```bash
TL="results/runs/tl_runs_full_k4.jsonl results/runs/tl_runs_tl-2048_k4.jsonl results/runs/tl_runs_tl-4096_k4.jsonl results/runs/tl_runs_tl-8192_k4.jsonl"
FD="results/runs/fd_runs_fq-kv2_had_k4.jsonl results/runs/fd_runs_fq-kv4_k4.jsonl"
python3 analysis/fade_stats.py   --runs $TL $FD --cell-a kv2_had --cell-b kv4 --out results/stats/stats_fade_k4.json   # registered
python3 analysis/fade_explore.py --runs $TL $FD --out results/stats/stats_fade_explore_k4.json                        # post hoc
python3 analysis/fade_sl.py      --runs $TL $FD results/runs/sl_runs_sl-8192_k2.jsonl --out results/stats/stats_fade_sl.json
```

Figures (needs matplotlib):

```bash
python plots/fade_paper_figs.py --stats results/stats/stats_fade_k4.json --sl results/stats/stats_fade_sl.json --out paper/figures
python plots/fade_schematic.py  --out paper/figures
python plots/fade_plots.py --stats results/stats/stats_fade_k4.json --kl results/stats/h1_q35_math.json \
  --h1 results/stats/h1.json --pulse results/stats/pulse_stats.json --sl results/stats/stats_fade_sl.json --out paper/figures --no-notes
```

`plots/make_slides.py` also needs the course slide template and fonts in `slides/build/`. These are not included.

## Running the experiments (GPU)

These are the commands as we used them. They have not been re-tested from a clean checkout. Run them from `experiments/` with `bash setup_env.sh` and `bash download_models.sh` done first. Our runs used NVIDIA A100-80GB GPUs, Transformers 5.19, PyTorch 2.13 and fla 0.6.0.

**1. Problems and full-attention traces**
```bash
python pilot_data.py out/problems.jsonl
python gen_traces.py --problems out/problems.jsonl --out out/traces_s0.jsonl
```
Datasets are downloaded from Hugging Face. Problem files and traces are not committed.

**2. Teacher-forced KL sweep, used to pick the cells**
```bash
python kv_fade.py --problems out/problems.jsonl --traces out/traces_s0.jsonl --out fd.jsonl [--shard g --nshards 2]
```
Run with `--smoke` first.

**3. Dense-model comparison and pulse analysis (exploratory)**
```bash
python kv_fade_dense.py ...
python kv_pulse.py [--dense] --traces out/traces.jsonl --problems out/problems.jsonl --out pulse.jsonl
```
- `kv_fade_dense.py --tiny` runs an exactness check on CPU.
- The registered pulse run used `kv_pulse_registered.py`. `kv_pulse.py` adds `--distinct`.

**4. Layer and head maps**
```bash
python layer_map.py --configs ... --out lm.jsonl
FADE_ROOT=<run dir> python head_map.py --source e1 --n 40 --out hm.jsonl
```

**5. Decoding (the main experiment)**
```bash
python tl_gen.py --problems out/problems.jsonl --select-ids math_ids.txt \
  --cell {full|tl-8192|sl-8192|fq-kv4|fq-kv2_had} --k 4 --max-new 32768 --bmax 32 --out gen.jsonl
python tl_score.py --gen gen.jsonl --problems out/problems.jsonl --out runs.jsonl
```

**6. Measured memory**
```bash
python fade_mem.py --check
python fade_mem.py --bench --cells full,tl-8192,sl-8192,fq-kv4,fq-kv2_had \
  --ctx 8192,16384,32768 --batch 1,8,32 --steps 128 --out fade_mem.jsonl
python ../analysis/fade_mem_stats.py --rows fade_mem.jsonl --out ../results/stats/stats_fade_mem.json
```

**7. Tests**
```bash
python fade_mem_test.py
python fade_q_unit.py
python fade_dec_test.py
pytest tests/
```

The decode emulates quantization in bf16; only `fade_mem.py` stores packed codes. No speed claim is made, since dequantization is unfused PyTorch.
