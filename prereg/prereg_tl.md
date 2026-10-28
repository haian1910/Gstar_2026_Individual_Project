# Pre-registration: Think-Local on a hybrid reasoner (Qwen3.5-9B)

> Drafted by nti-e2 on 2026-10-26 at ~17:50 UTC. Fixed before any Think-Local run is scored. Changes after the first scored run are listed at the end with a time and a reason. Analysis: `src/tl_stats.py`.

## Question
In a hybrid (Qwen3.5-9B: [3× Gated DeltaNet + 1× gated full attention] × 8), can the full-attention (FA) layers be limited to the prompt plus a recent window of generated tokens, with the GDN layers untouched and no training? Is GDN's recurrent state a better long-range carrier than re-encoded text (Delethink) at the same peak FA memory?

## Cells (same fa_loop, same sampler as E3; seeds are nominal, so every contrast pairs by problem)
| Cell | FA sees | GDN | Re-encode |
|---|---|---|---|
| Full | everything | untouched | none |
| TL-W (W = 1K, 2K, 4K) | prompt + last W generated tokens (ring buffer, original RoPE positions) | untouched | none |
| DT(C = 4K, m = 2K) | prompt + current chunk; at each reset, prompt + last m tokens re-encoded, positions restart after the prompt | rebuilt at each reset | m tokens per reset |
| SL-2K (StreamingLLM) | 4 sink tokens + last 2K tokens, prompt **not** pinned | untouched | none |

DT follows Delethink's convention: the context after the prompt grows from m to C, then both memories are rebuilt from prompt + the last m tokens. So DT's peak FA memory of generated tokens is C, and each chunk generates C − m new tokens. TL-W4K and DT(4K/2K) therefore match on peak FA memory (TL holds 4K throughout; DT averages ~3K).

## Data
- Math: 120 problems (AIME25 30, AIME26 30, HMMT25 30, MATH-500 level 5 30), k = 4, 32K-token cap. A 16K view is derived post hoc: a run counts as finished and correct at 16K only if it finished within 16,384 tokens.
- Countdown: 100 four-number tasks (`Jiayi-Pan/Countdown-Tasks-3to4`), k = 4. Scored by the unit-tested validator [annotation 2026-10-27 ~12:50 UTC: the unit test, `code/tests/test_countdown.py` with 15 cases, was added at the audit; before that, the validator had only a sanity script]: each number used exactly once, the expression equals the target, parsed by an AST walk with no exec.
- Problem ids are fixed in `out/tl_ids.txt` (md5 b699f214; 220 ids). AIME25, AIME26 and HMMT25 are complete; MATH-500 L5 is 30 drawn with Random(0); Countdown is 100 of 200 drawn with Random(0).
- **k = 8 extension.** Samples 4–7 of Full and TL-W2K are queued after every k = 4 cell is done.
  - They count only if they are complete for both cells on all 220 problems by **2026-10-27 22:00 UTC**. They are never stopped or extended because of results.
  - If they count, they are used for the TL-W2K − Full contrasts only (primary, S3 at W = 2K, S4, S5). Every other contrast, cell summary and matched-memory figure uses samples 0–3, so all cells have the same k.
  - If they don't count, k = 4 everywhere.

## Primary analysis (one test)
- **Contrast:** math accuracy at the 32K cap, TL-W2K − Full.
- **Estimator:** the mean over problems of the per-problem difference in accuracy (each over its k samples), paired by problem.
- **CI:** two-sided 95% percentile CI, from a cluster bootstrap that resamples problems within each dataset (10,000 resamples, seed 0).
- **Non-inferiority margin δ = 5 pp.** TL-W2K is called non-inferior only if the CI's lower bound is above −5 pp.
- **Wording rule:**
  - CI lower bound > −5: "non-inferior at 5 pp".
  - CI upper bound < 0: "worse".
  - CI upper bound < −5: "worse by more than the margin".
  - Anything else: "inconclusive at n = 120".
  - Never "no loss".
- **Why 5 pp:**
  - (i) A Full sample of 12K–32K generated tokens uses ~4–9× the per-sample FA + GDN memory of TL-W2K (below). At matched memory TL can run that many samples, and majority voting over ~4 samples often gains a few points on AIME-level sets. A per-sample loss above 5 pp would likely cancel that gain, so 5 pp is where the trade-off becomes doubtful.
  - (ii) E3 ruled out a loss above 7 pp for a much smaller deletion. A margin at 7 pp or more would not test anything new.
- **Power:** we expect a CI half-width of ~4.5 pp at k = 4 (per-run variance ~0.12, paired, 120 problems).
  - If the true difference is 0, P(non-inferior) ≈ 0.6.
  - If it is −3 pp (E3's point estimate), ≈ 0.15.
  - "Inconclusive" is a likely outcome and will be reported as such. The CI is always shown, so readers can apply their own margin.
- **Reported with the primary (descriptive, same bootstrap):**
  - mean generated tokens, capped at 32K;
  - finished rate (ended with EOS within the cap);
  - per-problem difference in mean tokens over finished runs (the cost, conditional on outcome).

## Secondary analyses (pre-specified, no multiplicity correction, labelled secondary)
- **S1. Recurrent state vs text carryover:** TL-W4K − DT(4K/2K), math, accuracy and tokens. Two-sided CI, no margin. Expected direction: TL ≥ DT.
- **S2. Pinning the prompt:** TL-W2K − SL-2K, math accuracy. Expected TL > SL if FA mainly serves as an input reader.
- **S3. Window size:** TL-W1K, TL-W2K and TL-W4K each minus Full, math, accuracy and tokens. Expected monotone in W.
- **S4. Non-math:** Countdown, TL-W − Full accuracy, with the same 5 pp margin and wording rule.
  - W is fixed from the 9B smoke test's Full Countdown lengths alone (no accuracy), before any main run is scored. The rule was written 2026-10-26 ~18:05 UTC, before the lengths were seen.
  - If at least half of the Full Countdown smoke runs finish within 2,048 tokens, W = 1K. Otherwise W = 2K.
  - The same W is used for the Countdown moderator. If W = 1K, the Countdown k = 8 extension runs Full and TL-W1K.
  - **Applied (2026-10-26 ~18:20 UTC):** 0 of 16 Full Countdown smoke runs finished within 2,048 tokens (0 of 16 within 2,560), so W = 2K.
- **S5. 16K budget:** the primary contrast and S1 re-scored on the 16K view.

## Exploratory (descriptive only)
- **Repetition:** rep_score = the fraction of a generation's token 64-grams that already occurred earlier in the same generation; `repetitive` = rep_score ≥ 0.30. The threshold is a constant in `tl_score.py`, fixed by research-54 on 2026-10-26 before any run, and written on each row. Reported as the fraction of repetitive runs, plus the truncation rate, per cell.
- **Per-dataset breakdown** of every contrast.
- **Memory, analytic:**
  - FA KV is 32 KiB per token (8 layers × 4 KV heads × 256 × K,V × bf16).
  - GDN state is 24 layers × 32 heads × 128 × 128 × fp32 = 48 MiB, plus a conv state of 24 × 8,192 × 3 × bf16 ≈ 1.1 MiB, so S ≈ 49 MiB per sample.
  - The prompt's FA KV is shared across samples (prefix caching) and is left out of the per-sample cost.
  - Per-sample cost: Full = n_gen × 32 KiB + S; TL-W = min(n_gen, W) × 32 KiB + S; SL = min(n_gen, W + 4) × 32 KiB + S; DT = min(n_gen, C) × 32 KiB + S. Throughput is reported only "in our HF loop".
- **Matched memory:**
  - m = ⌊mean Full cost / mean cell cost⌋, capped at k.
  - Math: majority vote over m samples, averaged over all m-subsets of the k samples. Ties count as the expected accuracy under a random tie-break, and unfinished samples cast no vote. Compared with Full pass@1.
  - Countdown: pass@m, because the validator needs only the problem's inputs, so best-of-m verification is legitimate at test time.
- **DT compute:** tokens re-encoded per run (resets × m).
- **Share of runs beyond the window:** per cell, the fraction of runs longer than W (TL, SL) or C (DT). In a run shorter than that the window never acts, so it is a draw from Full's distribution; this share dilutes every contrast.
- **Moderator: trigger density** (added 2026-10-26 ~18:00 UTC, before any run was scored; from haian's suggestion to look at many-attempt tasks).
  - Density = trigger sentences per 1K thinking tokens (tokens before `</think>`, or all tokens if it never appears; sentence-initial Wait / Alternatively / Hmm / Actually, negations, restarts: the `classify()` rules of the Forget-the-Attempt boundary code), computed in `tl_score.py` as `n_triggers_per_1k`.
  - It is called trigger density, not attempt count. The Forget-the-Attempt audit found that most trigger sentences are re-checks: 0 strict abandons in 160 random trigger sentences.
  - Statistic: Spearman between a problem's density (from its Full samples) and its TL-W2K − Full accuracy difference.
  - Cross-fitted so the density and the difference never share samples: density from Full samples {0,1} against the difference on samples {2,3}, and the reverse, averaging the two rhos. Without this, a Full sample that fails by chance (and so says "Wait" more) inflates both sides.
  - Bootstrap over problems within datasets (10,000 resamples, seed 0). Math and Countdown separately.
  - Also reported: Spearman(density, Full accuracy), as the difficulty confound.
  - Expected sign: positive. Exploratory; no decision depends on it.

## Exclusions and missing data
- Runs that crash are rerun with the same seed.
- Any run still missing at analysis time is reported. A problem missing a cell is dropped only from the contrasts that involve that cell, and the count is reported.
- No problem is dropped for being too easy or too hard.

## Clarifications before any main run was scored
- 2026-10-26 ~18:20 UTC: the 9B smoke test showed that same-seed Full and TL runs share only 1–332 tokens of prefix (bf16 batch-size effects, not the window). So seeds are nominal. Every contrast, including the finished-only token difference, pairs by problem, never by sample_k. The power estimate above already assumed independent cells within a problem.
- 2026-10-26 ~18:20 UTC: Countdown W fixed at 2K by the S4 length rule (above).
- 2026-10-26 23:05 UTC: the plan did not state Countdown's token cap. The generation code sets 32,768 for both domains, as fixed before any run (`max_tokens` on every row). This records a fact; nothing was chosen. Written after the Full cell's descriptives were seen (math 65.8% accuracy, 34% capped; Countdown 96.0%), but before any contrast was computed.

## Changes after the first scored run
- **2026-10-27 ~01:25 UTC: the k = 8 extension is cancelled. Its GPU time goes to the window dose-response.** This was decided by research-b0 and nti-e2 after seeing the primary (TL-W2K − Full −40.0 pp, 95% CI −47.1 to −32.9, k = 4).
  - This breaks the letter of the k = 8 rule above ("never stopped … because of results"). The extension existed only to narrow the TL-W2K − Full CIs. With the upper bound 28 pp below the margin, it cannot change any verdict: the primary, S4 and S5 are all "worse by more than the margin".
  - k = 4 is used everywhere, which is the plan's own fallback.
- **2026-10-27 ~01:25 UTC: an added cell, TL-W8K (post hoc, exploratory).** It runs after the pre-registered k = 4 cells, with the same 220 problems, k = 4 and settings.
  - It is reported only as a point in the window dose-response (accuracy, finished and tokens against W, vs Full), and labelled post hoc. No verdict.
  - The queue order changes to DT(4K/2K) (running), TL-W4K, TL-W8K, SL-2K, TL-W1K, all k = 4. TL-W8K goes before the last two pre-registered cells because the W sweep now carries the result. Every pre-registered cell still runs (estimated done by ~13:00 UTC). A cell not finished by the write-up deadline is reported as not run.
  - Mechanics (research-b0): the running chain is not edited. Its remaining tasks are pre-claimed, and a second chain (`chains/tl2.sh`, md5 6237e0c0) runs TL-W8K, SL-2K and TL-W1K with the identical generator (tl_gen 065f47a7), batch size, seeds, shards and ids.
- **Exploratory analyses added after seeing TL-W2K (post hoc, descriptive):**
  - accuracy among finished runs per cell;
  - TL accuracy binned by Full's per-problem mean length;
  - the distribution of finished lengths.
- **2026-10-27 ~03:45 UTC: an implementation fix in `src/tl_stats.py`, no analysis change.** Two loops ran over Python sets of (dataset, problem) keys, so the bootstrap draws depended on per-process string hashing. They were (1) the finished-only token difference and (2) the moderator's cross-fitted rows. Both loops are now sorted, and the output is identical across hash seeds.
  - It moved only the Monte Carlo digits of those two families of CIs (e.g. S5 finished-only lower bound +4 → −1 tokens; Countdown Spearman(density, Full accuracy) upper bound +0.11 → +0.12).
  - No point estimate, accuracy, finished or token CI, or verdict changed.
- **2026-10-27 07:28 UTC (operations, no design change; research-1f, with haian's OK):** the remaining cells (TL-W8K, SL-2K, TL-W1K) also run on GPUs 2–3. Workers take shards from the same queue, so each shard still runs once, with the same generator, seeds, sampler, batch size and ids. Seeds were already nominal (above), so the GPU a shard ran on cannot affect any pairing.
- **2026-10-27 ~10:47 UTC (operations: SL-2K shards 1–2 were moved from GPUs 2–3 back to GPU 0; written before any SL-2K run was scored).** research-4d stopped the two shards at 10:44 because other users' processes slowed them, then resumed them with the same args and seeds (`chains/tl4.sh`). Finished rows were kept. Rows that were mid-generation restart from scratch.
  - **Possible bias.** The rows in flight when a shard stops are more likely to be long runs, because long runs occupy a slot longer. Redrawing only those rows, while keeping the short rows that had finished, shifts those slots toward shorter runs. Shorter runs finish more often, so the shift favours SL-2K, and S2 (TL-W2K − SL-2K) would be biased toward zero.
  - **Pre-specified sensitivity check (before SL-2K is scored).** Given the list of restarted (problem, sample_k) rows, S2 and the SL-2K cell summary are also reported with those rows (a) scored as unfinished and wrong (the worst case for SL), and (b) left out, with per-problem means taken over the remaining samples.
  - S2 as pre-registered uses every row as generated. The as-generated value and (a) bracket the effect. If the restarted rows are ≤ 1% of the cell's runs, the check is reported in one line.
  - **Superseded at 10:48 UTC, before any SL-2K row was scored.** research-4d discarded both stopped shards (155 rows each, kept aside as `*.stopped_1044` and never scored) and is regenerating all 220 rows of each in one uninterrupted run (`chains/tl5.sh`). The discarded set is defined by shard, not by run length, so there is no selection, and the sensitivity check is not needed.
  - From here on, any shard stopped by `chains/gpu_guard.sh` (which releases a GPU when another user's process appears) is rerun in full, never resumed.
  - No crash reruns occurred in any cell (logs checked by research-4d).
- **2026-10-27 ~12:40 UTC (a fact found after SL-2K was scored; no analysis change).** The regenerated SL-2K shards 1–2 reproduce, bit for bit and in the same order, the 155 rows that each discarded shard had finished: 310 of 310 token sequences are identical. research-4d and nti-e2 checked this independently on `results/sl2048_fresh_check/`. So generation is deterministic for an identical rerun (same cell, shard, pool order, seeds and batch size), even on another A100, and the regenerated shards are exactly what an uninterrupted run would have produced. The 18:20 UTC clarification (same-seed runs share only 1–332 tokens) holds across cells and batch shapes, not for identical reruns. Seeds stay nominal for pairing across cells.
- **2026-10-27 ~12:50 UTC (exp-auditor round 1; disclosure and provenance only, no number or verdict changed).** The auditor recomputed the primary and S1–S5 from the run files and they match. Changes made in response:
  - The write-up now discloses the k = 8 cancellation (§3).
  - The write-up now says SL-2K keeps original RoPE positions (no StreamingLLM re-indexing).
  - SL and DT were checked by slot and rebuild accounting only, not against a reference implementation; this is now stated.
  - The wrong denominator is fixed (2 of 6,160 runs).
  - Exploratory "matches Full" wording is softened to the CI.
  - `math_verify` was absent for the statistics; this affects only the exploratory maj@m numbers.
  - The stats command and input order are recorded in `results/stats_tl_repro.txt`.
  - The trace-reading counts are documented in `results/tl_trace_reading.md`.
  - The Countdown unit test now exists (line 20 annotation above).
