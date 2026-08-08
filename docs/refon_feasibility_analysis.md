# Are RefON instructions solvable before the robot moves?

A RefON instruction is anaphoric: *"Find A1."*, *"Find the 2nd one again."*, *"Go back to
the previous one."* A navigation failure on one of those has two possible causes, and the
navigation score cannot tell them apart:

1. the agent never worked out **which object** the instruction meant, or
2. it knew, and failed to **find** it.

Everything below isolates (1). No habitat, no images, no navigation — the agent sees only
the instructions. The result is a ceiling: a subgoal that fails here cannot succeed in
navigation for reasons that have anything to do with perception or exploration.

Two probes, run through `ollama` on a single RTX 4090:

| probe | question | answer | runner |
|---|---|---|---|
| **referent** | which object does this instruction mean? | `new` / `back_reference` + instruction number / `no_object`, plus the object's category | `run_refonbench_feasibility.py` |
| **destination** | where does the robot go? | `explore` / `(x, y)` / `infeasible` | `run_refonbench_feasibility_nav.py` |

---

## 1. Method

### 1.1 The referent probe

Instructions are presented **incrementally** — at subgoal *i* the model sees instructions
1..*i* and is asked about *i*, which is exactly what a navigation run has at that point.
(An `all_at_once` mode showing the whole episode also exists; see §5.)

Scoring splits into two part-scores that are reported separately, because a model can
point at the right object and still fail to name it:

- **referent SR** — did it point at the right object?
- **category SR** — did it name that object's category?
- **joint SR** — both.

Two rules matter:

**Any *earlier* instruction that lands on the same `object_id` counts.** An episode can
visit one object several times, so "the 2nd one" and "the 3rd one" can both be true of the
same referent. The subgoal's *own* number is excluded on purpose — it trivially carries
the right `object_id`, so accepting it would score "this is a new object" as a correct
resolution of a back reference.

**The three goal-absent kinds are not the same question** for a text-only probe:

| kind | example | expected | why |
|---|---|---|---|
| `GA_unbound_alias` | "Find Z1." | `no_object` | the alias was never bound — the text says so |
| `GA_invalid_ordinal` | "Find the 8th one again." after 6 visits | `no_object` | out of range — the text says so |
| `GA_absent_object` | "Find the chandelier." | `new` + "chandelier" | goal-absent because the *scene* has no chandelier, which the model was never told |

Demanding `no_object` for the third would score a fact the model does not have.

### 1.2 The destination probe

Every object in an episode gets a made-up `(x, y)`, seeded per episode so every model sees
the same map. Positions are disclosed **only as earlier subgoals reach them**, so exactly
one action is right at each step: `explore` for an object nobody has visited, the
coordinates copied from the log for one already found, `infeasible` for an instruction
that cannot be carried out. Coordinates are kept ≥1.0 apart so a near miss cannot score as
a hit.

`GA_absent_object` gets **two turns** here, which is the only fair way to score it:

```
turn 1   "Find the chandelier."                     -> explore      (correct)
env      the robot searched and reported: not_found
turn 2   same instruction, now with that outcome    -> infeasible   (correct)
```

The second turn is reported as its own style, `GA_absent_object/not_found`.

The log shows **what actually happened, not what the model answered**. Every turn is
therefore an independent query — the run parallelises, and one bad answer at subgoal 3
cannot cascade into a run of failures at 4..N that measure nothing.

### 1.3 Datasets

Both built on the free HM3D example scene `00861-GLAQ4DNUx5U`, with instruction styles
balanced (the generator's `even_styles` dimension) so per-style SRs have comparable sample
counts.

| set | episodes | subgoals | lengths | ref. distance | config |
|---|---|---|---|---|---|
| **balanced** | 200 | 1308 | 4–8 | ≤7 | `generator.example_even.json` |
| **long** | 100 | 2273 | 5/14/23/32/41/50 | ≤49 | `generator.example_long.json` |

Style shares land at 12.7–17.2% (balanced) and 11.7–16.8% (long) against a 14.3% uniform
target. The residual `AB_post` surplus is structural: one alias bound by `AB_pre` can be
referenced by several `AB_post`, so those tokens outnumber their binders by design.

### 1.4 Models

All four run **non-thinking**, so the comparison is one condition throughout (see §6.1 for
why). `gemma4:26b-a4b-it-qat` appears as a thinking-enabled reference point only.

| model | size | note |
|---|---|---|
| `qwen3.5:2b` | 2B | |
| `qwen3.5:4b` | 4B | |
| `qwen2.5vl:7b` | 7B | previous generation |
| `qwen3.5:9b` | 9B | |
| `gemma4:26b-a4b-it-qat` | 26B (4B active) | thinking on — not directly comparable |

`qwen3.5:0.8b` was excluded: it spends the whole token budget thinking and returns empty
content (16384 completion tokens, 66k characters of `reasoning`, 240s a call).
`qwen3-vl:8b` was excluded from the non-thinking comparison because it cannot be put in
non-thinking mode — `think=False`, `/no_think`, `enable_thinking=False` and
`reasoning_effort=none` are all ignored (5089 tokens as-is vs 5701 with the flag).

---

## 2. Referent probe, balanced set (1308 subgoals)

| style | 2b | 4b | 7b | 9b | gemma4 26b† |
|---|---|---|---|---|---|
| S | 79.0 | 90.9 | **98.4** | 94.6 | 91.4 |
| AB_pre | 75.5 | 87.1 | **99.4** | 95.7 | 88.3 |
| AR_pre | 82.7 | 93.0 | **100.0** | 97.3 | 95.1 |
| AB_post | 0.0 | 71.4 | 53.2 | 85.9 | **97.7** |
| AR_post | 2.7 | 43.8 | 14.1 | 64.3 | **96.8** |
| OR_post | 0.0 | 52.2 | 11.8 | 84.8 | **94.9** |
| AB_pre+OR_post | 0.0 | 22.2 | 2.5 | 75.3 | **97.5** |
| GA_absent_object | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| GA_invalid_ordinal | 0.0 | 15.4 | 0.0 | 84.6 | **100.0** |
| GA_unbound_alias | 0.0 | 0.0 | 0.0 | 28.6 | **100.0** |
| **ALL** | **33.4** | **65.8** | **54.1** | **85.2** | **94.8** |
| *names its target* | 79.2 | 90.4 | **99.3** | 95.9 | 91.8 |
| *back-reference* | 0.7 | 49.3 | 22.6 | 78.0 | **96.8** |

† thinking enabled; shown for reference, not as part of the ladder.

**The split at the bottom is the whole story.** Every model is decent at instructions that
name their own target and they separate almost entirely on the ones that do not.

---

## 3. Destination probe, balanced set (1317 turns)

| | 2b | 4b | 7b | 9b |
|---|---|---|---|---|
| action SR | 28.9 | 85.0 | 71.1 | **90.2** |
| coord SR | 43.1 | 68.2 | 65.9 | **78.8** |
| **joint SR** | **28.3** | **63.8** | **60.7** | **77.9** |
| *names its target* | 63.9 | 80.5 | — | **91.0** |
| *back-reference* | 0.4 | 51.1 | — | **68.1** |

Per style for `qwen3.5:9b`, action vs joint:

| style | action SR | joint SR | gap |
|---|---|---|---|
| AR_post | 90.3 | 49.7 | **−40.6** |
| OR_post | 87.1 | 69.7 | −17.4 |
| AB_post | 90.0 | 78.6 | −11.4 |
| AB_pre+OR_post | 92.0 | 72.8 | −19.2 |

**Deciding to go back is easier than knowing where back is.** On `AR_post` the model picks
the right *kind* of action nine times out of ten and still gets the destination wrong in
half of those.

The two-turn goal-absent protocol separated two abilities cleanly:
`GA_absent_object/not_found` (give up after a failed search) is **100% for every model**,
while `GA_absent_object` turn 1 (go looking before concluding anything) is **0% for 2b and
4b** and 100% for 9b. Small models jump straight to `infeasible`. Asking for `infeasible`
in one turn would have hidden this entirely.

---

## 4. Long episodes (2273 subgoals, lengths 5–50)

| model | balanced | long | drop |
|---|---|---|---|
| qwen3.5:2b | 33.4 | 15.0 | −18.4 |
| qwen2.5vl:7b | 54.1 | 33.0 | −21.1 |
| qwen3.5:4b | 65.8 | 42.5 | −23.3 |
| qwen3.5:9b | 85.2 | **60.8** | −24.4 |

**Everything degrades by 18–24 points, and the biggest model degrades the most.** Scale
does not buy robustness to a longer conversation here — it buys a higher starting point.

Joint SR against how far back the referent sits:

| distance | 1 | 2 | 3–4 | 5–8 | 9–16 | 17+ |
|---|---|---|---|---|---|---|
| n | 208 | 471 | 152 | 173 | 156 | 140 |
| qwen3.5:9b | 75 | 53 | 55 | 44 | 40 | 41 |
| qwen3.5:4b | 34 | 25 | 24 | 25 | 15 | 17 |
| qwen2.5vl:7b | 22 | 27 | 21 | 26 | 24 | 23 |
| qwen3.5:2b | 0 | 1 | 0 | 0 | 0 | 0 |

9b halves from 75% to ~40% as the reference reaches further back and then flattens. 4b and
7b are near-flat and low — they are not failing *because* of distance, they are failing at
back references generally.

---

## 5. Cross-cutting findings

### 5.1 Generation beats size

`qwen3.5:4b` beats `qwen2.5vl:7b` on both datasets at **half the parameters** — 65.8 vs
54.1 balanced, 42.5 vs 33.0 long. And `qwen2.5vl:7b` is the **best model in the study** at
instructions that name their own target (99.3%). The gap is concentrated entirely in back
references (22.6% vs 49.3%), which is precisely the ability this benchmark exists to
measure.

### 5.2 The scale ladder is real but style-specific

Within one family and one condition: **33.4 → 65.8 → 85.2%** (2b → 4b → 9b). Almost all of
the gain is in back references (0.7 → 49.3 → 78.0), while "names its target" starts at
79.2% and creeps to 95.9%. Scale buys reference resolution, not instruction understanding.

### 5.3 The two probes are complementary, not a difficulty ladder

Paired subgoal by subgoal on the same 1308 items:

| model | referent | nav | both ok | referent only | **nav only** | both bad |
|---|---|---|---|---|---|---|
| qwen3.5:2b | 33.4 | 27.8 | 21.5 | 11.9 | 6.3 | 60.2 |
| qwen2.5vl:7b | 54.1 | **61.2** | 41.0 | 13.1 | **20.2** | 25.8 |
| qwen3.5:4b | 65.8 | 63.5 | 49.2 | 16.6 | 14.3 | 19.9 |
| qwen3.5:9b | 85.2 | 77.8 | 71.4 | 13.8 | 6.3 | 8.4 |

If the destination probe were the referent probe plus an extra step, "nav only" would be
near zero. It is 6–20%.

`qwen2.5vl:7b` actually scores **higher** on the harder-looking probe (61.2 vs 54.1),
and the per-style numbers say why: `AB_pre+OR_post` 2.5 → 48.1, `OR_post` 11.8 → 49.4,
`AR_post` 14.1 → 40.0. This model cannot produce *"the referent is instruction 3"* but can
copy that instruction's coordinates. Its low referent score was substantially a
**format** limitation, and the destination probe demonstrates that independently.

In the other direction, `qwen3.5:9b` loses 7.4 points going to the destination probe,
concentrated in `AR_post` and `OR_post` — knowing the referent and transferring its
coordinates are separable skills.

**Neither probe alone measures reference resolution cleanly.** One is contaminated by
answer formatting, the other by coordinate handling.

---

## 6. Caveats

### 6.1 Thinking models cannot be compared as they come

qwen3.5 at every size spends its whole token budget on `reasoning` and returns empty
`content`, at any budget tried up to 32768. The reported qwen3.5 numbers are all
`reasoning_effort=none`. `gemma4:26b` is thinking-enabled, so its 94.8% is **not** a
like-for-like entry in the ladder.

### 6.2 About 9–13% of new-object subgoals are unanswerable from text

A subgoal that introduces a new object whose *category* an earlier instruction already used
("Find the cardboard box." when instruction 1 was also "Find the cardboard box.") cannot be
told apart from a back reference without seeing the scene: **49/533 (9%)** on the balanced
set, **54/403 (13%)** on the long set. This puts an unreachable ceiling on the S / AB_pre /
AR_pre rows and explains why gemma4 scores *lower* there (88–95%) than on back references
(95–98%). It does not affect back-reference rows, where the answer is uniquely determined,
and it applies identically to every model, so model comparisons remain valid.

### 6.3 The headline number is prompt-sensitive on small models

Three answer schemas were tried on the referent probe. Asked for a bare instruction number,
qwen2.5vl:7b pointed at some earlier instruction for 231 of 534 fresh objects. Offered
`"new"` as one value of that same field, it answered `"new"` for 373 back references — 279
of which still named the true referent's category, i.e. it had resolved the reference and
only mislabelled it. The shipped schema (a label, plus a number only when the label is
`back_reference`) keeps the two decisions apart, but a feasibility number is only
comparable against navigation numbers taken with the **same prompt version**.

### 6.4 Single scene, single seed

One HM3D scene with ~19 usable goal objects, one generator seed, temperature 0. Category
diversity is limited (20 categories) and object reuse within an episode is heavy, which is
what makes §6.2 as large as it is.

---

## 7. Infrastructure notes

Three defects were found by the runs themselves and are worth knowing about before reading
any number:

- **A reasoning model that overruns `max_tokens` returns empty content with no exception.**
  At the client default of 4096, gemma4 returned nothing for 61 of 200 `all_at_once`
  episodes, and the scorer read that as 451 wrong answers — a reported 63.8% that was
  really a token budget. Records now carry `empty_response` separately from `parse_failed`,
  and the runner warns above the results table.
- **A reply cut off by the budget is not retryable** — the same prompt truncates the same
  way. Five *consecutive* cut-offs drop the model and write no results at all, so a dropped
  model cannot be mistaken for a bad score. Counting cumulatively instead discarded two
  otherwise healthy runs over a 0.3% blip rate.
- **`VLMClient` silently lost half its attributes** after a bad edit, so every request
  raised `AttributeError`, was retried five times 60s apart, and scored as wrong without a
  single request being sent. A model that had run 1308 queries in six minutes managed 36 in
  forty-six. Any run whose throughput drops by an order of magnitude should be checked for
  errors in the log before its numbers are read.

---

## 8. Reproducing

```bash
# referent probe (incremental), balanced set
OLLAMA_MODEL=qwen3.5:9b python run_refonbench_feasibility.py \
    -cf cfg/eval_refonbench_feasibility.yaml --workers 4 --reasoning-effort none

# destination probe
OLLAMA_MODEL=qwen3.5:9b python run_refonbench_feasibility_nav.py \
    -cf cfg/eval_refonbench_feasibility_nav.yaml --workers 4 --reasoning-effort none

# long-episode set
OLLAMA_MODEL=qwen3.5:9b python run_refonbench_feasibility.py \
    -cf cfg/eval_refonbench_feasibility_long.yaml --workers 4 --reasoning-effort none

# tables and charts
python scripts/compare_feasibility.py results/exp_feasibility_refonbench_nothink_* --plot cmp_even.png
python scripts/compare_feasibility.py --nav results/exp_feasibility_nav_refonbench_nothink_* --plot cmp_nav.png
python scripts/compare_probe_vs_nav.py --plot cmp_probe_vs_nav.png
```

Every wrong answer's full exchange — system prompt, user prompt, reply — is in
`feasibility_failures_<mode>.log` / `feasibility_nav_failures.log` in each results
directory, grouped so one `all_at_once` reply that got three subgoals wrong is one
transcript rather than three copies.

### Artifacts

| what | where |
|---|---|
| per-run results | `results/exp_feasibility_*/` (14 runs) |
| charts | `results/cmp_even.png`, `cmp_long.png`, `cmp_nav.png`, `cmp_probe_vs_nav.png` |
| datasets | `RefONEpisodeGenerator/out/refon_example_even_dataset/`, `refon_example_long_dataset/` |
| generator configs | `RefONEpisodeGenerator/configs/generator.example_even.json`, `generator.example_long.json` |
