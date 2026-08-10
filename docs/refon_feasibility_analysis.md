# Are RefON instructions solvable before the robot moves?

A RefON instruction is anaphoric: *"Find A1."*, *"Find the 2nd one again."*, *"Go back to
the previous one."* A navigation failure on one of those has two possible causes, and the
navigation score cannot tell them apart:

1. the agent never worked out **which object** the instruction meant, or
2. it knew, and failed to **find** it.

Everything below isolates (1). No habitat, no images, no navigation — the agent sees only
the instructions. The result is a ceiling: a subgoal that fails here cannot succeed in
navigation for reasons that have anything to do with perception or exploration.

Two probes. Open-weight models run through `ollama` on a single RTX 4090;
`claude-haiku-4-5` runs through the Anthropic API on the same prompts and scorer.

| probe | question | answer | runner |
|---|---|---|---|
| **referent** | which object does this instruction mean? | `new` / `back_reference` + instruction number / `no_object`, plus the object's category | `run_refonbench_feasibility.py` |
| **destination** | where does the robot go? | `explore` / `(x, y)` / `infeasible` | `run_refonbench_feasibility_nav.py` |

---

## 1. Method

### 1.1 The referent probe

Instructions are presented **incrementally** — at subgoal *i* the model sees instructions
1..*i* and is asked about *i*, which is exactly what a navigation run has at that point.
(An `all_at_once` mode showing the whole episode also exists; it is not part of the
comparison here — see the first note in §8 for why its early numbers were unusable.)

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

The four qwen models are the ladder: they run **non-thinking**, so that comparison is one
condition throughout (see §7.1 for why). The other two are reference points outside it.

| model | size | note |
|---|---|---|
| `qwen3.5:2b` | 2B | |
| `qwen3.5:4b` | 4B | |
| `qwen2.5vl:7b` | 7B | previous generation |
| `qwen3.5:9b` | 9B | |
| `gemma4:26b-a4b-it-qat` | 26B (4B active) | thinking on — not directly comparable |
| `claude-haiku-4-5` | undisclosed | API model, thinking disabled |

Anthropic publishes no parameter count for any Claude model, so `claude-haiku-4-5` cannot
be placed on the size axis and is excluded from the scale-ladder claims in §5.2. It is
here as an upper reference point: what the same prompts score when the reference-resolution
stage is not size-constrained.

`qwen3.5:0.8b` was excluded: it spends the whole token budget thinking and returns empty
content (16384 completion tokens, 66k characters of `reasoning`, 240s a call).
`qwen3-vl:8b` was excluded from the non-thinking comparison because it cannot be put in
non-thinking mode — `think=False`, `/no_think`, `enable_thinking=False` and
`reasoning_effort=none` are all ignored (5089 tokens as-is vs 5701 with the flag).

### 1.5 Reporting styles

`AB_pre` and `AR_pre` are folded into `S` throughout, as `scripts/summarize_refonbench.py`
does. All three name their own target and the answer has the same shape for each —
`AB_pre` only appends an alias binding ("Let's call it A1."), which is not something to
resolve. That makes `S` a 534-sample row on the balanced set and 958 on the long one.

`AB_pre+OR_post` stays separate: it binds an alias *and* refers back, so folding it in
would put an anaphoric case in the row that is supposed to be free of them. Pass
`--merge-roles` to the comparison scripts to reproduce this grouping, or drop it to see
all ten styles.

---

## 2. Referent probe, balanced set (1308 subgoals)

`S` here folds in `AB_pre` and `AR_pre` (§1.5).

| style | n | 2b | 4b | 7b | 9b | gemma4 26b† | haiku 4.5‡ |
|---|---|---|---|---|---|---|---|
| **S** | 534 | 79.2 | 90.4 | **99.3** | 95.9 | 91.8 | 91.4 |
| AB_post | 220 | 0.0 | 71.4 | 53.2 | 85.9 | **97.7** | 95.5 |
| AR_post | 185 | 2.7 | 43.8 | 14.1 | 64.3 | **96.8** | 88.1 |
| OR_post | 178 | 0.0 | 52.2 | 11.8 | 84.8 | 94.9 | **96.1** |
| AB_pre+OR_post | 162 | 0.0 | 22.2 | 2.5 | 75.3 | **97.5** | 94.4 |
| GA_absent_object | 9 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| GA_invalid_ordinal | 13 | 0.0 | 15.4 | 0.0 | 84.6 | **100.0** | **100.0** |
| GA_unbound_alias | 7 | 0.0 | 0.0 | 0.0 | 28.6 | **100.0** | **100.0** |
| **ALL** | 1308 | **33.4** | **65.8** | **54.1** | **85.2** | **94.8** | **92.8** |
| *back-reference* | 745 | 0.7 | 49.3 | 22.6 | 78.0 | **96.8** | 93.6 |

† thinking enabled; shown for reference, not as part of the ladder.
‡ API model, no published size; see §1.4.

**The first row against the last is the whole story.** Every model is decent at
instructions that name their own target, and they separate almost entirely on the ones
that do not: 79→96 across the ladder on `S`, 0.7→78 on back references.

Both reference points invert that pattern: gemma4 and haiku each score **lower** on `S`
(91.8, 91.4) than on back references (96.8, 93.6). That is not a regression — §7.2 shows
why the `S` row has a ceiling below 100% that no amount of capability can pass.

---

## 3. Destination probe, balanced set (1317 turns)

| | 2b | 4b | 7b | 9b | haiku 4.5 |
|---|---|---|---|---|---|
| action SR | 28.9 | 85.0 | 71.1 | 90.2 | **95.2** |
| coord SR | 43.1 | 68.2 | 65.9 | 78.8 | **90.7** |
| **joint SR** | **28.3** | **63.8** | **60.7** | **77.9** | **90.6** |
| *S* | 63.9 | 80.5 | 85.0 | 91.0 | **91.9** |
| *back-reference* | 0.4 | 51.1 | 45.5 | 68.1 | **89.1** |

Per style, action vs joint:

| style | n | 9b action | 9b joint | 9b gap | haiku action | haiku joint | haiku gap |
|---|---|---|---|---|---|---|---|
| S | 534 | 91.0 | 91.0 | 0.0 | 91.9 | 91.9 | 0.0 |
| AB_post | 220 | 90.0 | 78.6 | −11.4 | 100.0 | 100.0 | 0.0 |
| **AR_post** | 185 | 90.3 | 49.7 | **−40.6** | 97.3 | 75.7 | **−21.6** |
| OR_post | 178 | 87.1 | 69.7 | −17.4 | 94.4 | 89.3 | −5.1 |
| AB_pre+OR_post | 162 | 92.0 | 72.8 | −19.2 | 96.9 | 89.5 | −7.4 |

`S` has no gap by construction — `explore` is the whole answer, there is no coordinate to
get wrong. Every back-reference row does, for every model except haiku on `AB_post`.

**Deciding to go back is easier than knowing where back is.** On `AR_post` the model picks
the right *kind* of action nine times out of ten and still gets the destination wrong in
half of those. Haiku narrows the gap everywhere but does not close it, and `AR_post`
("go back to the previous one", "find the one before that") remains its worst style by a
wide margin — 75.7% against 89–100% for every other style. Relative anaphora is the last
thing to fall into place, not alias binding or ordinals.

The two-turn goal-absent protocol separated two abilities cleanly:
`GA_absent_object/not_found` (give up after a failed search) is **100% for every model**,
while `GA_absent_object` turn 1 (go looking before concluding anything) is **0% for 2b and
4b** and 100% for 9b and haiku. Small models jump straight to `infeasible`. Asking for
`infeasible` in one turn would have hidden this entirely.

---

## 4. Long episodes (2273 subgoals, lengths 5–50)

| model | balanced | long | drop |
|---|---|---|---|
| qwen3.5:2b | 33.4 | 15.0 | −18.4 |
| qwen2.5vl:7b | 54.1 | 33.0 | −21.1 |
| qwen3.5:4b | 65.8 | 42.5 | −23.3 |
| qwen3.5:9b | 85.2 | 60.8 | −24.4 |
| claude-haiku-4-5 | 92.8 | **82.5** | **−10.3** |

**Within the open-weight ladder everything degrades by 18–24 points, and the biggest model
degrades the most.** Scale inside that ladder does not buy robustness to a longer
conversation — it buys a higher starting point.

**Haiku breaks that pattern**: it loses 10.3 points, less than half of what the 9b loses,
from a starting point 7.6 points higher. So the degradation is not an inherent property of
the task at length — it is a property of these models. Something separates the two groups
that the size axis does not capture, which is one more reason §5.2's ladder claim is
scoped to the qwen family.

Per style:

| style | n | 2b | 4b | 7b | 9b | haiku 4.5 |
|---|---|---|---|---|---|---|
| **S** | 958 | 34.6 | 66.4 | 44.2 | 71.0 | **85.1** |
| AB_post | 379 | 0.0 | 44.9 | 45.4 | 70.4 | **89.7** |
| AR_post | 325 | 1.8 | 21.2 | 25.2 | 48.3 | **61.8** |
| OR_post | 332 | 0.0 | 18.7 | 12.0 | 50.0 | **84.0** |
| AB_pre+OR_post | 264 | 0.0 | 6.4 | 10.6 | 36.7 | **86.0** |
| **ALL** | 2273 | **15.0** | **42.5** | **33.0** | **60.8** | **82.5** |
| *back-reference* | 1300 | 0.5 | 24.5 | 24.8 | 52.8 | **80.5** |

Note that `S` falls too — 95.9 → 71.0 for 9b, 91.4 → 85.1 for haiku — even though a direct
instruction carries everything it needs. Length hurts the easy cases as well, not only the
anaphoric ones, for every model.

Joint SR against how far back the referent sits:

| distance | 1 | 2 | 3–4 | 5–8 | 9–16 | 17+ |
|---|---|---|---|---|---|---|
| n | 208 | 471 | 152 | 173 | 156 | 140 |
| claude-haiku-4-5 | 92 | 71 | 86 | 82 | 83 | 87 |
| qwen3.5:9b | 75 | 53 | 55 | 44 | 40 | 41 |
| qwen3.5:4b | 34 | 25 | 24 | 25 | 15 | 17 |
| qwen2.5vl:7b | 22 | 27 | 21 | 26 | 24 | 23 |
| qwen3.5:2b | 0 | 1 | 0 | 0 | 0 | 0 |

9b halves from 75% to ~40% as the reference reaches further back and then flattens. 4b and
7b are near-flat and low — they are not failing *because* of distance, they are failing at
back references generally.

**Haiku is flat and high**: 87% at distance 17+ against 92% at distance 1. Reaching 17
instructions back costs it essentially nothing, which says the long-episode drop above is
not a retrieval-range problem. Its one dip is the distance-2 bin (71%), and that bin is
dominated by `AR_post` — "the one before that" — the same style that is its worst in
both other experiments. What it loses is not *distant* references but *relative* ones,
and relative references happen to be short-range.

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
the gain is in back references (0.7 → 49.3 → 78.0), while `S` starts at 79.2% and creeps
to 95.9%. Scale buys reference resolution, not instruction understanding.

This claim is scoped to the qwen family. `gemma4:26b-a4b` and `claude-haiku-4-5` are not
points on it — the first has thinking enabled, the second has no published size — and §4
shows the ladder's own trend (bigger degrades more at length) does not extend past it.

### 5.3 The two probes are complementary, not a difficulty ladder

Paired subgoal by subgoal on the same 1308 items:

| model | referent | nav | both ok | referent only | **nav only** | both bad |
|---|---|---|---|---|---|---|
| qwen3.5:2b | 33.4 | 27.8 | 21.5 | 11.9 | 6.3 | 60.2 |
| qwen2.5vl:7b | 54.1 | **61.2** | 41.0 | 13.1 | **20.2** | 25.8 |
| qwen3.5:4b | 65.8 | 63.5 | 49.2 | 16.6 | 14.3 | 19.9 |
| qwen3.5:9b | 85.2 | 77.8 | 71.4 | 13.8 | 6.3 | 8.4 |
| claude-haiku-4-5 | 92.8 | 90.5 | 87.8 | 5.0 | **2.8** | 4.4 |

If the destination probe were the referent probe plus an extra step, "nav only" would be
near zero. It is 6–20% across the ladder.

Haiku is the one model where it nearly is (2.8%), and that is the shape a genuine
difficulty ladder would have. The disagreement between the two probes is therefore mostly
a *capability* artifact, not a property of the probes: as models get better the two
questions converge on the same answer. It does not vanish, though — 5.0% "probe only" plus
2.8% "nav only" means 7.8% of subgoals still separate the two even at 90%+.

`qwen2.5vl:7b` actually scores **higher** on the harder-looking probe (61.2 vs 54.1), and
the per-style numbers say why. It moves the *wrong* way on `S` (99.3 → 85.0) and hugely the
right way on every back reference: `AB_pre+OR_post` 2.5 → 48.1, `OR_post` 11.8 → 49.4,
`AR_post` 14.1 → 40.0. This model cannot produce *"the referent is instruction 3"* but can
copy that instruction's coordinates. Its low referent score was substantially a
**format** limitation, and the destination probe demonstrates that independently.

In the other direction, `qwen3.5:9b` loses 7.4 points going to the destination probe,
concentrated in `AR_post` and `OR_post` — knowing the referent and transferring its
coordinates are separable skills.

**Neither probe alone measures reference resolution cleanly.** One is contaminated by
answer formatting, the other by coordinate handling.

---

## 6. One stage or two?

Everything above measures reference resolution *in isolation*, which is only interesting if
it bears on how the system should be built. It does. The current RefON evaluation runs
**one** VLM that infers the goal and navigates inside the same prompt. The alternative is to
split it: resolve the reference from text alone, hand the resulting goal to 3D-Mem, and let
3D-Mem do plain object-goal navigation.

```
1-step   instruction ─────────────────► VLM (resolve + navigate) ──► success?
2-step   instruction ──► resolver (text only) ──► "find: <goal>" ──► 3D-Mem (object-nav) ──► success?
                              = S1                                        = S2
```

### 6.1 What the 1-step pipeline actually scores

From `results/exp_eval_refonbench_default_qwen3_vl_30b` (`qwen3-vl:30b`,
`success_distance` 1.0 m). **37 subtasks were scored — the run is partial**, so every
number here carries roughly ±11 points.

| | n | SR |
|---|---|---|
| **overall** | 37 | **13.5%** (SPL 0.100) |
| merged-`S` — no reference to resolve, i.e. plain object-nav | 15 | **20.0%** |
| referential styles | 22 | **9.1%** |

`success_by_snapshot` is 0.0% throughout; only `success_by_distance` registers anything.

The two rows give the **embodied reference-resolution factor** directly:
9.1 / 20.0 = **0.455**. Attaching a referential expression to an instruction that the same
pipeline solves 20% of the time cuts it to 9%. Section 2 puts the same models' text-only
figure at **0.93–0.95**.

### 6.2 3D-Mem's published object-nav rate

From the 3D-Mem paper (Table 3, GOAT-Bench `val_unseen`, GPT-4o):

| goal type | SR | SPL |
|---|---|---|
| **object category** — what S2 would be | **79.2** | 55.8 |
| language | 61.9 | 46.0 |
| image | 65.2 | 44.2 |
| overall | 69.1 | 48.9 |

Baselines from the same table: Explore-EQA 55.0/37.9, ConceptGraph w/ Frontier Snapshots
61.5/45.3, 3D-Mem w/o memory 58.6/38.5. A-EQA (Table 1): LLM-Match 52.6, SPL 42.0.

### 6.3 The product

`SR₂ = S1 × S2`, with S2 taken two ways: **A** = the paper's 79.2% (different model,
different benchmark — an optimistic ceiling), **B** = the 20.0% merged-`S` rate measured
above (same model, same scene, same success criterion — pessimistic but internally
consistent).

| resolver | S1 | 2-step, anchor A | 2-step, anchor B |
|---|---|---|---|
| gemma4:26b-a4b | 94.8 | 75.1 | 19.0 |
| claude-haiku-4-5 | 92.8 | 73.5 | 18.6 |
| qwen3.5:9b | 85.2 | 67.5 | 17.0 |
| qwen3.5:4b | 65.8 | 52.1 | 13.2 |
| qwen2.5vl:7b | 54.1 | 42.8 | 10.8 |
| qwen3.5:2b | 33.4 | 26.5 | 6.7 |
| *1-step, measured* | — | *13.5* | *13.5* |

Break-even S1 is 13.5/79.2 = **17.0%** under A and 13.5/20.0 = **67.5%** under B.

### 6.4 Why this does not reverse for small on-robot models

Reading the table as "a 2B resolver scores 6.7%, so 1-step wins on small hardware" is a
mistake, and it is the table's fault: it holds the 1-step number fixed at a 30B measurement
while varying the resolver. Fix the model instead:

```
1-step(M) = objnav(M) × r_emb(M)      r_emb  = reference resolution while also perceiving
2-step(M) = objnav(M) × s_text(M)     s_text = reference resolution from text alone
```

`objnav(M)` cancels. The comparison reduces to **s_text vs r_emb for the same model** — a
question with no size term in it. Text-only resolution is a strict subproblem of the joint
task, so `s_text ≥ r_emb` should hold at every scale, and the measured pair (0.93 vs 0.455)
is a wide margin. There is no reason to expect a smaller model to be *better* at doing two
things at once.

The real consequence for robotics is the opposite of a warning: **S1 is text-only.** No
images, no 3D memory, one call per subgoal, a prompt of a few thousand tokens. It does not
have to run on the robot. The split decouples the size requirement — a 2–4B policy
on-device, a larger resolver off-board — which is exactly what 1-step forbids.

### 6.5 What the product overstates

- **Category is not instance.** The paper's 79.2% counts reaching *any* object of the right
  category; RefON scores distance to a *specific* one. **48.9% of subgoals in the balanced
  set are back references** to a particular earlier instance. A category-only handoff loses
  that, so anchor A is optimistic on half the set. Carrying instance identity across the
  boundary instead means S2 is no longer plain object-nav.
- **Independence.** `S1 × S2` assumes resolution failures and navigation failures are
  uncorrelated. Harder instructions plausibly go with harder-to-find objects, which would
  push the true value below the product.
- **Sample size and model mismatch.** 13.5% is n=37; 20.0% is n=15 (±20 points). And 13.5%
  is `qwen3-vl:30b`, 79.2% is GPT-4o, the S1 column is six other models.
- **Errors become irreversible.** A 1-step agent can in principle revise a misreading once
  it sees the scene; a 2-step one cannot. The product does not model this, and it favours
  2-step. (The 9.1% referential row suggests the revision channel barely works today.)

### 6.6 Two things a text-only stage cannot do

**Confirming an object is absent.** `GA_absent_object` ("Find the chandelier." in a scene
with none) is not answerable from text at any capability level — absence is a *result of
exploration*. The resolver can only emit "go find a chandelier"; the conclusion has to come
back from navigation. §1.2's two-turn protocol is precisely that admission, and the
referent probe sidesteps the problem by scoring `GA_absent_object` as `new` + category
(§1.1) — meaning the 92.8% headline **does not contain this failure mode at all**.

**Trajectory-relative references.** An instruction like *"find another toilet, not the
first one you found"* defines its exclusion set over the agent's **actual path**, not over
the instruction text. A text-only resolver cannot know which toilet was reached first.

RefON does not currently contain this case. Every referential form in the generated set is
*instruction*-relative:

```
OR_post   "Find the 1st one again."
AR_post   "Go back to the previous one."  /  "Find the one before that."
AB_post   "Find A1."
```

The ordinal indexes the instruction sequence and the generator binds it to a ground-truth
object, so counting instructions is sufficient — which is part of why the text-only scores
are as high as they are. A trajectory-relative style (call it `TR_post`) would be the
honest test of the 2-step split, and adding one is the obvious next dataset change.

### 6.7 Conclusion

The useful architecture is not 1-step or 2-step but **2-step with a thin state channel**:
the resolver gets the instruction history *plus* a compact text summary of what the agent
actually visited (object id, category, room, visit order) and any `not_found` outcome. That
stays text-only, stays cheap, stays off-board — and it is what
`run_refonbench_feasibility_nav.py` already models by disclosing coordinates as subgoals
reach them.

---

## 7. Caveats

### 7.1 Thinking models cannot be compared as they come

qwen3.5 at every size spends its whole token budget on `reasoning` and returns empty
`content`, at any budget tried up to 32768. The reported qwen3.5 numbers are all
`reasoning_effort=none`. `gemma4:26b` is thinking-enabled, so its 94.8% is **not** a
like-for-like entry in the ladder. `claude-haiku-4-5` runs with `thinking` disabled, which
matches the qwen condition — but it is an API model of undisclosed size, so it is outside
the ladder for a different reason (§1.4).

### 7.2 About 9–13% of new-object subgoals are unanswerable from text

A subgoal that introduces a new object whose *category* an earlier instruction already used
("Find the cardboard box." when instruction 1 was also "Find the cardboard box.") cannot be
told apart from a back reference without seeing the scene: **49/533 (9%)** on the balanced
set, **54/403 (13%)** on the long set. This puts an unreachable ceiling on the `S` row and
explains why the two strongest models both score *lower* there than on back references —
gemma4 91.8 vs 96.8, haiku 91.4 vs 93.6. It does not affect back-reference rows, where the
answer is uniquely determined, and it applies identically to every model, so model
comparisons remain valid.

### 7.3 The headline number is prompt-sensitive on small models

Three answer schemas were tried on the referent probe. Asked for a bare instruction number,
qwen2.5vl:7b pointed at some earlier instruction for 231 of 534 fresh objects. Offered
`"new"` as one value of that same field, it answered `"new"` for 373 back references — 279
of which still named the true referent's category, i.e. it had resolved the reference and
only mislabelled it. The shipped schema (a label, plus a number only when the label is
`back_reference`) keeps the two decisions apart, but a feasibility number is only
comparable against navigation numbers taken with the **same prompt version**.

### 7.4 Single scene, single seed

One HM3D scene with ~19 usable goal objects, one generator seed, temperature 0. Category
diversity is limited (20 categories) and object reuse within an episode is heavy, which is
what makes §7.2 as large as it is.

---

## 8. Infrastructure notes

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

## 9. Reproducing

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

# same three probes against the Anthropic API (key read from ANTHROPIC_API_KEY)
VLM_PROVIDER=anthropic ANTHROPIC_MODEL=claude-haiku-4-5 \
    python run_refonbench_feasibility.py \
    -cf cfg/eval_refonbench_feasibility.yaml --workers 4 --reasoning-effort none

# tables and charts (--merge-roles folds AB_pre / AR_pre into S, as reported here)
python scripts/compare_feasibility.py --merge-roles \
    results/exp_feasibility_refonbench_* --plot cmp_even.png
python scripts/compare_feasibility.py --merge-roles --nav \
    results/exp_feasibility_nav_refonbench_* --plot cmp_nav.png
python scripts/compare_probe_vs_nav.py --merge-roles --plot cmp_probe_vs_nav.png

# §6.1 — the 1-step pipeline's own score, from the navigation run's pickles
python3 -c "import pickle; t=pickle.load(open(
    'results/exp_eval_refonbench_default_qwen3_vl_30b/success_by_task_0.0_1.0_0.pkl','rb'));
    print({k: (len(v), sum(v)) for k, v in t.items()})"
```

`compare_probe_vs_nav.py` defaults to the `*_nothink_*` globs; pass `--probe-glob` /
`--nav-glob` to include the API runs. `compare_feasibility.py --plot` needs `--dataset` to
point at the shard directory itself
(`RefONEpisodeGenerator/out/refon_example_long_dataset/v1/val/content/`), not the set root,
or the reference-distance panel comes out empty.

Every wrong answer's full exchange — system prompt, user prompt, reply — is in
`feasibility_failures_<mode>.log` / `feasibility_nav_failures.log` in each results
directory, grouped so one `all_at_once` reply that got three subgoals wrong is one
transcript rather than three copies.

### Artifacts

| what | where |
|---|---|
| per-run results | `results/exp_feasibility_*/` (17 runs) |
| 1-step pipeline run (§6.1) | `results/exp_eval_refonbench_default_qwen3_vl_30b/` |
| charts | `results/cmp_even.png`, `cmp_long.png`, `cmp_nav.png`, `cmp_probe_vs_nav.png`, `cmp_even_haiku.png` |
| datasets | `RefONEpisodeGenerator/out/refon_example_even_dataset/`, `refon_example_long_dataset/` |
| generator configs | `RefONEpisodeGenerator/configs/generator.example_even.json`, `generator.example_long.json` |
| 3D-Mem paper figures (§6.2) | arXiv 2411.17735, Tables 1 and 3 |
