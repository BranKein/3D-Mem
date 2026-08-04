# Referential Object-Goal Navigation — Episode Generator Specification

This document specifies the task, its referencing mechanisms, the formal
generation rules (a one-counter pushdown automaton), and worked examples.
It is intended as an implementation reference for building an **episode
generator** that produces valid referential multi-goal navigation task
sequences.

> **Note on this revision.** Ordinal Referencing (OR) is modeled as a *free
> back-reference* keyed to instruction (visit) order, not as a balanced
> open/close binding. See Section 2.2 and Section 4. This simplifies the
> automaton: only Alias Binding remains a balanced (counter) mechanism, and
> the Anaphoric block no longer needs a stack.

---

## 1. Task Overview

### 1.1 Base task: Multi-Goal Object Navigation

An agent is placed in a scene and given a sequence of navigation goals. Each
goal instructs the agent to navigate to a particular target object (e.g.,
"Find the chair in the bedroom"). The agent completes the episode by reaching
each goal in order.

### 1.2 Extension: Referential re-targeting

This task extends multi-goal navigation by allowing later goals to **re-refer
to previously visited objects** instead of describing them afresh. Rather than
re-specifying an object by category and location, a later instruction may point
back to an earlier target using a *referring expression*.

This tests whether an agent can:
- maintain an **episodic memory** of previously visited objects,
- **resolve referring expressions** against that memory,
- and re-navigate to the correct previously-seen object.

### 1.3 Terminology

- **Target object** `O_i`: a distinct target object in the scene.
- **Goal / instruction**: a single navigation command in the sequence. Every
  instruction navigates to exactly one object.
- **Visit order**: the ordered list of objects navigated to, in instruction
  order, **including repeats**. The k-th entry is the object targeted by the
  k-th instruction.
- **Introduction (pre)**: an instruction that establishes a referent which can
  be pointed back to later (alias binding, anaphoric antecedent).
- **Back-reference (post)**: an instruction that re-refers to a previously
  visited object.

---

## 2. Referencing Mechanisms

There are three mechanisms. Their structural nature differs:

- **Alias Binding (AB)** — balanced: an explicit `AB_pre` introduces a name
  that must later be closed by a matching `AB_post`.
- **Ordinal Referencing (OR)** — *unbalanced, free*: there is no introduction
  token; every instruction implicitly gets a visit-order index, and `OR_post`
  freely references any past index.
- **Anaphoric Referencing (AR)** — balanced with a fixed gap: `AR_pre`
  introduces an antecedent, exactly one instruction intervenes, then `AR_post`
  refers back.

### 2.1 Alias Binding (AB)

| Role | Instruction template | Meaning |
|------|----------------------|---------|
| `AB_pre`  | `Find O_i. Let's say it as A_i.` | navigate to `O_i`, bind alias `A_i` to it |
| `AB_post` | `Find A_i.` | re-navigate to the object bound to alias `A_i` |

- `A_i` is the alias bound to object `O_i`.
- Resolution requires recalling **which object the name was assigned to**.
- AB is **balanced**: every `AB_pre` must be matched by exactly one later
  `AB_post`. Multiple aliases may be open at once; closing is **LIFO** by
  default (see Section 5.3).

### 2.2 Ordinal Referencing (OR) — free back-reference

There is **no `OR_pre`**. Every instruction navigates to an object and thereby
occupies the next visit-order slot automatically. `OR_post` references one of
these slots by its index.

| Role | Instruction template | Meaning |
|------|----------------------|---------|
| `OR_post` | `Find the k-th one again.` | re-navigate to `visit_order[k-1]`, the object targeted by the k-th instruction |

- **Index semantics**: `k` is the **instruction (visit) order index**, counting
  every navigation including repeats. If the agent navigated to the same object
  at instructions 1 and 4, those are visit slots 1 and 4 independently.
- **Validity**: an `OR_post` with index `k` is valid only if `1 ≤ k ≤ v`, where
  `v` is the number of instructions **before** this one. Hence `OR_post` cannot
  be the first instruction.
- **Structural effect**: none. For the automaton, `OR_post` behaves exactly
  like a plain goal `S` (it navigates but opens/closes no binding). Its only
  extra requirement is the index validity above.

### 2.3 Anaphoric Referencing (AR)

| Role | Instruction template | Meaning |
|------|----------------------|---------|
| `AR_pre`  | `Find O_i.` | navigate to `O_i`; mark `O_i` as the anaphoric antecedent |
| `AR_post` | `Go back to the previous one.` / `Find the one before that.` | re-navigate to the antecedent introduced by the matching `AR_pre` |

- Exactly **one** instruction must appear between `AR_pre` and `AR_post`.
- The intervening instruction may carry any role **except** `AR_pre`/`AR_post`
  (so AR blocks cannot nest and cannot be empty).
- **Resolution**: `AR_post` resolves to the **`AR_pre` target object**, i.e.,
  the antecedent — *not* the intervening instruction's object.

#### Resolution illustrated
With `[A, AR_pre(B), C, AR_post]` the agent visits A, then B, then C, and is now
standing at C. `AR_post` ("go back to the previous one") refers to **B** — the
object found *before* the most recent one (C). `AR_pre` is what marks B as that
antecedent; the single intervening instruction C is the "most recent" that the
back-reference steps over.

#### Phrasing variants for `AR_post`
- "Go back to the previous one."
- "Find the one before that."
- "Go back to the one before the last."

> **Design note (phrasing).** Avoid phrasings like "the one you just found" or
> "the last one": with one intervening instruction these read as the
> *intervening* object (C), not the antecedent (B). Use phrasings that clearly
> denote the object found *before* the most recent one.

---

## 3. Tokens and Roles

### 3.1 Role set

Every instruction is annotated with a **role set** `ρ`. The roles are:

```
AB_pre, AB_post, OR_post, AR_pre, AR_post
```

(There is **no** `OR_pre`.) A plain goal with no referencing role has
`ρ = ∅` (denoted `S`).

A useful way to read a role set: it picks one **reference mode** (how this
instruction's target is identified) and zero or more **forward markers** (what
this target sets up for the future).

- Reference modes (mutually exclusive): direct/`S`, `AB_post` (by alias),
  `OR_post` (by ordinal), `AR_post` (by anaphora).
- Forward markers: `AB_pre` (bind a new alias to this target).
  `AR_pre` is treated as a standalone introduction (direct nav + mark
  antecedent).

### 3.2 Valid role sets

The complete set of valid role sets (7 total):

```
∅                      plain goal S (direct navigation)
{AB_pre}               direct nav + bind a new alias
{AB_post}              navigate by alias
{OR_post}              navigate by visit-order ordinal
{AB_pre, OR_post}      navigate by ordinal + bind a new alias to that object
{AR_pre}               direct nav + mark as anaphoric antecedent (standalone)
{AR_post}              navigate by anaphora (standalone)
```

Constraints that produce this list:
1. **One reference mode per instruction.** Do not combine two
   "navigate-by-X" roles (e.g., no `{AB_post, OR_post}`).
2. **No self-conflict.** `{AB_pre, AB_post} ⊄ ρ`.
3. **AR exclusivity.** `AR_pre` and `AR_post` each appear alone. This is a
   structural consequence of the AR mode transitions (Section 4.3): in the
   "expecting `AR_post`" mode only `AR_post` is accepted, and `AR_pre` only
   transitions out of the normal mode, so neither can be combined with other
   roles.

### 3.3 Role processing order (for multi-role tokens)

Only `{AB_pre, OR_post}` is multi-role. Process as: resolve the target via
`OR_post` (navigation), then apply `AB_pre` (bind alias to that target). For
the automaton's counter the order is irrelevant since `OR_post` is a no-op;
`AB_pre` increments the open-alias count.

---

## 4. Generation Rules (One-Counter Pushdown Automaton)

A sequence is valid iff the automaton, starting from the initial state,
consumes all tokens and ends in the accepting state.

### 4.1 State

```
state = (n_AB, ar_mode)
```

- `n_AB ∈ ℕ₀`: number of currently **open** alias bindings (introduced by
  `AB_pre`, not yet closed by `AB_post`). This counter is the automaton's
  "pushdown" content — equivalently, a stack holding one symbol per open alias.
- `ar_mode ∈ { NONE, EXPECT_ONE, EXPECT_POST }`: tracks the anaphoric block.
  - `NONE`: not inside an AR block.
  - `EXPECT_ONE`: `AR_pre` seen; awaiting exactly one non-AR instruction.
  - `EXPECT_POST`: the single intervening instruction seen; awaiting `AR_post`.

The generator additionally maintains, outside the automaton state, a
`visit_order` list and an `alias_map` (Section 5). These are needed for object
assignment but not for structural validity (except the `OR_post` index
side-condition).

- **Initial state:** `(0, NONE)`
- **Accepting state:** `(0, NONE)` — all aliases closed and not mid-AR-block.

A counter-based `n_AB` is required so that multiple alias bindings can be open
simultaneously (LIFO).

### 4.2 No stack for AR

Earlier revisions pushed/popped the binding context on `AR_pre`/`AR_post`.
That is no longer needed: because the single intervening instruction's effect
on `n_AB` simply persists (it is not discarded), the AR block has **no net
effect** on `n_AB` beyond what the intervening instruction does. AR is therefore
a purely structural (finite-mode) constraint, and the only unbounded memory is
the alias counter `n_AB`.

### 4.3 Transitions

`AB_pre` ⇒ `n_AB += 1`; `AB_post` ⇒ `n_AB -= 1` (requires `n_AB > 0`);
`S`, `OR_post` ⇒ `n_AB` unchanged; `{AB_pre, OR_post}` ⇒ `n_AB += 1`.

#### From `ar_mode = NONE`

| Token `ρ` | Precondition | Effect |
|-----------|--------------|--------|
| `∅` (S)            | —                      | `n_AB` unchanged |
| `{OR_post}`        | `v ≥ 1` (prior visits) | `n_AB` unchanged |
| `{AB_pre}`         | —                      | `n_AB += 1` |
| `{AB_pre, OR_post}`| `v ≥ 1`                | `n_AB += 1` |
| `{AB_post}`        | `n_AB > 0`             | `n_AB -= 1` |
| `{AR_pre}`         | —                      | `ar_mode → EXPECT_ONE` |
| `{AR_post}`        | — (blocked)            | rejected (no open AR) |

#### From `ar_mode = EXPECT_ONE` (the single intervening instruction)

| Token `ρ` | Precondition | Effect |
|-----------|--------------|--------|
| any non-AR `ρ`: `∅`, `{OR_post}`, `{AB_pre}`, `{AB_pre,OR_post}`, `{AB_post}` | `OR_post` needs `v ≥ 1`; `AB_post` needs `n_AB > 0` | apply the `n_AB` effect, then `ar_mode → EXPECT_POST` |
| `{AR_pre}` or `{AR_post}` | — (blocked) | rejected (no AR nesting; block not yet closeable) |

#### From `ar_mode = EXPECT_POST`

| Token `ρ` | Precondition | Effect |
|-----------|--------------|--------|
| `{AR_post}` | — | `ar_mode → NONE` |
| anything else | — (blocked) | rejected |

`v` = number of instructions already emitted (every instruction is one
navigation, so `v` increments by 1 per token).

### 4.4 Constraints summary

1. **Balanced aliases** — every `AB_pre` matched by exactly one later
   `AB_post`. Enforced by acceptance only at `n_AB = 0`.
2. **LIFO alias closing** — `AB_post` closes the most recently opened alias
   (default; see Section 5.3).
3. **OR is free** — `OR_post` references any visit-order index `k ≤ v`; it opens
   nothing and need not be "matched."
4. **AR arity** — exactly one instruction between `AR_pre` and `AR_post`.
5. **No AR nesting** — an AR block cannot contain `AR_pre`/`AR_post`.
6. **AR resolution** — `AR_post` resolves to the `AR_pre` target (antecedent).

---

## 5. Object Assignment

The role sequence defines structure; concrete episodes also require assigning
**target objects** to instructions.

### 5.1 Which roles choose a (possibly new) object vs. resolve an existing one

| Role | Target object |
|------|---------------|
| `S`        | a chosen object (new or any existing), navigated by description |
| `AB_pre`   | a chosen object; an alias is bound to it |
| `AR_pre`   | a chosen object; becomes the anaphoric antecedent |
| `AB_post`  | **resolved**: the object of the alias being closed |
| `OR_post`  | **resolved**: `visit_order[k-1]` for a chosen valid `k` |
| `AR_post`  | **resolved**: the antecedent of the enclosing AR block |
| `{AB_pre, OR_post}` | **resolved** via `OR_post` (`visit_order[k-1]`); an alias is then bound to that object |

### 5.2 Bookkeeping maintained during generation

- **`visit_order`**: list of target objects in instruction order, with repeats.
  Append the resolved/chosen object after every instruction.
- **`alias_map`**: LIFO stack of `(A_i → O_i)` for currently open aliases.
  Push on `AB_pre` / `{AB_pre, OR_post}`; pop on `AB_post`.
- **`antecedent`**: the `AR_pre` target object; consumed by the matching
  `AR_post`.

### 5.3 Resolution semantics for back-references

- `AB_post` → the object of the alias being closed. Default **LIFO**: closes
  the most recently opened alias. (The automaton's counter does not enforce a
  particular alias; if non-LIFO closing is desired, track `alias_map`
  explicitly and choose among open aliases.)
- `OR_post` ("the k-th one") → `visit_order[k-1]`, for any `k` with
  `1 ≤ k ≤ v`. The generator chooses `k`.
- `AR_post` → the `antecedent` (the `AR_pre` target).

### 5.4 Object budget

A target-object budget (e.g., "at most 3 distinct objects") constrains how many
distinct `O_i` may appear across the episode. New-object choices (for `S`,
`AB_pre`, `AR_pre`) must respect the budget; back-references reuse existing
objects and do not consume budget.

---

## 6. Worked Examples

Each example lists the instruction, its role set, and the automaton state
`(n_AB, ar_mode)` after the step. `O_i` are objects; `A_i` are aliases.

### Example 1 — Plain goals `[S, S, S]`

```
1. Find O_1.                       ∅            (0, NONE)
2. Find O_2.                       ∅            (0, NONE)
3. Find O_3.                       ∅            (0, NONE)   ✓
```

### Example 2 — Single alias binding `[AB_pre, S, AB_post]`

```
1. Find O_1. Let's say it as A_1.  {AB_pre}     (1, NONE)
2. Find O_2.                       ∅            (1, NONE)
3. Find A_1.                       {AB_post}    (0, NONE)   ✓
```

### Example 3 — Ordinal back-reference `[S, S, OR_post]`

```
1. Find O_1.                       ∅            (0, NONE)   visit_order=[O_1]
2. Find O_2.                       ∅            (0, NONE)   visit_order=[O_1,O_2]
3. Find the 1st one again.         {OR_post}    (0, NONE)   → O_1   ✓
```

### Example 4 — Ordinal + alias in one instruction `[S, S, {AB_pre, OR_post}, AB_post]`

```
1. Find O_1.                              ∅                  (0, NONE)  [O_1]
2. Find O_2.                              ∅                  (0, NONE)  [O_1,O_2]
3. Find the 1st one again. Call it A_3.   {AB_pre, OR_post}  (1, NONE)  → O_1; bind A_3→O_1
4. Find A_3.                              {AB_post}          (0, NONE)  → O_1   ✓
```

### Example 5 — Anaphoric block `[S, AR_pre, S, AR_post]`

```
1. Find O_1.                       ∅            (0, NONE)
2. Find O_2.                       {AR_pre}     (0, EXPECT_ONE)    antecedent = O_2
3. Find O_3.                       ∅            (0, EXPECT_POST)   (single intervening)
4. Go back to the previous one.    {AR_post}    (0, NONE)          → O_2 (antecedent)   ✓
```

### Example 6 — Two concurrent aliases (LIFO) `[AB_pre, S, AB_pre, AB_post, S, AB_post]`

```
1. Find O_1. Let's say it as A_1.  {AB_pre}     (1, NONE)
2. Find O_2.                       ∅            (1, NONE)
3. Find O_3. Let's say it as A_3.  {AB_pre}     (2, NONE)
4. Find A_3.                       {AB_post}    (1, NONE)   ← closes most recent (A_3)
5. Find O_4.                       ∅            (1, NONE)
6. Find A_1.                       {AB_post}    (0, NONE)   ← closes A_1   ✓
```

### Example 7 — AR with an alias closing as the intervening instruction `[AB_pre, AR_pre, AB_post, AR_post]`

```
1. Find O_1. Let's say it as A_1.  {AB_pre}     (1, NONE)
2. Find O_2.                       {AR_pre}     (1, EXPECT_ONE)   antecedent = O_2
3. Find A_1.                       {AB_post}    (0, EXPECT_POST)  → O_1 (closes A_1)
4. Go back to the previous one.    {AR_post}    (0, NONE)         → O_2 (antecedent)   ✓
```

### Example 8 — Mixed `[AB_pre, S, AB_post, S, AR_pre, OR_post, AR_post]`

```
1. Find O_1. Let's say it as A_1.  {AB_pre}     (1, NONE)        [O_1]
2. Find O_2.                       ∅            (1, NONE)        [O_1,O_2]
3. Find A_1.                       {AB_post}    (0, NONE)        → O_1; [..,O_1]
4. Find O_3.                       ∅            (0, NONE)        [..,O_3]
5. Find O_4.                       {AR_pre}     (0, EXPECT_ONE)  antecedent = O_4; [..,O_4]
6. Find the 1st one again.         {OR_post}    (0, EXPECT_POST) → O_1 (visit slot 1); [..,O_1]
7. Go back to the previous one.    {AR_post}    (0, NONE)        → O_4 (antecedent)   ✓
```

---

## 7. Generation Algorithm (recommended)

Drive the automaton forward, choosing admissible tokens and assigning objects:

```
function generate_episode(max_length, max_objects, weights):
    n_AB        = 0
    ar_mode     = NONE
    visit_order = []        # objects, instruction order, with repeats
    alias_map   = []        # LIFO stack of (alias_id -> object)
    antecedent  = None
    objects_used = set()
    seq = []

    while True:
        v = len(visit_order)   # prior visits

        # 1. Admissible role sets given (n_AB, ar_mode, v)
        candidates = admissible_roles(n_AB, ar_mode, v)

        # 2. Bias toward closing (AB_post / AR_post) as length approaches max,
        #    and forbid termination unless n_AB == 0 and ar_mode == NONE.
        rho = weighted_choice(candidates, weights, remaining = max_length - len(seq))

        # 3. Resolve / choose the target object:
        if rho contains OR_post:
            k = choose_valid_ordinal(v)            # 1..v
            target = visit_order[k-1]
        elif rho == {AB_post}:
            (alias_id, target) = alias_map.pop()   # LIFO (or chosen open alias)
        elif rho == {AR_post}:
            target = antecedent; antecedent = None
        else:
            target = choose_object(objects_used, max_objects)  # new or existing
            objects_used.add(target)

        if rho contains AB_pre:
            alias_id = fresh_alias_id()
            alias_map.push((alias_id, target))
        if rho == {AR_pre}:
            antecedent = target

        # 4. Apply automaton transition (updates n_AB, ar_mode)
        (n_AB, ar_mode) = transition(n_AB, ar_mode, rho)

        visit_order.append(target)
        seq.append(render_instruction(rho, target, alias_id_if_any, k_if_any))

        # 5. Terminate when allowed and length target met
        if n_AB == 0 and ar_mode == NONE and len(seq) >= target_length:
            break

    return seq
```

### Termination
End only when `n_AB == 0` and `ar_mode == NONE`. Bias toward `AB_post` (and
completing any open AR block) as the length budget runs out.

### Admissibility helper
- `ar_mode == EXPECT_ONE` → any non-AR role (respecting `OR_post` needs `v ≥ 1`,
  `AB_post` needs `n_AB > 0`).
- `ar_mode == EXPECT_POST` → only `AR_post`.
- `ar_mode == NONE` → any role except `AR_post`; `OR_post` needs `v ≥ 1`;
  `AB_post` needs `n_AB > 0`.

---

## 8. Parameters for the Generator

| Parameter | Description |
|-----------|-------------|
| `max_length` | maximum number of instructions per episode |
| `max_objects` | maximum number of distinct target objects per episode |
| `mechanism_weights` | sampling weights for `S` / `AB` / `OR_post` / `AR` roles |
| `multi_role_prob` | probability of emitting `{AB_pre, OR_post}` when admissible |
| `min_gap` | optional minimum #instructions between an `AB_pre` and its `AB_post` |
| `ordinal_index_policy` | how `k` is chosen for `OR_post` (e.g., uniform over `1..v`, or biased to recent/old) |
| `phrasing_variants` | alternate surface forms per role (esp. `AR_post`) |

---

## 9. Open Design Notes

- **OR index target.** `k` indexes the visit (instruction) order **including
  repeats**. Confirm whether annotators/agents should interpret "the k-th one"
  over all navigations (current spec) or only over distinct first-visits. The
  current spec uses all navigations.
- **AB referencing multiplicity (decided).** AB is one-`pre`/one-`post`
  (balanced): an alias is referenced exactly once. Multiple references to the
  same alias are **not** allowed.
- **AB closing order.** Default LIFO. Non-LIFO closing is possible if
  `alias_map` is tracked explicitly rather than as a bare counter.
- **AR resolution (decided).** `AR_post` resolves to the antecedent (the
  `AR_pre` target), i.e., the object found *before* the single intervening
  instruction. Phrasings must denote "the one before the most recent," not
  "the one you just found."
- **AR intervening token.** Any non-AR role is structurally allowed as the
  intervening instruction; restrict to semantically sensible choices for
  natural episodes if desired.