"""Check whether RefON instructions are resolvable from text alone.

This is a *feasibility* probe, not a navigation run: habitat is never started, no
images are rendered, and the VLM is never asked to move. It is given only the
instructions of an episode and has to say, for one of them, *which object it is being
told to find* -- either the instruction number that first introduced that object, or 0
when the instruction names no object at all.

Why: a RefON instruction is anaphoric ("Find A1.", "Find the 2nd one again.", "Go back
to the previous one."). If the model cannot resolve the reference even when the whole
instruction history is handed to it in plain text, then a navigation failure on that
subgoal says nothing about perception or exploration -- the referent was never
recovered. This script isolates that step, so the navigation numbers in
run_refonbench_evaluation.py can be read against a ceiling.

Two ways of presenting an episode, mirroring the two things one might ask:

  --mode incremental   (default) one query per subgoal, showing instructions 1..i and
                       asking about instruction i. This is what the navigation run
                       actually has available at subgoal i.
  --mode all_at_once   one query per episode, showing every instruction and asking for
                       all referents at once. The model can look ahead, so it is an
                       upper bound on the same skill.

Success is reported per instruction style (the subtask `role`: S / AB_pre / AB_post /
AR_pre / AR_post / OR_post / AB_pre+OR_post / GA_*), which is the breakdown the
generator's styles are defined by.

Usage:
    python run_refonbench_feasibility.py -cf cfg/eval_refonbench_feasibility.yaml
    python run_refonbench_feasibility.py -cf cfg/... --mode all_at_once --workers 4
    python run_refonbench_feasibility.py -cf cfg/... --dry-run   # print prompts, no VLM
"""

import os

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

from omegaconf import OmegaConf

from src.refonbench_utils import (
    list_shard_files,
    load_shard,
    scene_name_from_shard,
    select_episodes,
)
from src.vlm_client import create_vlm_client

# presentation order, matching scripts/summarize_refonbench.py; unknown roles are
# appended alphabetically
ROLE_ORDER = [
    "S",
    "AB_pre",
    "AB_post",
    "AR_pre",
    "AR_post",
    "OR_post",
    "AB_pre+OR_post",
]

# scripts/summarize_refonbench.py folds these into "S" by default: as a *navigation*
# task, "Find the clothes. Let's call it A1." is the same job as "Find the clothes."
# Reference resolution is not the same job -- the model still has to work out that "A1"
# is a name being bound and not a second object to go looking for -- so the fold is off
# by default here, and --merge-roles turns it on when the two tables need to line up.
ROLE_MERGE = {"AB_pre": "S", "AR_pre": "S"}


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------

SYS_PROMPT = (
    "You are analysing the instructions of an indoor object-search task. A robot is "
    "given a sequence of instructions, one at a time, and each instruction sends it to "
    "exactly one object in the house.\n"
    "Instructions come in several forms:\n"
    "  * a direct one names the object: \"Find the toilet.\"\n"
    "  * an alias reference points at an object that was named earlier: \"Find A1.\", "
    "where A1 was bound by an earlier instruction ending in \"Let's call it A1.\"\n"
    "  * an ordinal reference counts over the instructions already given: \"Find the "
    "2nd one again.\" means the object of the 2nd instruction.\n"
    "  * a relative reference points just before the latest visit: \"Go back to the "
    "previous one.\" / \"Find the one before that.\" / \"Go back to the one before the "
    "last.\" all mean the object of the instruction *before the most recent one*, not "
    "the most recent one itself.\n"
    "\"Let's call it X\" binds the name X to the object being found by that very "
    "instruction; it is not a request to search for something called X.\n"
    "An alias that was never bound refers to no object at all, and neither does an "
    "ordinal that points past the instructions given so far.\n"
    "You are not navigating and you cannot see the house. Your only job is to work out "
    "which object each instruction is talking about, from the instructions themselves.\n"
    "Answer with JSON only. No prose, no markdown fences."
)

_ANSWER_FIELDS = (
    '  "answer": exactly one of\n'
    '      "new"            -- no earlier instruction went to this object; the '
    "instruction names a fresh one.\n"
    '      "back_reference" -- an earlier instruction already went to this same object.\n'
    '      "no_object"      -- the instruction points at nothing (an alias that was '
    "never bound, an ordinal past the end of the list).\n"
    '  "refers_to": only when "answer" is "back_reference" -- the number of the earlier '
    "instruction that went to the object. null otherwise.\n"
    '  "category": the object category as a short noun phrase ("toilet", "cardboard '
    'box"). Always name it -- for a back reference, repeat the category the earlier '
    'instruction used. Only "no_object" may leave it null.\n'
    '  "reason": one short sentence.\n'
)

_ANSWER_SCHEMA = (
    '{"answer": "new" | "back_reference" | "no_object", '
    '"refers_to": <integer 1..%d or null>, '
    '"category": <string or null>, "reason": <string>}'
)


def _numbered(instructions: List[str], upto: Optional[int] = None) -> str:
    upto = len(instructions) if upto is None else upto
    return "\n".join(
        f'  {i + 1}. "{instructions[i]}"' for i in range(min(upto, len(instructions)))
    )


def build_incremental_prompt(instructions: List[str], index: int) -> List[Tuple]:
    """Prompt asking about instruction `index` (0-based) given instructions 1..index+1."""
    n = index + 1
    text = (
        "Instructions given to the robot so far, in order:\n"
        + _numbered(instructions, n)
        + "\n\n"
        + f"Question: instruction {n} sends the robot to one object. Is that object one "
        f"an earlier instruction already went to, or a new one?\n\n"
        + "Reply with a single JSON object:\n"
        + (_ANSWER_SCHEMA % n)
        + "\n"
        + _ANSWER_FIELDS
    )
    return [(text,)]


def build_all_at_once_prompt(instructions: List[str]) -> List[Tuple]:
    """Prompt asking about every instruction of the episode in one shot."""
    n = len(instructions)
    text = (
        "The robot is given these instructions, in order:\n"
        + _numbered(instructions)
        + "\n\n"
        + f"Question: each of the {n} instructions sends the robot to one object. For "
        "each one, is that object one an earlier instruction already went to, or a new "
        "one?\n\n"
        + "Reply with a single JSON array of exactly "
        + f"{n} objects, one per instruction, in order:\n"
        + '[{"instruction": 1, '
        + (_ANSWER_SCHEMA % n).lstrip("{")
        + ", ...]\n"
        + _ANSWER_FIELDS
    )
    return [(text,)]


# ---------------------------------------------------------------------------
# ground truth and scoring
# ---------------------------------------------------------------------------


def ground_truth_refers_to(subtasks: List[Dict], index: int) -> int:
    """1-based number of the earliest instruction that mentions subtask `index`'s object.

    0 when the instruction refers to no object, and ``index + 1`` ("new") when it
    introduces one.

    The three goal-absent kinds are not the same question here, because this probe reads
    the instructions and nothing else:

      GA_unbound_alias   "Find Z1."              -> 0. The alias was never bound, and the
                                                   text says so.
      GA_invalid_ordinal "Find the 8th one again." with 6 prior visits
                                                 -> 0. Out of range, and the text says so.
      GA_absent_object   "Find the chandelier."  -> "new". It is goal-absent because the
                                                   *scene* holds no chandelier, which is
                                                   not knowable from the instructions.
                                                   Demanding "none" would score a fact
                                                   the model was never given.

    The first two carry no category; the third does, which is what distinguishes them.
    """
    subtask = subtasks[index]
    if subtask.get("goal_absent") or subtask.get("object_id") is None:
        return index + 1 if subtask.get("category") else 0
    object_id = subtask["object_id"]
    for i in range(index + 1):
        if subtasks[i].get("object_id") == object_id:
            return i + 1
    return index + 1  # unreachable: the subtask itself matches


def referent_object_id(
    subtasks: List[Dict], answer: Optional[int], max_index: Optional[int] = None
) -> Optional[str]:
    """The object a predicted instruction number points at, or None if it points nowhere.

    ``max_index`` caps which instructions may be pointed at (1-based, inclusive).
    """
    limit = len(subtasks) if max_index is None else min(max_index, len(subtasks))
    if answer is None or answer <= 0 or answer > limit:
        return None
    return subtasks[answer - 1].get("object_id")


_ARTICLES = ("the ", "a ", "an ")


def normalize_category(text: Optional[str]) -> Optional[str]:
    """Loose category comparison: case, punctuation, articles and plural 's' are noise."""
    if text is None:
        return None
    t = str(text).strip().lower()
    if t in ("", "null", "none", "n/a", "nothing"):
        return None
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    for article in _ARTICLES:
        if t.startswith(article):
            t = t[len(article) :]
            break
    head, _, last = t.rpartition(" ")
    last = _singular(last)
    t = f"{head} {last}".strip() if head else last
    return t or None


def _singular(word: str) -> str:
    """Crude de-pluralisation, enough to make "boxes"/"box" and "books"/"book" agree."""
    if word.endswith("es") and re.search(r"(s|x|z|ch|sh)es$", word):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def parse_refers_to(value, index: int) -> Optional[int]:
    """A ``refers_to`` value -> instruction number (1-based), 0 for "none", None if unusable."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        v = value.strip().strip('"').lower()
        if v.lstrip("-").isdigit():
            return int(v)
        if v.startswith("new"):
            return index + 1
        if v in ("none", "null", "no object", "nothing", "n/a"):
            return 0
        # "instruction 3", "#3", "3rd"
        match = re.search(r"\d+", v)
        if match:
            return int(match.group())
    if value is None:
        return None
    return None


def resolve_answer(pred: Dict, index: int) -> Optional[int]:
    """The ("answer", "refers_to") pair -> one instruction number, in ground-truth terms.

    ``index + 1`` for "new", 0 for "no_object", and the given number for
    "back_reference". The two fields are asked separately on purpose: when the model was
    asked for a bare number, it defaulted to pointing at some earlier instruction (231 of
    534 fresh objects were answered with an earlier number, the reason line often naming
    the right one); when "new" was one of the values that field could take, it defaulted
    the other way (373 back-references answered "new", 279 of them still naming the true
    referent's category). Classifying first, then numbering, stops either default from
    swallowing the other case.
    """
    label = pred.get("answer")
    label = label.strip().strip('"').lower() if isinstance(label, str) else ""

    if label.startswith("no_object") or label in ("none", "no object", "nothing"):
        return 0
    if label.startswith("new"):
        return index + 1
    if label.startswith("back"):
        # a back reference with no usable number resolves nowhere, which is a wrong
        # answer rather than an unparseable one -- the model did commit to a claim
        return parse_refers_to(pred.get("refers_to"), index)

    # no (or unrecognised) label: fall back to whatever refers_to holds, so a reply that
    # ignored the schema is still scored on what it did say
    return parse_refers_to(pred.get("refers_to"), index)


def score_prediction(subtasks: List[Dict], index: int, pred: Dict) -> Dict:
    """Compare one parsed answer against the dataset."""
    gt_refers_to = ground_truth_refers_to(subtasks, index)
    gt_object_id = subtasks[index].get("object_id")
    gt_category = subtasks[index].get("category")

    pred_refers_to = resolve_answer(pred, index)

    # the category question is the same whatever the referent is: name the object
    category_correct = normalize_category(pred.get("category")) == normalize_category(
        gt_category
    )

    if gt_refers_to == 0:
        # refers to nothing (unbound alias / out-of-range ordinal)
        referent_correct = pred_refers_to == 0
    elif gt_refers_to == index + 1:
        # the instruction introduces the object, so "new" (= its own number) is the
        # only right answer
        referent_correct = pred_refers_to == index + 1
    else:
        # a back-reference: any EARLIER instruction that lands on the same object counts,
        # because an episode can visit one object several times and "the 3rd one" and
        # "the 2nd one" are then both true statements about the same referent. The
        # subgoal's own number is excluded on purpose -- it trivially carries the right
        # object_id, so counting it would score "this is a new object" as a correct
        # resolution of a back-reference.
        referent_correct = (
            referent_object_id(subtasks, pred_refers_to, max_index=index)
            == gt_object_id
        )

    return {
        "gt_refers_to": gt_refers_to,
        "gt_object_id": gt_object_id,
        "gt_category": gt_category,
        "pred_refers_to": pred_refers_to,
        "pred_answer": pred.get("answer"),
        "pred_refers_to_raw": pred.get("refers_to"),
        "pred_category": pred.get("category"),
        "reason": pred.get("reason"),
        "referent_correct": bool(referent_correct),
        "category_correct": bool(category_correct),
        "correct": bool(referent_correct and category_correct),
    }


# ---------------------------------------------------------------------------
# response parsing
# ---------------------------------------------------------------------------


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"```\s*$", "", text)
    # some models emit a <think>...</think> preamble
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    return text.strip()


def _first_json(text: str, opener: str, closer: str):
    """Extract the first balanced {...} / [...] block, ignoring braces inside strings."""
    start = text.find(opener)
    if start == -1:
        return None
    depth, in_string, escaped = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def parse_single_answer(response: Optional[str]) -> Optional[Dict]:
    if not response:
        return None
    text = _strip_fences(response)
    obj = _first_json(text, "{", "}")
    if isinstance(obj, dict):
        return obj
    # a model that ignored the format but still wrote an answer is worth recovering
    label = re.search(r"answer[\"'\s:=]*[\"']?(new|back_reference|no_object)", text, flags=re.I)
    number = re.search(r"refers_to[\"'\s:=]*[\"']?(new|none|\d+)", text, flags=re.I)
    if label or number:
        return {
            "answer": label.group(1) if label else None,
            "refers_to": number.group(1) if number else None,
            "category": None,
            "reason": text[:200],
        }
    return None


def parse_list_answer(response: Optional[str], n: int) -> Optional[List[Dict]]:
    if not response:
        return None
    text = _strip_fences(response)
    arr = _first_json(text, "[", "]")
    if not isinstance(arr, list):
        return None
    answers: List[Optional[Dict]] = [None] * n
    for position, item in enumerate(arr):
        if not isinstance(item, dict):
            continue
        # trust an explicit "instruction" index when it is in range, else fall back to
        # the position in the array
        idx = item.get("instruction")
        if isinstance(idx, str) and idx.strip().isdigit():
            idx = int(idx.strip())
        if not (isinstance(idx, int) and 1 <= idx <= n):
            idx = position + 1
        if idx <= n:
            answers[idx - 1] = item
    return [a if a is not None else {} for a in answers]


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------


def model_slug() -> str:
    """The configured model as a filename-safe tag, e.g. ``qwen2.5vl:7b`` -> ``qwen2_5vl_7b``.

    Read from src.const rather than from a live client, so --dry-run names its output
    directory the same way a real run would.
    """
    from src.const import OLLAMA_MODEL, OPENAI_MODEL, VLM_PROVIDER

    name = OLLAMA_MODEL if (VLM_PROVIDER or "").lower() == "ollama" else OPENAI_MODEL
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", str(name).lower())).strip("_")


def sort_roles(roles) -> List[str]:
    known = [r for r in ROLE_ORDER if r in roles]
    return known + sorted(r for r in roles if r not in ROLE_ORDER)


def aggregate(records: List[Dict], merge_roles: bool = False) -> Dict:
    """Success rates overall and per instruction style."""

    def _bucket():
        return {
            "count": 0,
            "referent_correct": 0,
            "category_correct": 0,
            "correct": 0,
            "parse_failed": 0,
        }

    per_role: Dict[str, Dict] = {}
    overall = _bucket()
    for r in records:
        role = ROLE_MERGE.get(r["role"], r["role"]) if merge_roles else r["role"]
        bucket = per_role.setdefault(role, _bucket())
        for target in (bucket, overall):
            target["count"] += 1
            target["referent_correct"] += int(r["referent_correct"])
            target["category_correct"] += int(r["category_correct"])
            target["correct"] += int(r["correct"])
            target["parse_failed"] += int(r["parse_failed"])

    def _rates(b):
        n = max(b["count"], 1)
        return {
            "count": b["count"],
            "referent_sr": b["referent_correct"] / n,
            "category_sr": b["category_correct"] / n,
            "sr": b["correct"] / n,
            "parse_failed": b["parse_failed"],
        }

    return {
        "overall": _rates(overall),
        "per_style": {role: _rates(per_role[role]) for role in sort_roles(per_role)},
    }


def format_table(summary: Dict) -> str:
    header = (
        f"{'instruction style':<20}{'n':>6}{'referent SR':>14}"
        f"{'category SR':>14}{'joint SR':>11}{'unparsed':>10}"
    )
    lines = [header, "-" * len(header)]
    for role, s in summary["per_style"].items():
        lines.append(
            f"{role:<20}{s['count']:>6}{s['referent_sr']:>13.1%}"
            f"{s['category_sr']:>14.1%}{s['sr']:>11.1%}{s['parse_failed']:>10}"
        )
    o = summary["overall"]
    lines.append("-" * len(header))
    lines.append(
        f"{'ALL':<20}{o['count']:>6}{o['referent_sr']:>13.1%}"
        f"{o['category_sr']:>14.1%}{o['sr']:>11.1%}{o['parse_failed']:>10}"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def collect_queries(cfg, mode: str) -> List[Dict]:
    """One entry per VLM call, with everything needed to build and score it.

    Each entry is independent -- the prompt is built from the dataset instructions, not
    from the model's earlier answers -- so the calls may be run in any order.
    """
    include_goal_absent = cfg.get("include_goal_absent", True)
    episodes_per_scene = cfg.get("episodes_per_scene", None)

    queries: List[Dict] = []
    shard_files = list_shard_files(cfg.test_data_dir)
    logging.info(f"Found {len(shard_files)} shard(s) in {cfg.test_data_dir}")

    for shard_file in shard_files:
        scene_name = scene_name_from_shard(shard_file)
        shard = load_shard(os.path.join(cfg.test_data_dir, shard_file))
        episodes = select_episodes(shard["episodes"], episodes_per_scene, split=0)
        logging.info(f"Scene {scene_name}: {len(episodes)} episode(s)")

        for episode in episodes:
            subtasks = episode["subtasks"]
            instructions = [s["instruction"] for s in subtasks]
            scored = [
                i
                for i, s in enumerate(subtasks)
                if include_goal_absent or not s.get("goal_absent")
            ]
            if not scored:
                continue
            common = {
                "scene": scene_name,
                "episode_id": episode.get("episode_id"),
                "subtasks": subtasks,
                "instructions": instructions,
            }
            if mode == "all_at_once":
                queries.append(
                    dict(common, indices=scored, contents=build_all_at_once_prompt(instructions))
                )
            else:
                for i in scored:
                    queries.append(
                        dict(
                            common,
                            indices=[i],
                            contents=build_incremental_prompt(instructions, i),
                        )
                    )
    return queries


def run_query(query: Dict, client, mode: str) -> List[Dict]:
    """Send one prompt and score every subtask it covers."""
    subtasks = query["subtasks"]
    indices = query["indices"]

    start = time.time()
    response = client.call(SYS_PROMPT, query["contents"]) if client else None
    elapsed = time.time() - start

    if mode == "all_at_once":
        parsed = parse_list_answer(response, len(subtasks))
        answers = {i: (parsed[i] if parsed else None) for i in indices}
    else:
        answer = parse_single_answer(response)
        answers = {indices[0]: answer}

    prompt = "\n".join(c[0] for c in query["contents"])

    records = []
    for i in indices:
        answer = answers.get(i)
        parse_failed = not answer
        scoring = score_prediction(subtasks, i, answer or {})
        record = {
            "scene": query["scene"],
            "episode_id": query["episode_id"],
            "order": subtasks[i].get("order", i + 1),
            "role": subtasks[i]["role"],
            "instruction": subtasks[i]["instruction"],
            "goal_absent": bool(subtasks[i].get("goal_absent")),
            "mode": mode,
            "parse_failed": parse_failed,
            "raw_response": response,
            "elapsed": elapsed,
            **scoring,
        }
        # The prompt is what makes a failure diagnosable -- which history the model had
        # in front of it when it picked the wrong referent. It is also the bulk of the
        # record and identical for every correct answer, so it is kept only on failures.
        if not record["correct"]:
            record["prompt"] = prompt
        records.append(record)
    return records


def write_failure_transcripts(records: List[Dict], path: str) -> int:
    """Write the full exchange for every wrong answer. Returns the transcript count.

    Failures are grouped by the prompt that produced them, so an ``all_at_once`` episode
    whose single reply got three subgoals wrong is one transcript listing all three
    rather than three copies of the same conversation.
    """
    groups: Dict[Tuple, List[Dict]] = {}
    for r in records:
        if r["correct"]:
            continue
        key = (r["scene"], r["episode_id"], r.get("prompt"))
        groups.setdefault(key, []).append(r)

    with open(path, "w") as f:
        f.write(f"# {len(groups)} failed exchange(s)\n")
        f.write("# system prompt (identical for every exchange below)\n\n")
        f.write(SYS_PROMPT + "\n")
        for (scene, episode_id, prompt), items in groups.items():
            first = items[0]
            f.write("\n" + "=" * 78 + "\n")
            f.write(f"scene {scene} / episode {episode_id} / mode {first['mode']}\n")
            for r in items:
                f.write(
                    f"  WRONG subgoal #{r['order']} ({r['role']}) "
                    f"\"{r['instruction']}\"\n"
                    f"      predicted: {r.get('pred_answer')!r} / refers_to="
                    f"{r['pred_refers_to_raw']!r} -> instruction {r['pred_refers_to']}, "
                    f"category={r['pred_category']!r}\n"
                    f"      truth    : refers_to={r['gt_refers_to']} "
                    f"object_id={r['gt_object_id']!r} category={r['gt_category']!r}\n"
                    f"      wrong    : "
                    + ", ".join(
                        part
                        for part, ok in (
                            ("referent", r["referent_correct"]),
                            ("category", r["category_correct"]),
                        )
                        if not ok
                    )
                    + (" (reply could not be parsed)" if r["parse_failed"] else "")
                    + "\n"
                )
            f.write("\n--- USER ---\n")
            f.write((prompt or "(prompt not recorded)").rstrip() + "\n")
            f.write("\n--- ASSISTANT ---\n")
            f.write((first["raw_response"] or "(no response: every retry failed)").rstrip() + "\n")
    return len(groups)


def main(cfg, mode: str, dry_run: bool = False):
    queries = collect_queries(cfg, mode)
    logging.info(f"Mode '{mode}': {len(queries)} VLM call(s) to make")

    if dry_run:
        logging.info("--- system prompt ---\n" + SYS_PROMPT)
        for query in queries[: cfg.get("dry_run_examples", 3)]:
            logging.info(
                f"--- {query['scene']} / episode {query['episode_id']} "
                f"/ instructions {[i + 1 for i in query['indices']]} ---\n"
                + query["contents"][0][0]
            )
        logging.info("Dry run: no VLM was queried.")
        return

    client = create_vlm_client(temperature=cfg.get("temperature", 0.0))

    records: List[Dict] = []
    workers = max(int(cfg.get("workers", 1)), 1)
    done = 0
    if workers == 1:
        results = (run_query(q, client, mode) for q in queries)
    else:
        pool = ThreadPoolExecutor(max_workers=workers)
        results = pool.map(lambda q: run_query(q, client, mode), queries)

    for batch in results:
        records.extend(batch)
        done += 1
        for r in batch:
            mark = "OK " if r["correct"] else "BAD"
            logging.info(
                f"[{done}/{len(queries)}] {mark} {r['scene']} ep{r['episode_id']} "
                f"#{r['order']} ({r['role']}) \"{r['instruction']}\" -> "
                f"pred {r['pred_refers_to']}/{r['pred_category']} "
                f"gt {r['gt_refers_to']}/{r['gt_category']}"
            )

    summary = aggregate(records, merge_roles=cfg.get("merge_roles", False))
    os.makedirs(cfg.output_dir, exist_ok=True)
    records_path = os.path.join(cfg.output_dir, f"feasibility_records_{mode}.jsonl")
    with open(records_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    results_path = os.path.join(cfg.output_dir, f"feasibility_results_{mode}.json")
    with open(results_path, "w") as f:
        json.dump(
            {
                "mode": mode,
                "test_data_dir": cfg.test_data_dir,
                "model": getattr(client, "model", None),
                "num_records": len(records),
                **summary,
            },
            f,
            indent=2,
        )

    failures_path = os.path.join(cfg.output_dir, f"feasibility_failures_{mode}.log")
    num_failed = write_failure_transcripts(records, failures_path)

    logging.info("\n" + format_table(summary))
    logging.info(f"Per-subgoal records: {records_path}")
    logging.info(f"Summary: {results_path}")
    logging.info(f"Failed exchanges ({num_failed} transcript(s)): {failures_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--cfg_file", help="cfg file path", default="", type=str)
    parser.add_argument(
        "--mode",
        choices=["incremental", "all_at_once"],
        default=None,
        help="how the instructions are presented; overrides cfg.mode "
        "(default: incremental)",
    )
    parser.add_argument(
        "--episodes-per-scene",
        type=int,
        default=None,
        help="truncate each shard to its first N episodes; overrides cfg",
    )
    parser.add_argument(
        "--workers", type=int, default=None, help="parallel VLM calls; overrides cfg"
    )
    parser.add_argument(
        "--test-data-dir", default=None, help="shard directory; overrides cfg"
    )
    parser.add_argument(
        "--skip-goal-absent",
        action="store_true",
        help="drop GA_* subtasks instead of scoring them as 'refers to no object'",
    )
    parser.add_argument(
        "--merge-roles",
        action="store_true",
        help="fold AB_pre / AR_pre into S, as scripts/summarize_refonbench.py does",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the prompts that would be sent and exit without querying the VLM",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg_file)
    OmegaConf.resolve(cfg)

    # CLI wins over the config file
    mode = args.mode or cfg.get("mode", "incremental")
    if args.episodes_per_scene is not None:
        cfg.episodes_per_scene = args.episodes_per_scene
    if args.workers is not None:
        cfg.workers = args.workers
    if args.test_data_dir is not None:
        cfg.test_data_dir = args.test_data_dir
    if args.skip_goal_absent:
        cfg.include_goal_absent = False
    if args.merge_roles:
        cfg.merge_roles = True

    # Which model answered is the first thing you need to know about a feasibility
    # number -- it moves the headline SR by tens of points -- so it goes in the
    # directory name rather than only inside the summary json.
    exp_name = cfg.exp_name
    if cfg.get("append_model_to_exp_name", True):
        slug = model_slug()
        if slug and not exp_name.endswith(slug):
            exp_name = f"{exp_name}_{slug}"
    cfg.output_dir = os.path.join(cfg.output_parent_dir, exp_name)
    os.makedirs(cfg.output_dir, exist_ok=True)
    if args.cfg_file:
        os.system(f"cp {args.cfg_file} {cfg.output_dir}")

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(
                os.path.join(cfg.output_dir, f"log_feasibility_{mode}.log"), mode="w"
            ),
            logging.StreamHandler(),
        ],
    )

    logging.info(f"***** Running {cfg.exp_name} (feasibility, mode={mode}) *****")
    main(cfg, mode=mode, dry_run=args.dry_run)
