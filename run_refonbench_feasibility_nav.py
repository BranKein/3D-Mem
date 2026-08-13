"""Can the agent turn a RefON instruction into a *destination*?

run_refonbench_feasibility.py asks which object an instruction refers to. This asks the
next question: given that, where does the robot go? Still no habitat and no images -- the
house is replaced by virtual coordinates that exist only inside the conversation, so what
is measured is whether the agent tracks positions across an episode, not whether it can
perceive anything.

Every object of an episode is assigned a made-up (x, y). The agent never sees the whole
map: a position is disclosed only once an earlier subgoal has been to that object. So at
each subgoal exactly one action is right:

    explore      the instruction names an object nobody has been to yet (S / AB_pre /
                 AR_pre). Its position cannot be known, so the only correct move is to go
                 looking -- and the position is then disclosed in the next prompt.
    (x, y)       the instruction sends the robot back to an object an earlier subgoal
                 already found. The agent has to work out which one and repeat its
                 coordinates.
    infeasible   the instruction cannot be carried out at all.

GA_absent_object takes two turns, because one would be unfair. "Find the chandelier."
is goal-absent only because the *scene* holds no chandelier, and nothing in the
conversation says so -- demanding "infeasible" straight away would score a fact the agent
was never given. So it is asked twice: first the instruction alone, where "explore" is
correct, and then again after the search comes back `not_found`, where "infeasible" is.
The second turn is reported as its own style, GA_absent_object/not_found. The other two
goal-absent kinds stay single-turn: an alias that was never bound and an ordinal past the
end of the list are both refusable from the text.

That makes a back reference concrete: resolving "Find A1." is only worth anything if it
produces the place A1 actually is.

The history shows what *did* happen, not what the model answered, so every subgoal is an
independent query (and the run parallelises). A model that mishandles subgoal 3 is still
given the true state at subgoal 4, which keeps one bad answer from cascading into a run
of failures that measure nothing.

Usage:
    python run_refonbench_feasibility_nav.py -cf cfg/eval_refonbench_feasibility_nav.yaml
    python run_refonbench_feasibility_nav.py -cf cfg/... --dry-run
"""

import os

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
import logging
import random
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

# the pieces that are identical to the referent probe
from run_refonbench_feasibility import (
    _strip_fences,
    _first_json,
    aggregate,
    format_table,
    model_slug,
)

EXPLORE = "explore"
INFEASIBLE = "infeasible"

# GA_absent_object is asked twice; the follow-up is reported under its own style so the
# two questions ("would you go looking?" and "do you give up once the search failed?")
# never share a row
NOT_FOUND_ROLE = "GA_absent_object/not_found"

# what the table breaks the joint score into
PART_SCORES = (("action_correct", "action SR"), ("coordinate_correct", "coord SR"))


# ---------------------------------------------------------------------------
# the virtual map
# ---------------------------------------------------------------------------


def assign_coordinates(episode: Dict, room_size: float, min_sep: float) -> Dict[str, Tuple[float, float]]:
    """A made-up (x, y) for every object of one episode.

    Seeded per episode so a re-run, and every model, sees the same map. Positions are
    kept `min_sep` apart: two objects a decimal place away from each other would turn a
    wrong answer into a right one by accident.
    """
    rng = random.Random(f"refon-nav:{episode.get('episode_id')}")
    coords: Dict[str, Tuple[float, float]] = {}
    for subtask in episode["subtasks"]:
        object_id = subtask.get("object_id")
        if not object_id or object_id in coords:
            continue
        for _ in range(500):
            point = (
                round(rng.uniform(-room_size, room_size), 1),
                round(rng.uniform(-room_size, room_size), 1),
            )
            if all(
                abs(point[0] - x) + abs(point[1] - y) >= min_sep
                for x, y in coords.values()
            ):
                coords[object_id] = point
                break
        else:  # pragma: no cover - only if room_size is set absurdly small
            raise RuntimeError(
                f"could not place {object_id} at least {min_sep} from the others; "
                f"raise nav.room_size"
            )
    return coords


def format_coord(point: Tuple[float, float]) -> str:
    return f"({point[0]:g}, {point[1]:g})"


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------

SYS_PROMPT = (
    "You are directing a robot through a house. It is given instructions one at a time, "
    "and each instruction sends it to exactly one object.\n"
    "You cannot see the house. The only thing you know about it is what the log of "
    "earlier instructions tells you: once the robot has been to an object, that "
    "object's (x, y) position is known, and it stays known for the rest of the episode. "
    "An object nobody has been to yet has no known position, and you have no way to "
    "guess it.\n"
    "When the robot is sent to explore, one of two things comes back: the object's "
    "position, or `not_found`. `not_found` means no such object exists anywhere in this "
    "house, so that instruction can never be carried out.\n"
    "\n"
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
    "\"Let's call it X\" binds the name X to the object the robot is being sent to by "
    "that very instruction; it is not a request to search for something called X.\n"
    "\n"
    "Answer with JSON only. No prose, no markdown fences."
)

_ACTION_FIELDS = (
    '  "action": exactly one of\n'
    '      "explore"      -- the instruction sends the robot to an object whose position '
    "is not in the log. Nobody has been there, so it has to be searched for.\n"
    '      "(x, y)"       -- the instruction sends the robot back to an object whose '
    "position IS in the log. Copy that position exactly, e.g. \"(3.2, -1.4)\".\n"
    '      "infeasible"   -- the instruction cannot be carried out at all: it uses an '
    "alias that was never bound, or an ordinal past the end of the list, or a search for "
    "it has already come back `not_found`.\n"
    '  "reason": one short sentence.\n'
)


def _history_lines(subtasks: List[Dict], coords: Dict, index: int) -> str:
    """The log the agent sees: what each earlier instruction did, with positions."""
    seen = set()
    lines = []
    for i in range(index):
        subtask = subtasks[i]
        object_id = subtask.get("object_id")
        instruction = subtask["instruction"]
        if not object_id or object_id not in coords:
            outcome = (
                f"explored, and the search reported: not_found"
                if subtask.get("role") == "GA_absent_object"
                else "could not be carried out"
            )
        elif object_id in seen:
            outcome = f"went to {format_coord(coords[object_id])}"
        else:
            seen.add(object_id)
            outcome = (
                f"explored, and found the {subtask.get('category')} at "
                f"{format_coord(coords[object_id])}"
            )
        lines.append(f'  {i + 1}. "{instruction}"  ->  {outcome}')
    return "\n".join(lines)


def build_prompt(
    subtasks: List[Dict], coords: Dict, index: int, after_not_found: bool = False
) -> List[Tuple]:
    """Ask for the action at subgoal `index` (0-based), given everything before it.

    With ``after_not_found``, this is the second turn of a GA_absent_object subgoal: the
    agent already said "explore" and the search came back empty.
    """
    log = _history_lines(subtasks, coords, index)
    header = (
        "Log of the instructions given so far, and what came of each:\n" + log + "\n\n"
        if log
        else "The robot has not been given any instruction yet, so nothing is known "
        "about the house.\n\n"
    )
    now = f'Now: instruction {index + 1}. "{subtasks[index]["instruction"]}"\n'
    if after_not_found:
        now += (
            "You sent the robot to explore. It searched and reported: not_found\n"
        )
    text = (
        header
        + now
        + "\nWhere does the robot go? Reply with a single JSON object:\n"
        + '{"action": "explore" | "(x, y)" | "infeasible", "reason": <string>}\n'
        + _ACTION_FIELDS
    )
    return [(text,)]


# ---------------------------------------------------------------------------
# ground truth and scoring
# ---------------------------------------------------------------------------


def ground_truth_action(
    subtasks: List[Dict], coords: Dict, index: int, after_not_found: bool = False
):
    """(kind, point) the agent should produce at subgoal `index`.

    kind is "explore", "coordinate" or "infeasible"; point is set only for "coordinate".

    GA_absent_object is the awkward one and gets two turns: nothing in the conversation
    says the house has no chandelier, so the first turn expects "explore" like any other
    unseen object, and only the turn after the search reported `not_found` expects
    "infeasible". The other goal-absent kinds are refusable from the text alone.
    """
    subtask = subtasks[index]
    if after_not_found:
        return INFEASIBLE, None
    object_id = subtask.get("object_id")
    if subtask.get("goal_absent") or object_id is None:
        if subtask.get("role") == "GA_absent_object":
            return EXPLORE, None
        return INFEASIBLE, None
    if any(subtasks[i].get("object_id") == object_id for i in range(index)):
        return "coordinate", coords[object_id]
    return EXPLORE, None


_COORD_RE = re.compile(r"\(?\s*(-?\d+(?:\.\d+)?)\s*[,;]\s*(-?\d+(?:\.\d+)?)\s*\)?")


def parse_action(pred: Dict) -> Tuple[Optional[str], Optional[Tuple[float, float]]]:
    """The model's "action" field -> (kind, point)."""
    raw = pred.get("action")
    if raw is None:
        return None, None
    text = str(raw).strip().strip('"').lower()
    if text.startswith(EXPLORE):
        return EXPLORE, None
    if text.startswith(INFEASIBLE) or "infeasib" in text:
        return INFEASIBLE, None
    match = _COORD_RE.search(text)
    if match:
        return "coordinate", (float(match.group(1)), float(match.group(2)))
    return None, None


def score_prediction(subtasks, coords, index, pred, tol, after_not_found=False):
    gt_kind, gt_point = ground_truth_action(subtasks, coords, index, after_not_found)
    pred_kind, pred_point = parse_action(pred)

    action_correct = pred_kind == gt_kind

    if gt_kind == "coordinate":
        coordinate_correct = bool(
            pred_point
            and abs(pred_point[0] - gt_point[0]) <= tol
            and abs(pred_point[1] - gt_point[1]) <= tol
        )
    else:
        # nothing to point at, so this part-score is satisfied by not inventing a place
        coordinate_correct = pred_kind != "coordinate" or action_correct

    return {
        "gt_action": gt_kind,
        "gt_coord": list(gt_point) if gt_point else None,
        "pred_action": pred_kind,
        "pred_action_raw": pred.get("action"),
        "pred_coord": list(pred_point) if pred_point else None,
        "reason": pred.get("reason"),
        "action_correct": bool(action_correct),
        "coordinate_correct": bool(coordinate_correct),
        "correct": bool(action_correct and coordinate_correct),
    }


def parse_reply(response: Optional[str]) -> Optional[Dict]:
    if not response:
        return None
    text = _strip_fences(response)
    obj = _first_json(text, "{", "}")
    if isinstance(obj, dict):
        return obj
    # a reply that ignored the schema but still stated an action is worth recovering
    match = re.search(r"(explore|infeasible|\(\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\))",
                      text, flags=re.I)
    if match:
        return {"action": match.group(1), "reason": text[:200]}
    return None


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def collect_queries(cfg) -> List[Dict]:
    nav = cfg.get("nav", {})
    room_size = float(nav.get("room_size", 8.0))
    min_sep = float(nav.get("min_separation", 1.0))
    include_goal_absent = cfg.get("include_goal_absent", True)

    queries: List[Dict] = []
    for shard_file in list_shard_files(cfg.test_data_dir):
        scene = scene_name_from_shard(shard_file)
        shard = load_shard(os.path.join(cfg.test_data_dir, shard_file))
        episodes = select_episodes(
            shard["episodes"], cfg.get("episodes_per_scene", None), split=0
        )
        logging.info(f"Scene {scene}: {len(episodes)} episode(s)")
        for episode in episodes:
            subtasks = episode["subtasks"]
            coords = assign_coordinates(episode, room_size, min_sep)
            for i, subtask in enumerate(subtasks):
                if not include_goal_absent and subtask.get("goal_absent"):
                    continue
                common = {
                    "scene": scene,
                    "episode_id": episode.get("episode_id"),
                    "subtasks": subtasks,
                    "coords": coords,
                    "index": i,
                }
                queries.append(
                    dict(common, after_not_found=False,
                         contents=build_prompt(subtasks, coords, i))
                )
                # the follow-up turn: it explored, the search came back empty
                if subtask.get("role") == "GA_absent_object":
                    queries.append(
                        dict(common, after_not_found=True,
                             contents=build_prompt(subtasks, coords, i,
                                                   after_not_found=True))
                    )
    return queries


def run_query(query: Dict, client, cfg) -> Optional[Dict]:
    if client is not None and client.gave_up:
        return None  # the model has been dropped; stop spending time on it
    start = time.time()
    response = client.call(SYS_PROMPT, query["contents"]) if client else None
    elapsed = time.time() - start

    pred = parse_reply(response)
    i, subtasks = query["index"], query["subtasks"]
    after_not_found = query["after_not_found"]
    scoring = score_prediction(
        subtasks,
        query["coords"],
        i,
        pred or {},
        float(cfg.get("nav", {}).get("coord_tolerance", 0.05)),
        after_not_found=after_not_found,
    )
    record = {
        "scene": query["scene"],
        "episode_id": query["episode_id"],
        "order": subtasks[i].get("order", i + 1),
        "role": NOT_FOUND_ROLE if after_not_found else subtasks[i]["role"],
        "after_not_found": after_not_found,
        "instruction": subtasks[i]["instruction"],
        "goal_absent": bool(subtasks[i].get("goal_absent")),
        "parse_failed": not pred,
        "empty_response": not response,
        "raw_response": response,
        "elapsed": elapsed,
        **scoring,
    }
    if not record["correct"]:
        record["prompt"] = query["contents"][0][0]
    return record


def write_failure_transcripts(records: List[Dict], path: str) -> int:
    failed = [r for r in records if not r["correct"]]
    with open(path, "w") as f:
        f.write(f"# {len(failed)} failed exchange(s)\n")
        f.write("# system prompt (identical for every exchange below)\n\n")
        f.write(SYS_PROMPT + "\n")
        for r in failed:
            f.write("\n" + "=" * 78 + "\n")
            f.write(
                f"scene {r['scene']} / episode {r['episode_id']} / subgoal "
                f"#{r['order']} ({r['role']})\n"
                f"  instruction: \"{r['instruction']}\"\n"
                f"  predicted  : {r['pred_action_raw']!r} -> {r['pred_action']}"
                f"{' ' + str(r['pred_coord']) if r['pred_coord'] else ''}\n"
                f"  truth      : {r['gt_action']}"
                f"{' ' + str(r['gt_coord']) if r['gt_coord'] else ''}\n"
                + ("  (no reply at all)\n" if r["empty_response"] else "")
            )
            f.write("\n--- USER ---\n")
            f.write((r.get("prompt") or "(prompt not recorded)").rstrip() + "\n")
            f.write("\n--- ASSISTANT ---\n")
            f.write((r["raw_response"] or "(no response)").rstrip() + "\n")
    return len(failed)


def main(cfg, dry_run: bool = False):
    queries = collect_queries(cfg)
    logging.info(f"{len(queries)} VLM call(s) to make")

    if dry_run:
        logging.info("--- system prompt ---\n" + SYS_PROMPT)
        for query in queries[: cfg.get("dry_run_examples", 4)]:
            logging.info(
                f"--- {query['scene']} / episode {query['episode_id']} / subgoal "
                f"{query['index'] + 1} ---\n" + query["contents"][0][0]
            )
        logging.info("Dry run: no VLM was queried.")
        return

    client = create_vlm_client(
        temperature=cfg.get("temperature", 0.0),
        max_tokens=cfg.get("max_tokens", 16384),
        presence_penalty=cfg.get("presence_penalty", 0.0),
        reasoning_effort=cfg.get("reasoning_effort", None),
        max_length_stops=int(cfg.get("max_length_stops", 5)),
    )

    workers = max(int(cfg.get("workers", 1)), 1)
    if workers == 1:
        results = (run_query(q, client, cfg) for q in queries)
    else:
        pool = ThreadPoolExecutor(max_workers=workers)
        results = pool.map(lambda q: run_query(q, client, cfg), queries)

    records = []
    for done, record in enumerate(results, 1):
        if record is None:
            continue
        records.append(record)
        logging.info(
            f"[{done}/{len(queries)}] {'OK ' if record['correct'] else 'BAD'} "
            f"{record['scene']} ep{record['episode_id']} #{record['order']} "
            f"({record['role']}) \"{record['instruction']}\" -> "
            f"{record['pred_action']}{record['pred_coord'] or ''} "
            f"gt {record['gt_action']}{record['gt_coord'] or ''}"
        )

    if client.gave_up:
        logging.error(
            f"DROPPED {client.model}: {client.length_stops} replies were cut off by the "
            f"token budget (finish_reason 'length'), the limit is "
            f"{client.max_length_stops}. Ran {len(records)}/{len(queries)} queries. "
            f"This model does not answer within cfg.max_tokens ({client.max_tokens}) -- "
            f"for a reasoning model, set cfg.reasoning_effort: none, or raise max_tokens."
        )
        raise SystemExit(3)

    summary = aggregate(
        records,
        merge_roles=cfg.get("merge_roles", False),
        part_scores=tuple((f, r) for f, r in
                          (("action_correct", "action_sr"),
                           ("coordinate_correct", "coordinate_sr"))),
    )
    os.makedirs(cfg.output_dir, exist_ok=True)
    records_path = os.path.join(cfg.output_dir, "feasibility_nav_records.jsonl")
    with open(records_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    results_path = os.path.join(cfg.output_dir, "feasibility_nav_results.json")
    with open(results_path, "w") as f:
        json.dump(
            {
                "test_data_dir": cfg.test_data_dir,
                "model": getattr(client, "model", None),
                "num_records": len(records),
                **summary,
            },
            f,
            indent=2,
        )
    failures_path = os.path.join(cfg.output_dir, "feasibility_nav_failures.log")
    num_failed = write_failure_transcripts(records, failures_path)

    empty = sum(1 for r in records if r["empty_response"])
    if empty:
        logging.warning(
            f"WARNING: {empty}/{len(records)} subgoals got no reply at all. These score "
            f"as wrong but measure nothing -- raise cfg.max_tokens and re-run."
        )

    logging.info("\n" + format_table(
        summary, columns=(("action SR", "action_sr"), ("coord SR", "coordinate_sr"))
    ))
    logging.info(f"Per-subgoal records: {records_path}")
    logging.info(f"Summary: {results_path}")
    logging.info(f"Failed exchanges ({num_failed}): {failures_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--cfg_file", default="", type=str)
    parser.add_argument("--episodes-per-scene", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None, help='sampling temperature; overrides cfg. Thinking models need a nonzero value: greedy decoding sends Qwen3.5 into a repetition loop that burns the whole token budget (measured: 32768 tokens, empty content, every query). Qwen recommends 0.6 with thinking on.')
    parser.add_argument("--presence-penalty", type=float, default=None, help='presence_penalty; overrides cfg. The brake on a thinking model that loops instead of answering -- measured on Qwen3.5-2B over 23 real prompts, 1.5 takes truncation from 11/23 to 1/23. Raising max_tokens does not help (13/23 truncated at 60k): it is a repetition loop, not a budget shortfall.')
    parser.add_argument("--test-data-dir", default=None)
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        help='pass reasoning_effort to the model ("none" turns thinking off). Appends '
             "_nothink to the output directory so the two runs stay apart.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg_file)
    OmegaConf.resolve(cfg)
    if args.episodes_per_scene is not None:
        cfg.episodes_per_scene = args.episodes_per_scene
    if args.workers is not None:
        cfg.workers = args.workers
    if args.temperature is not None:
        cfg.temperature = args.temperature
    if args.presence_penalty is not None:
        cfg.presence_penalty = args.presence_penalty
    if args.test_data_dir is not None:
        cfg.test_data_dir = args.test_data_dir

    if args.reasoning_effort:
        cfg.reasoning_effort = args.reasoning_effort
        if args.reasoning_effort == "none":
            cfg.exp_name = f"{cfg.exp_name}_nothink"

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
            logging.FileHandler(os.path.join(cfg.output_dir, "log_feasibility_nav.log"), mode="w"),
            logging.StreamHandler(),
        ],
    )
    logging.info(f"***** Running {cfg.exp_name} (feasibility-nav) *****")
    main(cfg, dry_run=args.dry_run)
