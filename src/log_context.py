"""Where-am-I context for evaluation logs.

An evaluation run interleaves thousands of lines from the planner, the detector and the
VLM, so a line on its own says nothing about which subgoal produced it. This module
keeps the current scene / episode / subgoal in module state and stamps every log record
with it:

    00:12:38 [S1/1 E3/26 G2/5] Response: [Snapshot 0, Object 2]

and draws separators at the boundaries, so the log can be skimmed or split by subgoal.

Usage from an evaluation script:

    from src import log_context
    log_context.install()                        # after logging.basicConfig
    log_context.set_scene(1, n_scenes, scene_id)
    log_context.set_episode(1, n_episodes, episode_id)   # logs the episode banner
    log_context.set_subtask(1, n_subtasks, role, instruction)
    log_context.clear_subtask()                  # after a subgoal finishes
"""

import logging

_BANNER_WIDTH = 78

_state = {
    "scene_idx": None,
    "scene_total": None,
    "scene_id": None,
    "episode_idx": None,
    "episode_total": None,
    "episode_id": None,
    "subtask_idx": None,
    "subtask_total": None,
    "subtask_role": None,
}


# --------------------------------------------------------------------------- #
# context
# --------------------------------------------------------------------------- #
def _banner(body_lines, char):
    """Emit a boxed block, one logging call per line.

    Deliberately not a single multi-line record: the formatter only stamps the
    scene/episode/subgoal prefix on the first line of a record, so a multi-line banner
    would come out ragged and the separator rows would not be greppable.
    """
    logging.info("")
    logging.info(char * _BANNER_WIDTH)
    for line in body_lines:
        logging.info(line)
    logging.info(char * _BANNER_WIDTH)


def set_scene(idx, total, scene_id):
    _state.update(scene_idx=idx, scene_total=total, scene_id=scene_id)
    _reset_episode()
    _banner([f"  SCENE {idx}/{total}   {scene_id}"], "=")


def set_episode(idx, total, episode_id):
    _state.update(episode_idx=idx, episode_total=total, episode_id=episode_id)
    _reset_subtask()
    _banner(
        [
            f"  EPISODE {idx}/{total}   id={episode_id}",
            f"  scene {_state['scene_idx']}/{_state['scene_total']} {_state['scene_id']}",
        ],
        "=",
    )


def set_subtask(idx, total, role=None, instruction=None):
    _state.update(subtask_idx=idx, subtask_total=total, subtask_role=role)
    body = [f"  SUBGOAL {idx}/{total}" + (f"   role={role}" if role else "")]
    if instruction:
        body.append(f'  instruction: "{instruction}"')
    _banner(body, "-")


def end_subtask(result_line=None):
    """Close the current subgoal's block; `result_line` is shown inside it."""
    _banner([result_line] if result_line else [f"  SUBGOAL {_state['subtask_idx']} end"], "-")
    _reset_subtask()


def _reset_subtask():
    _state.update(subtask_idx=None, subtask_total=None, subtask_role=None)


def _reset_episode():
    _state.update(episode_idx=None, episode_total=None, episode_id=None)
    _reset_subtask()


clear_subtask = _reset_subtask


# --------------------------------------------------------------------------- #
# logging integration
# --------------------------------------------------------------------------- #
def prefix() -> str:
    """`[S1/1 E3/26 G2/5]`, dropping the parts that are not set yet."""
    parts = []
    if _state["scene_idx"] is not None:
        parts.append(f"S{_state['scene_idx']}/{_state['scene_total']}")
    if _state["episode_idx"] is not None:
        parts.append(f"E{_state['episode_idx']}/{_state['episode_total']}")
    if _state["subtask_idx"] is not None:
        parts.append(f"G{_state['subtask_idx']}/{_state['subtask_total']}")
    return f"[{' '.join(parts)}]" if parts else ""


class ContextFilter(logging.Filter):
    """Adds a `ctx` field to every record so the formatter can print it."""

    def __init__(self, width=18):
        super().__init__()
        self.width = width

    def filter(self, record):
        record.ctx = prefix().ljust(self.width)
        return True


def install(width=18):
    """Attach the filter to every root handler. Call after logging is configured."""
    ctx_filter = ContextFilter(width=width)
    for handler in logging.getLogger().handlers:
        handler.addFilter(ctx_filter)
    return ctx_filter
