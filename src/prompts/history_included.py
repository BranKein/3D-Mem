"""Prompt version that carries the episode's earlier subgoals into both prompts.

RefON instructions are anaphoric: "Find A1.", "Find the 2nd one again.", "Find the one
before that." refer to objects bound by *earlier subgoals of the same episode*, and the
baseline prompts show none of that. Two separate failures follow, and fixing only one
does nothing:

1. explore -- the model has no referent. Observed with qwen2.5vl:7b on
   "Find the 2nd one again. Let's call it A1.": it read "the 2nd one" as the second
   instance of some category in the room, and treated "A1" as an unknown object to go
   looking for, when A1 is a name being bound right now.

2. prefiltering -- the baseline asks the model to pick relevant categories out of the
   question, but an anaphoric instruction contains no category noun. The reply comes
   back empty, and eval_utils_gpt_goatbench.prefiltering() keeps only snapshots whose
   cluster intersects the selected classes, so an empty selection drops *every*
   snapshot. The step prompt is then left with frontiers only and the agent is forced
   to explore no matter what it would have preferred. In the observed log:
   "Prefiltering selected classes: [] -> Prefiltering snapshot: 9 -> 0".

So the history goes into the prefiltering prompt as well, where it lets the model map
"the 2nd one" back to a concrete class name that exists in the candidate list.
"""

from src.prompts import register
from src.prompts.default import DefaultPrompt

_REFERENCE_NOTE = (
    "The question may refer back to an earlier subgoal instead of naming an object: "
    "an alias bound earlier (\"A1\"), an ordinal over the list above "
    "(\"the 2nd one\"), or a relative reference (\"the one before that\"). "
    "Resolve it against that list. A name introduced by "
    "\"Let's call it X\" is being assigned to the object you find in THIS subgoal -- "
    "it is not an object to search for.\n"
)


def _history_lines(history):
    """`history` -> numbered lines, or None when there is nothing to show yet."""
    if not history:
        return None
    lines = []
    for item in history:
        order = item.get("order", len(lines) + 1)
        found = item.get("found_class")
        outcome = f'you selected: {found}' if found else "you did not settle on an object"
        lines.append(f'  {order}. "{item.get("instruction", "")}" -> {outcome}')
    return "\n".join(lines)


@register("history_included")
class HistoryIncludedPrompt(DefaultPrompt):
    name = "history_included"
    description = "default + the episode's earlier subgoals, in both prompts"

    def format_explore_prompt(self, question, *args, history=None, **kwargs):
        sys_prompt, content = super().format_explore_prompt(
            question, *args, history=history, **kwargs
        )

        lines = _history_lines(history)
        if lines is None:
            block = (
                "This is the first subgoal of the episode, so there is no earlier "
                "subgoal to refer back to.\n"
            )
        else:
            block = (
                "Earlier subgoals in this episode, in order:\n"
                + lines
                + "\n"
                + _REFERENCE_NOTE
            )

        # Insert right after the question, before "Select the Frontier/Snapshot ...",
        # so the referent is known before the instruction to choose. The question is
        # content[0] (and content[1] when an image goal splits it), so anchor on the
        # "Select the" line rather than a fixed index -- the snapshot and frontier
        # blocks further down must stay contiguous or the answer indices shift.
        insert_at = next(
            (
                i
                for i, c in enumerate(content)
                if c[0].startswith("Select the Frontier/Snapshot")
            ),
            len(content),
        )
        content.insert(insert_at, (block,))
        return sys_prompt, content

    def format_prefiltering_prompt(
        self, question, class_list, top_k=10, image_goal=None, history=None
    ):
        sys_prompt, content = super().format_prefiltering_prompt(
            question, class_list, top_k=top_k, image_goal=image_goal, history=history
        )

        lines = _history_lines(history)
        if lines is None:
            return sys_prompt, content

        block = (
            "Context: objects you already found in this episode, in order:\n"
            + lines
            + "\n"
            + "If the question refers back to one of them instead of naming a category "
            "(\"A1\", \"the 2nd one\", \"the one before that\"), work out which entry it "
            "means and rank that entry's class name first. Still answer with class "
            "names taken from the list below, one per line, and nothing else.\n"
        )

        # Before the "Following is the concrete content of the task" block, so the
        # context is in place when the question is read.
        insert_at = next(
            (
                i
                for i, c in enumerate(content)
                if c[0].startswith("Following is the concrete content")
            ),
            len(content),
        )
        content.insert(insert_at, (block,))
        return sys_prompt, content
