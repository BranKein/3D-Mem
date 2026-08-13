"""history_included, plus explicit rules for choosing a snapshot over a frontier.

The baseline defines the two actions but never states when to take which, and it never
says that a snapshot choice is final. Both gaps were visible in a qwen3.5:9b run:

    instruction: "Find the clothes."
    Prefiltering selected classes: ['cabinet', 'bed', 'chair']
    Snapshot 0: chair | Snapshot 1: bed, cabinet | Snapshot 2: bed
    Response: snapshot 2, object 0        -> bed
    SUBGOAL 2/5 done -- success_by_distance=False found=bed

No snapshot held clothes, so the only correct action was a frontier. Picking a bed for
"clothes" is a reasonable guess if the task reads as "choose the most plausible option"
-- clothes are often on a bed -- and nothing in the prompt says otherwise. The baseline
only implies it, in the definition of what choosing a snapshot means.

It cost the whole subgoal, because a snapshot choice terminates it:
run_refonbench_evaluation.py sets ``task_success = True`` and breaks once the agent
reaches a chosen snapshot, so that one call at step 0 ended the subgoal before any
exploration happened. The model has no way to know that from the prompt.

Kept apart from history_included rather than folded into it, so the two changes stay
attributable: one supplies the referent, this one states the action rules.
"""

from src.prompts import register
from src.prompts.history_included import HistoryIncludedPrompt


_ACTION_RULES = (
    "How to choose:\n"
    "- Choose a Snapshot only when you can see the object the question asks for in "
    "that snapshot. Choosing one submits that object as your final answer for this "
    "subgoal and ends it -- you will not get to revise it, and no further exploration "
    "happens.\n"
    "- Choose a Frontier when none of the snapshots contains the object asked for. "
    "This is the right answer even when a snapshot holds something related or "
    "plausibly nearby: an object that is merely associated with the target is wrong, "
    "and exploring costs you nothing but a step.\n"
    "- If no snapshot is available at all, choose a Frontier.\n"
)


@register("action_rules")
class ActionRulesPrompt(HistoryIncludedPrompt):
    name = "action_rules"
    description = "history_included + explicit snapshot-vs-frontier rules"

    def format_explore_prompt(self, question, *args, **kwargs):
        sys_prompt, content = super().format_explore_prompt(question, *args, **kwargs)

        # Immediately before the answer-format block, so the rules are the last thing
        # read before answering and every snapshot/frontier image has already been
        # seen. Anchoring there also keeps the block outside the snapshot and frontier
        # runs, which must stay contiguous or the indices in the answer shift.
        insert_at = next(
            (
                i
                for i, c in enumerate(content)
                if c[0].startswith("Please provide your answer in the following format")
            ),
            len(content),
        )
        content.insert(insert_at, (_ACTION_RULES,))
        return sys_prompt, content
