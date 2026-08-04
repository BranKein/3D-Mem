"""Prompt version that will carry the exploration history into the step prompt.

Currently a byte-for-byte copy of `default` -- it inherits both prompts unchanged, so
running with `--prompt-version history_included` today produces exactly the same input
as the baseline. That is deliberate: it fixes the plumbing (registry entry, CLI flag,
logging) so the only thing left to change is the prompt text itself.

To start diverging, override one or both methods, e.g.

    def format_explore_prompt(self, question, ..., image_goal=None):
        sys_prompt, content = super().format_explore_prompt(
            question, ..., image_goal=image_goal
        )
        content.insert(<position>, ("Previously visited: ...\n",))
        return sys_prompt, content

Note that `content` is ordered, and images bind to the text tuple they ship with, so
inserting in the middle shifts what the model sees -- keep the "Snapshot i" / "Frontier i"
blocks contiguous or the indices in the answer stop lining up.
"""

from src.prompts import register
from src.prompts.default import DefaultPrompt


@register("history_included")
class HistoryIncludedPrompt(DefaultPrompt):
    name = "history_included"
    description = "same as default for now; will add exploration history"
