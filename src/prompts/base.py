"""Interface every prompt version implements.

A prompt version owns the two prompts the evaluators send to the VLM:

  format_explore_prompt      the navigation step -- pick a Snapshot or a Frontier
  format_prefiltering_prompt the category shortlist that keeps the step prompt small

Both return ``(sys_prompt, content)``, where ``content`` is the list of tuples the
VLM client expects: ``(text,)`` for text, ``(text, base64_png)`` to attach an image
after that text. See src/vlm_client.py.

``history`` is the list of subgoals already finished in this episode, oldest first::

    [{"order": 1, "role": "S", "instruction": "Find the toilet.",
      "found_class": "toilet", "arrived": True}, ...]

``found_class`` is what the agent itself settled on, not the ground truth, so a version
that shows it is not leaking the answer. It is None when the subgoal ended without
committing to an object. Versions that do not need it simply ignore the argument.

To add a version, subclass this (or DefaultPrompt, to change only one prompt), decorate
with @register("<name>"), and import the module in src/prompts/__init__.py.
"""


class PromptVersion:
    #: identifier used by --prompt-version / cfg.prompt_version
    name: str = ""
    #: one-line description, shown by --list-prompt-versions
    description: str = ""

    def format_explore_prompt(
        self,
        question,
        egocentric_imgs,
        frontier_imgs,
        snapshot_imgs,
        snapshot_classes,
        snapshot_crops,
        egocentric_view=False,
        use_snapshot_class=True,
        image_goal=None,
        history=None,
    ):
        raise NotImplementedError

    def format_prefiltering_prompt(
        self, question, class_list, top_k=10, image_goal=None, history=None
    ):
        raise NotImplementedError

    def __repr__(self):
        return f"<PromptVersion {self.name!r}>"
