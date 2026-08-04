"""Logger for the RefON referential-navigation benchmark.

Everything about result bookkeeping, SPL, and visualisation is identical to GOAT-Bench,
so this only overrides how a subtask is turned into a prompt: a RefON subtask already
carries its own natural-language instruction, and its target is one specific instance
rather than a whole category.
"""

from src.logger_goatbench import Logger as GoatBenchLogger


class Logger(GoatBenchLogger):
    def init_subtask(
        self,
        subtask_id,
        role,
        instruction,
        subtask_goal,
        pts,
        scene,
        tsdf_planner,
    ):
        # The parent builds viewpoints, voxel positions and the GT explore distance for
        # us. Present the subtask to it as a "description" goal: that is the branch that
        # targets a single named instance, which is what a referential subtask needs
        # ("object" would accept any instance of the category and make every back
        # reference trivially satisfiable). It reads lang_desc, so supply it.
        goal_with_desc = [dict(goal, lang_desc=instruction) for goal in subtask_goal]
        subtask_metadata = super().init_subtask(
            subtask_id=subtask_id,
            goal_type="description",
            subtask_goal=goal_with_desc,
            pts=pts,
            scene=scene,
            tsdf_planner=tsdf_planner,
        )

        # The RefON instruction is the prompt itself -- keep it verbatim instead of
        # wrapping it in GOAT's "object exactly described as ..." template.
        subtask_metadata["question"] = instruction
        # Report per-role rather than per-goal-type results, so success/SPL break down
        # by referential mechanism (S / AB_pre / AB_post / AR_* / OR_post / ...).
        subtask_metadata["task_type"] = role
        subtask_metadata["role"] = role
        subtask_metadata["instruction"] = instruction
        return subtask_metadata
