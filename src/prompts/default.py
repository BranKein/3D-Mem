"""The original 3D-Mem prompts, moved here verbatim from eval_utils_gpt_goatbench.py.

This is the baseline every other version is compared against, so the wording must not
drift. To try a change, add a new version rather than editing this one.
"""

from src.prompts import register
from src.prompts.base import PromptVersion


@register("default")
class DefaultPrompt(PromptVersion):
    name = "default"
    description = "original 3D-Mem prompts (baseline)"

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
    ):
        sys_prompt = "Task: You are an agent in an indoor scene that is able to observe the surroundings and explore the environment. You are tasked with indoor navigation, and you are required to choose either a Snapshot or a Frontier image to explore and find the target object required in the question.\n"

        content = []
        # 1 here is some basic info
        text = "Definitions:\n"
        text += (
            "Snapshot: A focused observation of several objects. It contains a full image of the cluster of objects, and separate image crops of each object. "
            + "Choosing a snapshot means that the object asked in the question is within the cluster of objects that the snapshot represents, and you will choose that object as the final answer of the question. "
            + "Therefore, if you choose a snapshot, you should also choose the object in the snapshot that you think is the answer to the question.\n"
        )
        text += "Frontier: An unexplored region that could potentially lead to new information for answering the question. Selecting a frontier means that you will further explore that direction.\n"

        # 2 here is the question
        text += f"Question: {question}"
        if image_goal is not None:
            content.append((text, image_goal))
            content.append(("\n",))
        else:
            content.append((text + "\n",))

        text = "Select the Frontier/Snapshot that would help find the answer of the question.\n"
        content.append((text,))

        # 3 here is the egocentric views
        if egocentric_view:
            text = (
                "The following is the egocentric view of the agent in forward direction: "
            )
            content.append((text, egocentric_imgs[-1]))
            content.append(("\n",))

        # 4 here is the snapshot images
        text = "The followings are all the snapshots that you can choose. Following each snapshot image are the class name and image crop of each object contained in the snapshot.\n"
        text += "Please note that the class name may not be accurate due to the limitation of the object detection model. "
        text += "So you still need to utilize the images to make the decision.\n"
        content.append((text,))
        if len(snapshot_imgs) == 0:
            content.append(("No Snapshot is available\n",))
        else:
            for i, rgb_id in enumerate(snapshot_imgs.keys()):
                content.append((f"Snapshot {i} ", snapshot_imgs[rgb_id]))
                for j in range(len(snapshot_crops[rgb_id])):
                    content.append(
                        (
                            f"Object {j}: {snapshot_classes[rgb_id][j]}",
                            snapshot_crops[rgb_id][j],
                        )
                    )
                content.append(("\n",))

        # 5 here is the frontier images
        text = "The followings are all the Frontiers that you can explore: \n"
        content.append((text,))
        if len(frontier_imgs) == 0:
            content.append(("No Frontier is available\n",))
        else:
            for i in range(len(frontier_imgs)):
                content.append((f"Frontier {i} ", frontier_imgs[i]))
                content.append(("\n",))

        # 6 here is the format of the answer
        text = "Please provide your answer in the following format: 'Snapshot i, Object j' or 'Frontier i', where i, j are the index of the snapshot or frontier you choose. "
        text += "For example, if you choose the fridge in the first snapshot, please return 'Snapshot 0, Object 2', where 2 is the index of the fridge in that snapshot.\n"
        text += "You can explain the reason for your choice, but put it in a new line after the choice.\n"
        content.append((text,))

        return sys_prompt, content

    def format_prefiltering_prompt(self, question, class_list, top_k=10, image_goal=None):
        content = []
        sys_prompt = "You are an AI agent in a 3D indoor scene.\n"
        prompt = "Your goal is to answer questions about the scene through exploration.\n"
        prompt += "To efficiently solve the problem, you should first rank objects in the scene based on their importance.\n"
        prompt += "These are the rules for the task.\n"
        prompt += "1. Read through the whole object list.\n"
        prompt += "2. Rank objects in the list based on how well they can help your exploration given the question.\n"
        prompt += f"3. Reprint the name of all objects that may help your exploration given the question. "
        prompt += "4. Do not print any object not included in the list or include any additional information in your response.\n"
        content.append((prompt,))
        # ------------------format an example-------------------------
        prompt = "Here is an example of selecting helpful objects:\n"
        prompt += "Question: What can I use to watch my favorite shows and movies?\n"
        prompt += (
            "Following is a list of objects that you can choose, each object one line\n"
        )
        prompt += "painting\nspeaker\nbox\ncabinet\nlamp\ntv\nbook rack\nsofa\noven\nbed\ncurtain\n"
        prompt += "Answer: tv\nspeaker\nsofa\nbed\n"
        content.append((prompt,))
        # ------------------Task to solve----------------------------
        prompt = f"Following is the concrete content of the task and you should retrieve helpful objects in order:\n"
        prompt += f"Question: {question}"
        if image_goal is not None:
            content.append((prompt, image_goal))
            content.append(("\n",))
        else:
            content.append((prompt + "\n",))
        prompt = (
            "Following is a list of objects that you can choose, each object one line\n"
        )
        for i, cls in enumerate(class_list):
            prompt += f"{cls}\n"
        prompt += "Answer: "
        content.append((prompt,))
        return sys_prompt, content
