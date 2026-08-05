from PIL import Image
import base64
from io import BytesIO
from typing import Optional
import logging
from src.const import *
from src.vlm_client import create_vlm_client
from src import prompts


# the backend (openai / ollama) is selected in src/const.py
client = create_vlm_client(max_tries=5, rate_limit_wait=3, error_wait=3)


# send information to the vlm
def call_vlm_api(sys_prompt, contents) -> Optional[str]:
    return client.call(sys_prompt, contents)


# encode tensor images to base64 format
def encode_tensor2base64(img):
    img = Image.fromarray(img)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    return img_base64


def format_question(step):
    question = step["question"]
    image_goal = None
    if "task_type" in step and step["task_type"] == "image":
        with open(step["image"], "rb") as image_file:
            image_goal = base64.b64encode(image_file.read()).decode("utf-8")

    return question, image_goal


def get_step_info(step, verbose=False):
    # 1 get question data
    question, image_goal = format_question(step)

    # 2 get step information(egocentric, frontier, snapshot)
    # 2.1 get egocentric views
    egocentric_imgs = []
    if step.get("use_egocentric_views", False):
        for egocentric_view in step["egocentric_views"]:
            egocentric_imgs.append(encode_tensor2base64(egocentric_view))

    # 2.2 get frontiers
    frontier_imgs = []
    for frontier in step["frontier_imgs"]:
        frontier_imgs.append(encode_tensor2base64(frontier))

    # 2.3 get snapshots
    snapshot_classes = {}  # rgb_id -> list of classes
    snapshot_full_imgs = {}  # rgb_id -> full img
    snapshot_crops = {}  # rgb_id -> list of crops
    snapshot_clusters = {}  # rgb_id -> list of clusters
    obj_map = step["obj_map"]
    seen_classes = set()
    for i, rgb_id in enumerate(step["snapshot_imgs"].keys()):
        snapshot_img = step["snapshot_imgs"][rgb_id]["full_img"]
        snapshot_full_imgs[rgb_id] = encode_tensor2base64(snapshot_img)
        snapshot_crops[rgb_id] = [
            encode_tensor2base64(crop_data["crop"])
            for crop_data in step["snapshot_imgs"][rgb_id]["object_crop"]
        ]
        snapshot_class = [
            crop_data["obj_class"]
            for crop_data in step["snapshot_imgs"][rgb_id]["object_crop"]
        ]
        cluster_class = [
            obj_map[int(obj_id)] for obj_id in step["snapshot_objects"][rgb_id]
        ]
        # remove duplicates
        seen_classes.update(sorted(list(set(snapshot_class))))
        snapshot_classes[rgb_id] = snapshot_class
        snapshot_clusters[rgb_id] = cluster_class

    # 3 prefiltering, note that we need the obj_id_mapping
    keep_index = list(range(len(snapshot_full_imgs)))
    keep_index_snapshot = {
        rgb_id: list(range(len(snapshot_crops[rgb_id]))) for rgb_id in snapshot_crops
    }
    if step.get("use_prefiltering") is True:
        use_full_obj_list = step["use_full_obj_list"]
        n_prev_snapshot = len(snapshot_full_imgs)
        snapshot_classes, keep_index, keep_index_snapshot = prefiltering(
            question,
            snapshot_classes,
            snapshot_clusters,
            seen_classes,
            step["top_k_categories"],
            image_goal,
            use_full_obj_list,
            verbose=verbose,
            prompt_version=step.get("prompt_version"),
            history=step.get("history"),
        )
        snapshot_full_imgs = {
            rgb_id: snapshot_full_imgs[rgb_id] for rgb_id in keep_index_snapshot.keys()
        }
        for rgb_id in snapshot_classes.keys():
            snapshot_crops[rgb_id] = [
                snapshot_crops[rgb_id][i] for i in keep_index_snapshot[rgb_id]
            ]
        if verbose:
            logging.info(
                f"Prefiltering snapshot: {n_prev_snapshot} -> {len(snapshot_full_imgs)}"
            )

    return (
        question,
        image_goal,
        egocentric_imgs,
        frontier_imgs,
        snapshot_full_imgs,
        snapshot_classes,
        snapshot_crops,
        keep_index,
        keep_index_snapshot,
    )


# The prompt text itself lives in src/prompts/. These two keep working for anything
# that imports them directly (e.g. vlm_smoke_test.py) and always use the baseline;
# the evaluators go through the selected version instead -- see explore_step below.
def format_explore_prompt(*args, **kwargs):
    return prompts.get("default").format_explore_prompt(*args, **kwargs)


def format_prefiltering_prompt(*args, **kwargs):
    return prompts.get("default").format_prefiltering_prompt(*args, **kwargs)


def get_prefiltering_classes(
    question, seen_classes, top_k=10, image_goal=None, prompt_version=None, history=None
):
    prompt_version = prompt_version or prompts.get()
    prefiltering_sys, prefiltering_content = prompt_version.format_prefiltering_prompt(
        question,
        sorted(list(seen_classes)),
        top_k=top_k,
        image_goal=image_goal,
        history=history,
    )

    message = ""
    for c in prefiltering_content:
        message += c[0]
        if len(c) == 2:
            message += f": image {c[1][:10]}..."
    response = call_vlm_api(prefiltering_sys, prefiltering_content)
    if response is None:
        return []

    # parse the response and return the top_k objects
    selected_classes = response.strip().split("\n")
    selected_classes = [cls.strip() for cls in selected_classes]
    selected_classes = [cls for cls in selected_classes if cls in seen_classes]
    selected_classes = selected_classes[:top_k]

    return selected_classes


def prefiltering(
    question,
    snapshot_classes,
    snapshot_clusters,
    seen_classes,
    top_k=10,
    image_goal=None,
    use_full_obj_list=False,
    verbose=False,
    prompt_version=None,
    history=None,
):
    selected_classes = get_prefiltering_classes(
        question,
        seen_classes,
        top_k,
        image_goal,
        prompt_version=prompt_version,
        history=history,
    )
    if verbose:
        logging.info(f"Prefiltering selected classes: {selected_classes}")

    keep_index = [
        i
        for i, k in enumerate(snapshot_clusters.keys())
        if len(set(snapshot_clusters[k]) & set(selected_classes)) > 0
    ]
    keep_snapshot_id = [list(snapshot_classes.keys())[i] for i in keep_index]
    snapshot_classes = {rgb_id: snapshot_classes[rgb_id] for rgb_id in keep_snapshot_id}

    keep_index_snapshot = {}
    for rgb_id in keep_snapshot_id:
        keep_index_snapshot[rgb_id] = [
            i
            for i in range(len(snapshot_classes[rgb_id]))
            if snapshot_classes[rgb_id][i] in selected_classes
        ]
        snapshot_classes[rgb_id] = [
            snapshot_classes[rgb_id][i] for i in keep_index_snapshot[rgb_id]
        ]

    return snapshot_classes, keep_index, keep_index_snapshot


def explore_step(step, cfg, verbose=False):
    step["use_prefiltering"] = cfg.prefiltering
    step["top_k_categories"] = cfg.top_k_categories
    # Resolved once per step and carried in `step` so prefiltering, which runs inside
    # get_step_info, uses the same version as the explore prompt below.
    step["prompt_version"] = prompts.get(cfg.get("prompt_version", None))
    (
        question,
        image_goal,
        egocentric_imgs,
        frontier_imgs,
        snapshot_full_imgs,
        snapshot_classes,
        snapshot_crops,
        snapshot_id_mapping,
        snapshot_crop_mapping,
    ) = get_step_info(step, verbose)
    sys_prompt, content = step["prompt_version"].format_explore_prompt(
        question,
        egocentric_imgs,
        frontier_imgs,
        snapshot_full_imgs,
        snapshot_classes,
        snapshot_crops,
        egocentric_view=step.get("use_egocentric_views", False),
        use_snapshot_class=True,
        image_goal=image_goal,
        history=step.get("history"),
    )

    if verbose:
        logging.info(f"Input prompt:")
        message = sys_prompt
        for c in content:
            message += c[0]
            if len(c) == 2:
                message += f"[{c[1][:10]}...]"
        logging.info(message)

    retry_bound = 3
    final_response = None
    final_reason = None
    for _ in range(retry_bound):
        response = call_vlm_api(sys_prompt, content)

        if response is None:
            print("call_vlm_api returns None, retrying")
            continue

        response = response.strip()
        if "\n" in response:
            response = response.split("\n")
            response, reason = response[0], response[-1]
        else:
            reason = ""
        response = response.lower()
        try:
            choice_type, choice_id = response.split(",")[0].strip().split(" ")
        except Exception as e:
            print(f"Error in splitting response: {response}")
            print(e)
            continue

        response_valid = False
        if (
            choice_type == "snapshot"
            and choice_id.isdigit()
            and 0 <= int(choice_id) < len(snapshot_full_imgs)
        ):
            try:
                object_choice_type, object_choice_id = (
                    response.split(",")[1].strip().split(" ")
                )
            except Exception as e:
                print(f"Error in splitting response: {response}")
                print(e)
                continue
            if (
                object_choice_type == "object"
                and object_choice_id.isdigit()
                and 0
                <= int(object_choice_id)
                < len(list(snapshot_crop_mapping.values())[int(choice_id)])
            ):
                response_valid = True
        elif (
            choice_type == "frontier"
            and choice_id.isdigit()
            and 0 <= int(choice_id) < len(frontier_imgs)
        ):
            response_valid = True

        if response_valid:
            final_response = response
            final_reason = reason
            break

    return (
        final_response,
        snapshot_id_mapping,
        snapshot_crop_mapping,
        final_reason,
        len(snapshot_full_imgs),
    )
