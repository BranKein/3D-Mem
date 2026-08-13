"""Smoke test for a candidate local VLM against the real 3D-Mem prompts.

Run from the repo root, e.g.:
    VLM_PROVIDER=ollama OLLAMA_MODEL=gemma4:26b-a4b-it-qat \
        ~/anaconda3/envs/3dmem/bin/python vlm_smoke_test.py
    VLM_PROVIDER=vllm VLLM_MODEL=Qwen3.5-9B \
        ~/anaconda3/envs/3dmem/bin/python vlm_smoke_test.py

Checks, in order:
  0. server reachable, model loaded, effective context length
  1. label<->image alignment: does the model bind "Snapshot i" to the i-th image
  2. real explore prompt (real snapshot/frontier PNGs) -> does the answer parse
     with the repo's own parsing logic, and how long does a step take
  3. prefiltering prompt (text only) -> does it return a bare class list
"""

import base64
import glob
import json
import os
import subprocess
import sys
import time
from io import BytesIO

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.const import (
    OLLAMA_END_POINT,
    OLLAMA_MODEL,
    VLLM_END_POINT,
    VLLM_MODEL,
    VLM_PROVIDER,
)
from src.eval_utils_gpt_goatbench import (
    call_vlm_api,
    format_explore_prompt,
    format_prefiltering_prompt,
)

PROMPT_HW = 360  # matches prompt_h / prompt_w in cfg/eval_*.yaml
RESULTS_DIR = "results/exp_eval_aeqa/00c2be2a-1377-4fae-a889-30936b7890c3"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def to_b64(img: Image.Image) -> str:
    img = img.convert("RGB").resize((PROMPT_HW, PROMPT_HW))
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def load_b64(path: str) -> str:
    return to_b64(Image.open(path))


def shape_img(color: str, shape: str) -> Image.Image:
    """A big unambiguous colored shape on white -- used for the alignment probe."""
    img = Image.new("RGB", (512, 512), "white")
    d = ImageDraw.Draw(img)
    box = (96, 96, 416, 416)
    if shape == "circle":
        d.ellipse(box, fill=color)
    elif shape == "square":
        d.rectangle(box, fill=color)
    else:  # triangle
        d.polygon([(256, 96), (416, 416), (96, 416)], fill=color)
    return img


def vram_used_mb() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    return int(out.stdout.strip().splitlines()[0])


def backend_id() -> tuple:
    """(model, endpoint) for whichever local backend is configured."""
    if VLM_PROVIDER.lower() == "vllm":
        return VLLM_MODEL, VLLM_END_POINT
    return OLLAMA_MODEL, OLLAMA_END_POINT


def loaded_model() -> str:
    """What the server actually has resident, when it can say.

    `/api/ps` is an ollama extension. vLLM has no equivalent -- it loads one model at
    startup and holds it, and its own startup log is where the KV cache size and the
    effective context length are reported.
    """
    if VLM_PROVIDER.lower() != "ollama":
        return (
            f"  ({VLM_PROVIDER} has no /api/ps; the vram reading above and the server's "
            f"own startup log are the check)"
        )
    import urllib.request
    try:
        with urllib.request.urlopen(f"{OLLAMA_END_POINT}/api/ps", timeout=10) as r:
            data = json.load(r)
    except Exception as e:
        return f"  (could not read /api/ps: {e})"
    if not data.get("models"):
        return "  (no model currently loaded)"
    lines = []
    for m in data["models"]:
        size = m.get("size", 0) / 1e9
        vram = m.get("size_vram", 0) / 1e9
        ctx = m.get("context_length", "?")
        offload = "100% GPU" if vram >= size * 0.99 else f"{vram / size * 100:.0f}% GPU (CPU offload!)"
        lines.append(
            f"  {m['name']}: total {size:.1f}GB, vram {vram:.1f}GB -> {offload}, context_length={ctx}"
        )
    return "\n".join(lines)


def parse_choice(response: str):
    """The exact parsing used in eval_utils_gpt_goatbench.explore_step."""
    response = response.strip()
    if "\n" in response:
        parts = response.split("\n")
        response, reason = parts[0], parts[-1]
    else:
        reason = ""
    response = response.lower()
    try:
        choice_type, choice_id = response.split(",")[0].strip().split(" ")
    except Exception as e:
        return None, None, None, f"split error: {e}"
    obj_id = None
    if choice_type == "snapshot":
        try:
            object_choice_type, object_choice_id = response.split(",")[1].strip().split(" ")
        except Exception as e:
            return choice_type, choice_id, None, f"object split error: {e}"
        if object_choice_type != "object":
            return choice_type, choice_id, None, "second field is not 'object'"
        obj_id = object_choice_id
    return choice_type, choice_id, obj_id, None


# --------------------------------------------------------------------------- #
# test 1: label <-> image alignment
# --------------------------------------------------------------------------- #
def test_alignment():
    print("\n=== TEST 1: Snapshot label <-> image alignment ===")
    palette = [("red", "circle"), ("green", "square"), ("blue", "triangle"), ("orange", "circle")]
    ok = 0
    trials = [(2, "green square"), (0, "blue triangle"), (3, "red circle")]
    for target_idx, target_name in trials:
        color, shape = target_name.split()
        # build 4 snapshots; the target shape sits at target_idx, the others are fillers
        fillers = [p for p in palette if f"{p[0]} {p[1]}" != target_name]
        imgs, crops, classes = {}, {}, {}
        f = 0
        for i in range(4):
            if i == target_idx:
                c, s = color, shape
            else:
                c, s = fillers[f % len(fillers)]
                f += 1
            im = to_b64(shape_img(c, s))
            imgs[i] = im
            crops[i] = [im]
            # deliberately uninformative class names: only the image can answer it
            classes[i] = ["object"]

        sys_p, content = format_explore_prompt(
            question=f"Find the {target_name}.",
            egocentric_imgs=None,
            frontier_imgs=[],
            snapshot_imgs=imgs,
            snapshot_classes=classes,
            snapshot_crops=crops,
            egocentric_view=False,
        )
        t0 = time.time()
        resp = call_vlm_api(sys_p, content)
        dt = time.time() - t0
        if resp is None:
            print(f"  target=Snapshot {target_idx} ({target_name}): NO RESPONSE ({dt:.1f}s)")
            continue
        ctype, cid, oid, err = parse_choice(resp)
        got = f"{ctype} {cid}"
        hit = ctype == "snapshot" and cid == str(target_idx)
        ok += hit
        print(f"  target=Snapshot {target_idx} ({target_name}): got '{got}' "
              f"{'OK' if hit else 'MISMATCH'} ({dt:.1f}s){' | ' + err if err else ''}")
        print(f"    raw: {resp.strip()[:160]!r}")
    print(f"  --> alignment {ok}/{len(trials)}")
    return ok, len(trials)


# --------------------------------------------------------------------------- #
# test 2: real explore prompt
# --------------------------------------------------------------------------- #
def test_real_explore(n_snapshots=4, n_crops=3, n_frontiers=4, repeats=3):
    print(f"\n=== TEST 2: real explore prompt "
          f"({n_snapshots} snapshots x {n_crops} crops + {n_frontiers} frontiers) ===")
    snaps = sorted(glob.glob(os.path.join(RESULTS_DIR, "snapshot", "*.png")))
    fronts = sorted(glob.glob(os.path.join(RESULTS_DIR, "frontier", "*.png")))
    if not snaps or not fronts:
        print(f"  SKIP: no saved images under {RESULTS_DIR}")
        return 0, 0

    imgs, crops, classes = {}, {}, {}
    cursor = 0
    names = ["chair", "table", "sofa", "cabinet", "picture", "lamp", "bed", "sink"]
    for i in range(n_snapshots):
        imgs[i] = load_b64(snaps[cursor]); cursor += 1
        crops[i] = [load_b64(snaps[cursor + k]) for k in range(n_crops)]
        cursor += n_crops
        classes[i] = [names[(i * n_crops + k) % len(names)] for k in range(n_crops)]
    frontier_imgs = [load_b64(p) for p in fronts[:n_frontiers]]

    n_images = n_snapshots * (1 + n_crops) + n_frontiers
    print(f"  {n_images} images per prompt, each {PROMPT_HW}x{PROMPT_HW}")

    sys_p, content = format_explore_prompt(
        question="Where is the sofa in the living room?",
        egocentric_imgs=[frontier_imgs[0]],
        frontier_imgs=frontier_imgs,
        snapshot_imgs=imgs,
        snapshot_classes=classes,
        snapshot_crops=crops,
        egocentric_view=True,
    )

    ok, times = 0, []
    for r in range(repeats):
        base_vram = vram_used_mb()
        t0 = time.time()
        resp = call_vlm_api(sys_p, content)
        dt = time.time() - t0
        times.append(dt)
        peak_vram = vram_used_mb()
        if resp is None:
            print(f"  run {r + 1}: NO RESPONSE after retries ({dt:.1f}s)")
            continue
        ctype, cid, oid, err = parse_choice(resp)
        valid = err is None and ctype in ("snapshot", "frontier") and (cid or "").isdigit()
        in_range = valid and (
            (ctype == "snapshot" and 0 <= int(cid) < n_snapshots and (oid or "").isdigit()
             and 0 <= int(oid) < n_crops)
            or (ctype == "frontier" and 0 <= int(cid) < n_frontiers)
        )
        ok += bool(in_range)
        print(f"  run {r + 1}: {dt:.1f}s | parse={'OK' if in_range else 'FAIL'}"
              f"{' (' + err + ')' if err else ''} | vram {base_vram}->{peak_vram} MiB")
        print(f"    raw: {resp.strip()[:200]!r}")
    if times:
        print(f"  --> format {ok}/{repeats}, latency avg {sum(times) / len(times):.1f}s "
              f"(min {min(times):.1f} / max {max(times):.1f})")
    return ok, repeats


# --------------------------------------------------------------------------- #
# test 3: prefiltering prompt (text only)
# --------------------------------------------------------------------------- #
def test_prefiltering():
    print("\n=== TEST 3: prefiltering prompt (text only) ===")
    class_list = ["chair", "table", "sofa", "cabinet", "picture", "lamp", "bed",
                  "sink", "toilet", "tv", "refrigerator", "oven", "curtain", "book"]
    sys_p, content = format_prefiltering_prompt(
        "What can I use to cook dinner?", class_list, top_k=10
    )
    t0 = time.time()
    resp = call_vlm_api(sys_p, content)
    dt = time.time() - t0
    if resp is None:
        print(f"  NO RESPONSE ({dt:.1f}s)")
        return 0, 1
    lines = [c.strip() for c in resp.strip().split("\n") if c.strip()]
    kept = [c for c in lines if c in class_list]
    clean = len(kept) == len(lines) and len(kept) > 0
    print(f"  {dt:.1f}s | {len(lines)} lines, {len(kept)} valid class names "
          f"-> {'OK' if clean else 'DIRTY (extra prose in output)'}")
    print(f"    raw: {resp.strip()[:200]!r}")
    return int(clean), 1


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    _model, _end_point = backend_id()
    print(f"provider: {VLM_PROVIDER}")
    print(f"model   : {_model}")
    print(f"endpoint: {_end_point}")
    print(f"vram before load: {vram_used_mb()} MiB")

    results = []
    results.append(("alignment", *test_alignment()))
    print("\n--- loaded model after first call ---")
    print(loaded_model())
    results.append(("explore format", *test_real_explore()))
    results.append(("prefiltering", *test_prefiltering()))

    print("\n================ SUMMARY ================")
    for name, ok, total in results:
        print(f"  {name:<16} {ok}/{total}")
    print("\n--- loaded model at the end ---")
    print(loaded_model())
