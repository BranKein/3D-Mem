# RefONEpisodeGenerator

A pipeline that generates episodes for an HM3D-based **Referential Object-goal
Navigation (MultiON)** benchmark. It does not *run* the benchmark (like goat-bench);
it **produces the episodes (shard json)** the benchmark consumes.

The point of this benchmark is not simply finding N objects with N instructions, but a
**referential** structure where a later goal must be inferred by referring back to an
earlier instruction (e.g. "Find the chair. Let's call it A1." → … → "Find A1.").

The referencing mechanisms and grammar (a one-counter pushdown automaton) are defined
in [`episode_generator_spec.md`](./episode_generator_spec.md).

---

## Design overview (OOP)

The procedural algorithm of spec section 7 is **not used**. Instead, **each instruction
style (role) is a class**, which keeps it extensible: adding a new style does not touch
the generator / balancer / builder.

```
refon/
├── automaton.py        # GenContext: pushdown automaton state (n_AB, ar_mode, v) + bookkeeping
├── style_token.py      # StyleToken / StyleList — command 1 output (structure only)
├── styles/             # ⭐ instruction style class hierarchy (extension point)
│   ├── base.py         #   InstructionStyle ABC: admissible/apply/resolve/render
│   ├── plain.py        #   S
│   ├── alias_binding.py#   AB_pre, AB_post
│   ├── ordinal.py      #   OR_post
│   ├── anaphoric.py    #   AR_pre, AR_post
│   ├── multi_role.py   #   {AB_pre, OR_post}
│   ├── goal_absent.py  #   GA_unbound_alias / GA_absent_object / GA_invalid_ordinal
│   └── __init__.py     #   registry (register())
├── config.py           # GeneratorConfig / BuilderConfig
├── generator.py        # single style list generation (drives the automaton)
├── balancer.py         # generate -> measure -> prune -> refill, to hit target ratios
├── runner.py           # resumable, per-scene orchestration of command 1
├── progress.py         # checkpoint file (_progress.json) for resumability
├── statistics.py       # statistics accumulation/formatting
├── validate.py         # re-validate a style list (automaton replay)
├── storage.py          # folder save/load
├── episode_builder.py  # command 2: style list → HM3D episode json (uses HM3DSceneLoader)
└── cli.py              # argparse CLI
```

What each `InstructionStyle` class owns:

| method | when | role |
|---|---|---|
| `admissible(ctx)` | generation | can this style appear here (automaton state)? (constraints) |
| `apply(ctx, rng)` | generation | automaton transition + fix the reference structure → `StyleToken` |
| `resolve(token, b)` | build | map the reference structure onto a real scene object |
| `render(resolved)` | build | produce the natural-language instruction |

---

## Two commands

### command 1 — `generate` (scene-independent, structure only)

Generates instruction style lists. The config inputs are the **per-length ratio
(`length_ratios`)**, the **episodes per scene (`episodes_per_scene`)**, and the CLI
argument **number of scenes (`--num-scenes`)**.

> **Scenes are NOT chosen here.** command 1 only takes the *count* and produces that many
> abstract "scene slots" (`scene_000.json`, `scene_001.json`, …), each holding
> `episodes_per_scene` structural style lists with no scene identity. The actual scene is
> assigned later in command 2, which maps slot *i* → the *i*-th scene_id you pass. So the
> number of scene_ids given to `build` must be ≥ `num_scenes`.

**Balancing (`balancer.py`)** — no biasing at generation time. Instead it **generates
freely → measures the distributions → deletes the lists that hurt the fit most (those
feeding over-represented bins) → regenerates → keeps the ones that fit → repeats**. A
single mechanism balances every dimension at once:

| dimension | unit | target | config |
|---|---|---|---|
| length        | one list      | `length_ratios`      | (always) |
| ab_distance   | AB close span | `ab_distance_ratios` | (if set) |
| or_distance   | OR back-ref   | `or_distance_ratios` | (if set) |
| styles        | token role    | uniform              | `even_styles` |

The fit 'error' is the weighted (`balance_weights`) sum of the L1 divergence between
observed and target ratios per dimension. Each round it drops `prune_fraction` of the
pool by highest harm (contribution to over-represented bins) and refills with freshly
generated candidates of highest benefit (filling under-represented bins). It repeats for
`refine_iterations` rounds and keeps the pool with the lowest error.

**Hard rejection** keeps out-of-target values from appearing at all: a list whose length
is outside `length_ratios`, or whose AB/OR reference distance is outside the configured
distance set, is discarded at the candidate stage (e.g. if `ab_distance_ratios` allows
1–3, no list with an AB distance of 5 ever survives — same idea as rejecting lengths 9/10
when the length set is 2–8).

Three reference-distance statistics are produced (each with count/min/max/avg/distribution):
- `ab_pre_post_distance` : distance between an AB_post and the AB_pre that opened the alias it closes
- `or_reference_distance`: the OR back-reference distance (`order - ordinal_k`; includes both
  pure `OR_post` and the multi-role `{AB_pre,OR_post}`)
- `ref_distance_combined`: the two merged

If `ab_distance_ratios` / `or_distance_ratios` are set, the statistics/plots show achieved
vs target.

```bash
python main.py generate -o out/run1 --num-scenes 5 --episodes-per-scene 10
# or with a config file
python main.py generate -o out/run1 --config configs/generator.example.json
# or just put config.json in the folder (the folder's config.json is read automatically)
mkdir -p out/run1 && cp configs/generator.example.json out/run1/config.json
python main.py generate -o out/run1
# parallelize scene generation across processes (pure-Python CPU work → multiprocessing)
python main.py generate -o out/run1 --config configs/generator.example.json --workers 8
```

> **`--workers N`** generates scenes in parallel. Generation is pure-Python CPU work, so
> this uses multiprocessing (threads wouldn't help under the GIL). Each scene is
> independent and deterministic per `(seed, slot)`, so parallel output is byte-identical
> to a sequential run; the main process merges each worker's statistics and owns the
> progress file, so it stays resumable. Pick `N` near your core count.

At the end, statistics charts are written automatically (disable with `--no-plot`).

**Resumable (checkpointing)** — generation is per-scene, and **each scene is written to
disk the moment it is built** (only one scene's worth of lists is in memory at a time;
statistics are accumulated incrementally). Progress is recorded atomically in
`_progress.json`, so if the run is interrupted, **re-running with the same `-o` folder
skips finished scenes and continues**. The per-scene RNG is fixed by `(seed, slot)`, so a
resumed run reproduces identical scenes. Command 2 (`build`) resumes the same way at the
shard level (already-built shards are skipped). This suits large / long-running jobs.

Output folder:

```
out/run1/
├── config.json        # the GeneratorConfig used
├── _progress.json     # progress checkpoint (completed scenes / accumulated stats) — for resume
├── statistics.json    # statistics (machine-readable, written when fully done)
├── statistics.txt     # statistics (human-readable: length dist / style freq / goal-absent / distances)
└── scenes/
    ├── scene_000.json # episodes_per_scene style lists (scene slot 0; written immediately)
    └── ...
```

### command 2 — `build` (scene/object assignment, needs habitat-sim)

Takes a generated folder, **maps each scene slot onto a real scene_id**, and uses
`HM3DSceneLoader` to pull the scene's objects/viewpoints and **resolve the reference
structure onto real objects**, producing a GOAT-compatible referential shard.

```bash
python main.py build -i out/run1 \
    --scenes-file configs/scenes.example.txt \
    --builder-config configs/builder.example.json
# or pass scene ids directly as a comma list
python main.py build -i out/run1 --scenes "hm3d/train/00000-xxx/xxx.basis.glb,..."
# parallelize across scenes (each worker runs its own habitat-sim; ~2GB RAM each)
python main.py build -i out/run1 --scenes-file configs/scenes.example.txt --workers 6
```

Output: `{dataset_root}/{split}/content/{scene}.json.gz` (one shard per scene).

> **`--workers N`** builds scenes in parallel with a process pool. Scenes are fully
> independent (one shard each), and threads would not help (Python GIL + habitat-sim is
> not thread-safe), so this uses multiprocessing. It stays resumable: a shard's existence
> means "done", the main process owns the progress file, and interrupted runs skip finished
> scenes. Pick `N` by RAM/GPU headroom (each worker ≈ 2GB + its own sim).

> **`hm3d-scene-loader` is bundled** at `hm3d_scene_loader/`, so a fresh clone builds
> with no extra checkout — see [bundled dependency](#bundled-dependency--hm3d_scene_loader)
> below.

### Getting HM3D scenes (one-time download)

command 2 needs the HM3D scene meshes (`.glb`) + semantic annotations locally. These are
downloaded once with habitat-sim's tool. The **credential is download-only** — neither
this pipeline nor the benchmark runtime ever uses it again, so there is no `.env`; just run
the command (this is also what goat-bench assumes: scenes are pre-downloaded by the user).

```bash
# habitat env (osx-arm64 builds are Python 3.9); wget is required by the downloader
conda create -y -n refon39 python=3.9
conda install -y -n refon39 -c conda-forge -c aihabitat habitat-sim withbullet numpy matplotlib wget

# (a) FREE example scenes — no credentials (3 scenes, 1 with semantics). Good for a smoke test.
python -m habitat_sim.utils.datasets_download --uids hm3d_example --data-path data

# (b) FULL HM3D — needs a Matterport API token (id + secret) passed on the command line.
#     Get the token by signing the HM3D agreement at matterport.com (habitat-matterport-3d).
python -m habitat_sim.utils.datasets_download \
    --username <api-token-id> --password <api-token-secret> \
    --uids hm3d_train_v0.2 hm3d_train_semantic_annots_v0.2 hm3d_train_semantic_configs_v0.2 \
    --data-path data
#   (swap 'train' for 'val' / 'minival' for the smaller splits)

# build the scene-id list (paths relative to data/scene_datasets/, the loader's scene_root)
find data/scene_datasets/hm3d/train -name "*.basis.glb" \
    | sed 's#.*/scene_datasets/##' | sort > configs/scenes_train.txt
```

Then run the two commands (use the Python 3.9 habitat env for `build`):

```bash
# 1) generate the style lists for as many scene slots as you have scenes
python main.py generate -o out/run1 --num-scenes 136 --episodes-per-scene 10 \
    --config configs/generator.example.json

# 2) assign real scenes and build the shards (resumable; skips finished scenes)
conda run -n refon39 python main.py build -i out/run1 \
    --scenes-file configs/scenes_train.txt \
    --builder-config configs/builder.hm3d_train.json
```

A quick smoke test with the FREE example scene (no credentials):

```bash
python main.py generate -o out/example_test --num-scenes 1 --episodes-per-scene 20 \
    --config configs/generator.example.json
conda run -n refon39 python main.py build -i out/example_test \
    --scenes "hm3d/example/00861-GLAQ4DNUx5U/GLAQ4DNUx5U.basis.glb" \
    --builder-config configs/builder.example_hm3d.json
```

> The builder config sets `scene_root`, `scene_dataset_config`, and viewpoint thresholds
> for the HM3D loader: `configs/builder.hm3d_train.json` targets the full HM3D set
> (`hm3d_annotated_basis.scene_dataset_config.json`), while `configs/builder.example_hm3d.json`
> targets the free example scenes (`hm3d_annotated_example_basis...`). Set `--num-scenes` to
> match the number of scene ids in your `--scenes-file`.

### Building against 3D-Mem's HM3D copy

3D-Mem points at HM3D with one directory holding the split folders — `scene_data_path`
in `cfg/eval_*.yaml`, i.e. `<root>/val/00877-4ok3usBNeis/4ok3usBNeis.basis.glb`. Set
`hm3d_root` to that same directory and the generator builds episodes against exactly the
meshes the evaluator will load, instead of a second copy under `data/scene_datasets/`.
The matching `hm3d_annotated_basis.scene_dataset_config.json` is picked up automatically
from inside `hm3d_root` or from its parent (where 3D-Mem keeps it).

With `hm3d_root` set, scenes may be named in any of these forms — all resolve to the same
scene, and the shard is always written as `<hash>.json[.gz]`, which is the name 3D-Mem
looks the scene up by:

```
4ok3usBNeis                                        # bare hash
00877-4ok3usBNeis                                  # split folder name
val/00877-4ok3usBNeis                              # split-qualified
hm3d/val/00877-4ok3usBNeis/4ok3usBNeis.basis.glb   # GOAT-style relative id
```

`scenes` lists what is actually there (directory listing only, no scene is opened):

```bash
# from the 3D-Mem repo root
python RefONEpisodeGenerator/main.py scenes --hm3d-root data/hm3d --split val \
    -o RefONEpisodeGenerator/configs/scenes_val_unseen.txt
```

#### Smoke test with the free example scene

The HM3D example download ships three scenes but only `00861-GLAQ4DNUx5U` has semantic
annotations, and 3D-Mem asserts on all four files (`.basis.glb`, `.basis.navmesh`,
`.semantic.glb`, `.semantic.txt`). 3D-Mem also picks the split arithmetically
(`scene_goatbench.py`: index `< 800` → `train`, else `val`), so `00861` has to live under
`val/` regardless of it being an "example" scene:

```bash
# from the 3D-Mem repo root
mkdir -p data/hm3d/train data/hm3d/val          # train/ must exist; the evaluator lists both
ln -sfn "$PWD/RefONEpisodeGenerator/data/versioned_data/hm3d-0.2/hm3d/example/00861-GLAQ4DNUx5U" \
        data/hm3d/val/00861-GLAQ4DNUx5U

python RefONEpisodeGenerator/main.py generate \
    -o RefONEpisodeGenerator/out/refon_example --num-scenes 1 --episodes-per-scene 30
conda run -n refon39 python RefONEpisodeGenerator/main.py build \
    -i RefONEpisodeGenerator/out/refon_example --scenes GLAQ4DNUx5U \
    --builder-config RefONEpisodeGenerator/configs/builder.3dmem_example.json
```

No `scene_dataset_config` needs to be created: `hm3d_root=data/hm3d` resolves to
3D-Mem's own `data/hm3d_annotated_basis.scene_dataset_config.json` one level up, so the
builder and the evaluator load the scene through the identical file.

> `min_iou` is a **frame-coverage fraction** (object pixels ÷ 512×512 frame), not an
> intersection-over-union, despite the field name — see `loader.py:_frame_coverage`. It
> is not comparable to the `iou` values in GOAT's own shards, which exceed 1.0 and are
> therefore a different quantity entirely. Measured on this scene, of 60 whitelist
> objects only 19 are visible from any navigable point, and raising the threshold starves
> the goal pool fast: `0.001` → 32% usable, `0.005` → 20%, `0.05` → 8%. `0.005` is the
> working default.

Then generate and build:

```bash
python RefONEpisodeGenerator/main.py generate \
    -o RefONEpisodeGenerator/out/refon_val_unseen --num-scenes <N> --episodes-per-scene 200

conda run -n refon39 python RefONEpisodeGenerator/main.py build \
    -i RefONEpisodeGenerator/out/refon_val_unseen \
    --scenes-file RefONEpisodeGenerator/configs/scenes_val_unseen.txt \
    --builder-config RefONEpisodeGenerator/configs/builder.3dmem_val_unseen.json
```

Point `test_data_dir` in `cfg/eval_refonbench.yaml` at the resulting `.../content/`
directory. `run_refonbench_evaluation.py` reads `.json` and `.json.gz` shards alike, so
there is nothing to decompress; `--no-compress` (or `"compress": false`) only exists for
when you want to open a shard in an editor.

```bash
python run_refonbench_evaluation.py -cf cfg/eval_refonbench.yaml --split 0
```

Three things about how the evaluator treats these shards:

- **`episodes_per_scene`** truncates each shard to a stable prefix (default 10). Shards
  hold thousands of episodes; without this, `--split k` would run one of them and ignore
  the rest. `--split 0` runs the whole subset in one process, `--split k` runs its k-th
  episode so k can be fanned out across processes (GOAT's parallelisation contract).
- **Results break down by role**, not by GOAT goal type: `success_by_task` /
  `spl_by_task` are keyed `S`, `AB_pre`, `AB_post`, `AR_pre`, `AR_post`, `OR_post`,
  `AB_pre+OR_post`.
- **`GA_*` (goal-absent) subtasks are skipped.** They name no object, and 3D-Mem has no
  way to score "the correct behaviour is to stop" — it only scores reaching a target. So
  the ~15% goal-absent share of a generated dataset is not exercised by this evaluator.
  `skip_goal_absent: false` in the config logs them instead of dropping them silently.

Each subtask is also still evaluated independently: the agent gets `instruction`
verbatim as its prompt but no history of earlier subtasks, so back references
("Find A1.", "Go back to the previous one.") are under-specified from its point of view.
That is a prompt-construction problem, not a data-format one.

### instruction feasibility — `run_refonbench_feasibility.py`

Before asking whether the agent can *navigate* to a referent, it is worth knowing
whether the referent is recoverable from the instructions at all. That is what
`run_refonbench_feasibility.py` measures: no habitat, no images, no navigation — the VLM
sees only the episode's instructions and answers, for one of them, *which object it is
being sent to* (`"new"`, the number of the instruction that introduced the object, or
`"none"`).

```bash
python run_refonbench_feasibility.py -cf cfg/eval_refonbench_feasibility.yaml
python run_refonbench_feasibility.py -cf cfg/eval_refonbench_feasibility.yaml --mode all_at_once
python run_refonbench_feasibility.py -cf cfg/eval_refonbench_feasibility.yaml --dry-run
```

- **`--mode incremental`** (default) sends one query per subgoal showing instructions
  1..i — the history a navigation run actually has at subgoal i. **`--mode all_at_once`**
  sends the whole episode in one query, so the model may look ahead.
- An answer is scored **correct** when it points at the ground-truth object *and* names
  its category. Any instruction number that lands on the same `object_id` counts: an
  episode may revisit one object several times, and then "the 2nd one" and "the 3rd one"
  are both true of the same referent. Results are broken down by instruction style
  (`role`), with `referent SR` / `category SR` / joint `SR` reported separately.
- **`GA_*` subtasks are scored here**, unlike in the navigation runner: "refers to no
  object" is a perfectly checkable answer even though "stop" is not a reachable target.
  `--skip-goal-absent` drops them.
- Output: `feasibility_records_<mode>.jsonl` (one row per subgoal, with the raw model
  reply) and `feasibility_results_<mode>.json` under `results/<exp_name>/`.

### auxiliary — `validate` / `plot`

```bash
python main.py validate -i out/run1   # re-validate every style list against the automaton grammar + stats

python main.py plot -i out/run1        # save statistics pie/bar charts as PNG (matplotlib)
#   -> out/run1/plots/dashboard.png (+ individual PNGs)

python main.py sample -i out/run1      # print one random episode per length to the console
#   --seed N (reproducible)  --max-scenes K (scenes to sample from)  --no-color
```

`sample` prints one random episode per length with the referential structure made
visible: symbolic object labels (O1, O2, …), each subgoal's rendered instruction, and a
`↩ refers to #N` arrow for every back-reference, plus goal-absent `✗ STOP` markers. Since
command 1 output has no scene objects, it uses symbolic labels (no habitat needed).

`plot` saves PNGs with matplotlib. Charts: role (instruction style) distribution pie,
length distribution bar (target vs actual), goal-absent vs normal pie, goal-absent kind
pie, AB / OR / combined reference-distance bars, and a single-sheet `dashboard.png`.

---

## Output episode JSON schema (referential)

GOAT's top-level `goals` catalog (viewpoints) is kept as-is, and the episode is extended
with a **`subtasks` dict array** that carries the referential info (the chosen schema).

```jsonc
{
  "episodes": [{
    "episode_id": "0",
    "scene_id": "hm3d/train/00000-xxx/xxx.basis.glb",
    "start_position": [x,y,z], "start_rotation": [qx,qy,qz,qw],
    "info": { "geodesic_distance": 3.1 },
    "goal_absent": false,
    "subtasks": [
      {"order":1,"role":"AB_pre","category":"chair","object_id":"chair_3",
       "alias":"A1","instruction":"Find the chair. Let's call it A1."},
      {"order":2,"role":"S","category":"table","object_id":"table_1",
       "instruction":"Find the table."},
      {"order":3,"role":"AB_post","category":"chair","object_id":"chair_3",
       "alias":"A1","ref_order":1,"instruction":"Find A1."}
    ]
  }],
  "goals": { "<scene_key>_chair": [ { "object_id":"chair_3", "view_points":[...] }, ... ] }
}
```

- `ref_order`: the `order` of the earlier subgoal this one points back to. It is the
  resolution target of a back-reference (`AB_post`/`OR_post`/`AR_post`/`{AB_pre,OR_post}`),
  and `object_id` is filled to be **identical** to that target.
- The `goals` catalog contains **only the objects actually used**, with their viewpoints.

### goal-absent (nonsensical instruction → the agent stops)

An **impossible instruction**: an alias that was never bound, an object not in the scene,
or an ordinal that does not exist yet. It only ever appears as the **last** subgoal of an
episode (no subgoals after it), and the agent is expected to stop there.

```jsonc
{"order":2,"role":"GA_invalid_ordinal","category":null,"object_id":null,
 "ordinal_k":5,"goal_absent":true,"expected_behavior":"stop",
 "instruction":"Find the 5th one again.",
 "reason":"only 1 object(s) visited before, no 5th one exists"}
```

Three kinds: `unbound_alias` ("Find Z3." with Z3 undefined), `absent_object`
("Find the dragon."), `invalid_ordinal` (visited 2, "Find the 5th one again.").

---

## Adding a new instruction style

1. Subclass `InstructionStyle` — implement `admissible / apply / resolve / render`.
2. Add one `register(MyStyle())` line in `refon/styles/__init__.py`.

Done. The generator, balancer, builder, and validator all consult only the registry, so
nothing else changes. If the style is terminal (goal-absent-like), set `is_terminal = True`
and it is automatically treated as a "last subgoal only" style.

---

## Dependencies

- **command 1 / validate**: standard library only (nothing to install).
- **plot**: `matplotlib`.
- **command 2 (build)**: `habitat-sim` + `numpy`. `hm3d-scene-loader` needs no install --
  it ships in this repo. (See `requirements.txt`. habitat-sim is best installed via conda.)

### bundled dependency — `hm3d_scene_loader/`

Verbatim copy of the `hm3d_scene_loader` package from the separate `hm3d-scene-loader`
project (`~/PycharmProjects/hm3d-scene-loader`, not a git repository — there is no
upstream URL to submodule). Only the package directory is copied; its `SPEC.md` and
`examples/` are not. It wraps habitat-sim to expose one HM3D scene's semantic objects,
sampled viewpoints, image-goal camera parameters, and start states in the GOAT
goal-catalog format. `refon/episode_builder.py` is its only consumer.

`_import_loader()` takes the first directory that actually contains
`hm3d_scene_loader/__init__.py`:

1. `$HM3D_SCENE_LOADER_PATH`
2. this project root (the bundled copy — the default)
3. `~/PycharmProjects/hm3d-scene-loader`

To develop against the original checkout instead:

```bash
HM3D_SCENE_LOADER_PATH=~/PycharmProjects/hm3d-scene-loader \
    python RefONEpisodeGenerator/main.py build ...
```

To re-sync the bundled copy after changing the original:

```bash
rsync -a --exclude='__pycache__' --exclude='.DS_Store' \
    ~/PycharmProjects/hm3d-scene-loader/hm3d_scene_loader/ \
    RefONEpisodeGenerator/hm3d_scene_loader/
```

Keep the two in sync deliberately: the viewpoint sampling parameters here have to match
the evaluator's navmesh settings, or generated viewpoints may not be navigable at
evaluation time (see the note at the top of `hm3d_scene_loader/config.py`).
