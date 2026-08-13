# Running 3D-Mem against vLLM: what breaks, and what it actually was

Every entry here was hit for real while bringing the RefON evaluations up on vLLM —
once on an RTX 4090 workstation and once on an 8×A30 server. Each one is written as
**symptom → cause → fix**, with the command that discriminated between candidates,
because in almost every case the symptom pointed at the wrong layer.

Two things are worth reading before the table.

**Nearly every failure here was silent.** Five separate times a run exited 0, printed
a summary, and meant nothing: every reply empty, every reply truncated, a crash that
still reported `empty replies: 0`, zero scenes evaluated, and a smoke test that
proved nothing because the object under test was never exercised. **A run that
finishes is not a run that measured something.** The three numbers to check before
believing any result are in [Checking a run actually ran](#checking-a-run-actually-ran).

**A test that passes can be worse than one that fails.** The single most misleading
half hour of the GL investigation came from a minimal habitat reproduction that
printed `OK`. It passed because `habitat_sim.agent.AgentConfiguration()` carries **no
sensors**, so no renderer is built and no GL context is ever created. Three
"successful" tests in a row proved nothing. When a probe passes unexpectedly, check
that it exercised the thing.

---

## Quick triage

| The message you got | It is actually | Section |
|---|---|---|
| `conda not at /opt/conda/bin/conda` | you passed the binary, it wants the root | [1](#1-conda_base-is-the-root-not-the-binary) |
| `The NVIDIA driver on your system is too old (found version 12020)` | vLLM's CUDA 13 build vs an older driver | [2](#2-driver-too-old-for-the-default-vllm-build) |
| `GLIBCXX_3.4.29' not found` | `$ENV/bin/python` skips `conda activate` | [3](#3-glibcxx-not-found-on-import) |
| `Total number of scenes: 0`, everything `nan` | `--end_ratio` on a single-scene dataset | [4](#4-zero-scenes-evaluated) |
| `GL::Context: cannot retrieve OpenGL version` | glvnd halves from two installations | [5](#5-habitat-cannot-create-a-gl-context) |
| `torch.OutOfMemoryError` while loading weights | 9B bf16 does not fit one 24 GiB card | [6](#6-out-of-memory-loading-the-model) |
| `CUDA out of memory ... warming up sampler with 256 dummy requests` | KV cache ate the headroom | [7](#7-out-of-memory-warming-up-the-sampler) |
| `client_loop: send disconnect: Broken pipe`, run gone | SSH hangup killed the process tree | [8](#8-the-run-dies-with-the-ssh-session) |
| every reply empty, `n/n subgoals got no reply` | thinking is off, so the parser eats everything | [9](#9-every-reply-is-empty) |
| every reply truncated at `max_tokens` | greedy decoding loops | [10](#10-every-reply-runs-out-of-budget) |
| `Error in splitting response ... too many values to unpack` | model did not answer on line 1 | [11](#11-unreadable-replies) |

---

## 1. CONDA_BASE is the root, not the binary

**Symptom**

```
conda not at /opt/conda/bin/conda, set CONDA_BASE
```

**Cause** — the scripts want the conda *root* (`/opt/conda`), and the name reads like
it wants the executable.

**Fix** — `scripts/setup_vllm_server.sh` and `run_nav_vllm.sh` now accept either, and
fall back to `conda info --base` when the variable is unset. That is more reliable
than guessing `$HOME/anaconda3`: on the A30 server the env lived in `~/.conda`.

```bash
CONDA_BASE=$(conda info --base) ./scripts/setup_vllm_server.sh
```

Also check `$CONDA_BASE/envs` is writable — a shared `/opt/conda` often is not, and
`conda create` failing halfway is a worse way to find out. Put the env in your home
and point at it instead:

```bash
conda create -y -p $HOME/envs/vllm python=3.12
VLLM=$HOME/envs/vllm/bin/vllm ./scripts/run_nav_vllm.sh ...
```

---

## 2. Driver too old for the default vLLM build

**Symptom** — the server never comes up. The tail of the log is the API server
re-raising and ends in `See root cause above`; the cause is further up, in the
`(EngineCore ...)` lines:

```
RuntimeError: The NVIDIA driver on your system is too old (found version 12020).
```

It dies inside `torch._C._cuda_init()` — before any model, memory or GPU assignment
is touched. **Nothing about memory settings will move it.**

**Cause** — `pip install vllm` takes the newest release, which is built for CUDA 13
and needs driver **r580+**. The server was on 535.230.02 (CUDA 12.2).

**The usable window.** torch 2.10 is the last release whose default PyPI wheel is
CUDA 12; from 2.11 on the wheels pull `nvidia-*-cu13`. Qwen3.5 support landed in vLLM
0.17.0. So on a CUDA 12 driver:

| vLLM | torch | CUDA | Qwen3.5 |
|---|---|---|---|
| 0.20 – 0.27 | 2.11 – 2.13 | 13 | yes, but will not start |
| **0.17 – 0.19.1** | **2.10** | **12** | **yes** |
| < 0.17 | 2.10 | 12 | no |

**Fix** — pin the newest that works:

```bash
pip uninstall -y vllm torch torchvision torchaudio
pip install vllm==0.19.1
python -c "import torch,vllm; print(vllm.__version__, torch.__version__, torch.cuda.is_available())"
```

`setup_vllm_server.sh` now reads the driver and pins `0.19.1` below r580 by itself
(override with `VLLM_VERSION=`), and fails the install step if torch cannot open a
CUDA context rather than leaving the first `serve` to discover it.

**The real fix is a driver update to r580+**, which removes this whole constraint.

**How to check a candidate version without installing it** — the `nvidia-*` pins in
the PyPI metadata say which CUDA generation a build targets:

```bash
python - <<'EOF'
import json, urllib.request
for pkg, ver in [("vllm","0.19.1"), ("torch","2.10.0")]:
    d = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/{pkg}/{ver}/json"))
    rd = d["info"].get("requires_dist") or []
    print(pkg, ver, [r.split(";")[0].strip() for r in rd if "nvidia-" in r][:3])
EOF
```

---

## 3. GLIBCXX not found on import

**Symptom**

```
ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29' not found
  (required by .../matplotlib/_c_internal_utils...so)
```

or the same thing via `numba` → `llvmlite` when importing `habitat_sim`.

**Cause** — the scripts call `$ENV/bin/python` directly, which skips `conda activate`
and the library paths it would have set. The host's `libstdc++` is older than what
the env's compiled extensions were built against.

**Fix** — preload the env's own, and **only** that:

```bash
LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
```

**Do not** put `$CONDA_PREFIX/lib` first on `LD_LIBRARY_PATH` instead. That was the
first attempt and it is a trap: the same directory holds conda's libGL/libEGL, which
then shadow the system NVIDIA ones. `run_nav_vllm.sh` builds the preload list itself.

If the env has no `libstdc++.so.6`:

```bash
conda install -n 3dmem -c conda-forge libstdcxx-ng
```

---

## 4. Zero scenes evaluated

**Symptom** — exit 0, and:

```
Total number of scenes: 0
Total success_by_distance results: nan, len: 0
```

**Cause** — `--end_ratio`. The scene list is sliced as
`[int(start*n) : int(end*n)]` (`run_refonbench_evaluation.py:76`), so on a
**single-scene** dataset — which every RefON set is — any `end_ratio` below 1.0
selects zero scenes.

**Fix** — size the run with `--split`, which selects episodes rather than scenes:

- `--split 1` — one episode per scene. This is the shakedown, and it is the default.
- `--split 0` — **all** episodes. A "full" run needs this explicitly; without it you
  quietly evaluate one episode per scene.

`run_nav_vllm.sh` maps `SMOKE=1` to `--split 1` and a normal run to `--split 0`, and
treats `0 scenes` as a failure.

If it really is 0 scenes, check the paths the config points at:

```bash
grep -E "^test_data_dir|^scene_data_path" cfg/eval_refonbench_default.yaml
```

---

## 5. habitat cannot create a GL context

**Symptom**

```
GL::Context: cannot retrieve OpenGL version: GL::Renderer::Error::InvalidValue
Aborted (core dumped)
```

**Cause** — glvnd with its two halves from different installations. habitat links the
**system** `libEGL` but resolves `libGLdispatch` out of the **conda env**. glvnd needs
both from one installation; mixed, the dispatch table is never populated, every GL
entry point is null, and the version query fails.

**Confirm it in one command** — the two paths tell the whole story:

```bash
ldd $CONDA_PREFIX/lib/python3.9/site-packages/habitat_sim-*/habitat_sim/_ext/habitat_sim_bindings*.so \
  | grep -iE "egl|glx|X11|libGL"
```

```
libEGL.so.1        => /lib/x86_64-linux-gnu/libEGL.so.1              ← system
libGLdispatch.so.0 => /home/you/.conda/envs/3dmem/lib/libGLdispatch.so.0   ← conda
```

**Fix** — force the system one:

```bash
LD_PRELOAD=/lib/x86_64-linux-gnu/libGLdispatch.so.0:$CONDA_PREFIX/lib/libstdc++.so.6
```

Working output then names the card:

```
Renderer: NVIDIA A30/PCIe/SSE2 by NVIDIA Corporation
OpenGL version: 4.6.0 NVIDIA 535.230.02
```

`run_nav_vllm.sh` assembles both preloads.

**Why it appeared out of nowhere on an unchanged container** — the env changed, not
the container. Installing something into `3dmem` later pulled conda's `libglvnd` in:

```bash
conda list -n 3dmem | grep -iE "glvnd|libegl|libgl"
```

**A test that actually tests GL.** `AgentConfiguration()` has an empty
`sensor_specifications`, so a Simulator built from it creates no renderer and no GL
context — it prints `OK` on a machine where GL is completely broken. Attach a sensor:

```bash
LD_PRELOAD=/lib/x86_64-linux-gnu/libGLdispatch.so.0:$CONDA_PREFIX/lib/libstdc++.so.6 \
python -c "
import habitat_sim
c = habitat_sim.SimulatorConfiguration(); c.scene_id='NONE'
spec = habitat_sim.CameraSensorSpec(); spec.uuid='color_sensor'
spec.sensor_type = habitat_sim.SensorType.COLOR; spec.resolution=[1280,1280]
a = habitat_sim.agent.AgentConfiguration(); a.sensor_specifications=[spec]
s = habitat_sim.Simulator(habitat_sim.Configuration(c,[a])); print('GL OK'); s.close()
"
```

---

## 6. Out of memory loading the model

**Symptom**

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.03 GiB.
GPU 0 has a total capacity of 23.50 GiB of which 209.69 MiB is free.
```

**Cause** — 9B in bf16 is 18.1 GiB of weights. On a 24 GiB card that leaves too
little for the KV cache and the multimodal encoder.

**Fix** — shard it, if there are spare GPUs:

```bash
VLLM_GPU=1,2 TP=2 ./scripts/run_nav_vllm.sh Qwen/Qwen3.5-9B
```

`QUANT=fp8` also fits it on one card (9.0 GiB), but on **Ampere there are no fp8
tensor cores** — vLLM accepts fp8 back to compute capability 7.5 and falls through to
the Marlin path, which stores the weights in 8 bits and converts back to bf16 to
multiply. Memory is saved, speed is not. On a box with spare cards, TP is the better
trade. On Ada or newer, fp8 is native and worth it.

Rough weight sizes:

| model | bf16 | fp8 |
|---|---|---|
| Qwen3.5-2B | 4.3 GiB | 2.1 GiB |
| Qwen3.5-4B | 8.8 GiB | 4.4 GiB |
| Qwen3.5-9B | **18.1 GiB** | 9.0 GiB |

Keep every rung of a size ladder on the **same** serving condition, even where the
small ones would fit either way — a ladder served at mixed precision is not one
condition.

---

## 7. Out of memory warming up the sampler

**Symptom**

```
CUDA out of memory occurred when warming up sampler with 256 dummy requests.
Please try lowering `max_num_seqs` or `gpu_memory_utilization`.
```

**Cause** — vLLM grows the KV cache to fill whatever `gpu_memory_utilization` leaves
after the weights, *then* warms the sampler for `max_num_seqs` concurrent requests
against what remains. The default 256 is far more than this evaluation ever uses:
`run_refonbench_evaluation.py` issues **one request at a time** (the feasibility
probes fan out with `--workers`; this one does not).

**Adding GPUs does not fix this** — the cache simply grows to fill them too.

**Fix**

```bash
MAX_NUM_SEQS=16 VLLM_UTIL=0.85 ./scripts/run_nav_vllm.sh ...
```

Both are the script defaults now. Tighten to `MAX_NUM_SEQS=8 VLLM_UTIL=0.80` if
needed.

Related: vLLM refuses to start when **free** memory is below
`gpu_memory_utilization × total`, so the ceiling is set by whatever *else* holds the
card. A desktop session (Xorg, gnome-shell) sitting on 2.5 GiB is enough to fail
0.90 on a 24 GiB card. Prefer an idle GPU and check first:

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

---

## 8. The run dies with the SSH session

**Symptom**

```
client_loop: send disconnect: Broken pipe
```

and afterwards the serving GPU is empty — SIGHUP took the script, the vLLM server and
the evaluation together.

**Fix** — detach:

```bash
tmux new -s nav -d 'CONDA_BASE=... VLLM_GPU=1,2 TP=2 EVAL_GPU=3 \
  MODELS="Qwen/Qwen3.5-9B Qwen/Qwen3.5-4B Qwen/Qwen3.5-2B" \
  ./scripts/run_nav_vllm.sh 2>&1 | tee results/vllm_nav_runs/sweep.log'
```

or `nohup env ... ./scripts/run_nav_vllm.sh > log 2>&1 & disown`.

For runs measured in days, a cron watchdog is better than either — the sweep scripts
skip work that already has results, so re-running them is idempotent and a guard can
simply restart the sweep when it is not running and not finished.

---

## 9. Every reply is empty

**Symptom** — HTTP 200 on every call, `content` empty, and the runner's own warning:

```
WARNING: 15/15 subgoals got no reply at all. These score as wrong but measure
nothing. A reasoning model that overruns max_tokens ... raise cfg.max_tokens
```

**The warning's advice is wrong for this cause.** Replies came back in 0.4 s; nothing
was truncated.

**Cause** — **Qwen3.5 defaults to thinking OFF**, and with `--reasoning-parser qwen3`
that combination answers nothing. The parser waits for the `</think>` that only a
thinking reply emits, so the entire response is classified as reasoning and `content`
is left empty.

The reply shape matters here: Qwen3.5 emits a **closing `</think>` with no opening
tag**, because the chat template opens the block in the prompt.

**Fix** — ask for thinking explicitly. `VllmClient` sends
`chat_template_kwargs: {"enable_thinking": true}` and never leaves it to the default.

**`reasoning_effort` is not the switch on vLLM.** vLLM reads it as a request to
*enable* thinking, so `--reasoning-effort none` would turn thinking on while claiming
to turn it off. `VllmClient` translates that flag into
`chat_template_kwargs: {"enable_thinking": false}` — the same thing `AnthropicClient`
does with `thinking: disabled`, so the flag keeps one meaning across backends.

**Do not run without `--reasoning-parser`.** The chain of thought then arrives in
`content`, and `_first_json` takes the *first* balanced `{...}` — a draft the model
wrote while reasoning, not its answer. Measured: correct answer `back_reference/1/toilet`,
parsed answer `new/None`. `VllmClient` strips a stray `</think>` block as a fallback
and warns loudly, but a reply cut off mid-thought has no closing tag and cannot be
recovered.

---

## 10. Every reply runs out of budget

**Symptom**

```
DROPPED Qwen3.5-2B: 9 replies were cut off by the token budget
(finish_reason 'length'), the limit is 5.
```

**Cause** — greedy decoding. With thinking on, `temperature 0.0` sends the model into
a repetition loop that never emits the closing `</think>`, so every reply consumes the
whole budget and returns empty. The configs say 0.0 because every earlier run had
thinking off, where it was correct.

**Raising `max_tokens` makes it worse, not better** — it is a loop, not a shortfall.

Measured on Qwen3.5-2B over 23 real prompts:

| temperature | presence_penalty | max_tokens | truncated | parsed |
|---|---|---|---|---|
| 0.6 | 0.0 | 32768 | 11/23 | 12/23 |
| 0.6 | 0.0 | **60000** | **13/23** | 10/23 |
| 0.6 | **1.5** | 32768 | 1/23 | 22/23 |
| **0.7** | **1.5** | 32768 | **0/23** | **23/23** |

**Fix** — `--temperature 0.7 --presence-penalty 1.5`. `presence_penalty` is the brake
that actually breaks the loop; temperature alone only halves it.

Pass them on the command line rather than editing the configs, so the configs keep
reproducing the earlier ollama runs unchanged. The navigation evaluations build their
client at import time and cannot read a config, so they take
`VLM_MAX_TOKENS` / `VLM_TEMPERATURE` / `VLM_PRESENCE_PENALTY` from the environment —
an explicit argument from a caller always wins over those.

This contradicts the feasibility write-up's claim that qwen3.5 cannot answer with
thinking at any budget up to 32768. It was a sampling problem, and ollama exposed no
way to reach the parameter that fixes it.

---

## 11. Unreadable replies

**Symptom**

```
Error in splitting response: the user wants to find the toilet.
too many values to unpack (expected 2)
```

**Cause** — the parser reads the **first line** of the reply and expects
`Snapshot i, Object j` or `Frontier i` (`src/eval_utils_gpt_goatbench.py:262`). Small
models write a preamble sentence first, despite the prompt asking for the reason on a
later line. The prefiltering parser is stricter still: each line must match a known
class name exactly, so prose there yields an empty list.

**This is not a vLLM artefact and not new.** The earlier `qwen3-vl:30b` run on ollama
hit it on **5 of 238 calls (2%)**. It climbs steeply on small models.

`run_nav_vllm.sh` reports the rate next to the score, because without it a run that
scored zero on formatting looks identical to one that scored zero on navigation:

```
  unreadable replies: 5 / 238 calls
```

**Not every oddity here is a failure.** `No Snapshot is available` appeared 38 times
in that same 30B run (16%) — early in an episode there are simply no snapshots yet.

---

## Checking a run actually ran

Before believing any number:

```
Total number of scenes: N        # 0 → evaluated nothing (§4)
empty replies: 0                 # high → no answers came back (§9)
unreadable replies: n / N calls   # high → answers came back unparseable (§11)
Total success_by_distance: ...   # nan → nothing was scored
```

For the feasibility probes, read the records rather than the summary:

```bash
python -c "
import json,glob,sys
d=sys.argv[1]
r=[json.loads(l) for l in open(glob.glob(d+'/*records*.jsonl')[0])]
n=len(r)
print(f'n={n} correct={100*sum(1 for x in r if x.get(\"correct\"))/n:.1f}% '
      f'empty={100*sum(1 for x in r if x.get(\"empty_response\"))/n:.1f}%')
" results/exp_feasibility_refonbench_qwen3_5_9b_vllm
```

And keep the smoke output out of the way: the sweep scripts skip a probe whose
results json already exists, so a 2-episode shakedown left in place will be mistaken
for a finished run. Delete those directories before the real sweep.

---

## Known-good configuration

Established on 2026-08-13. Feasibility: 9/9 runs, 0 failures, ~14 h on one RTX 4090.

| | 4090 workstation | A30 server |
|---|---|---|
| vLLM | 0.27.1 | **0.19.1** (driver 535 → §2) |
| driver | 580.126.09 | 535.230.02 |
| precision | fp8 (native, Ada) | **bf16 + TP=2** (§6) |
| `--reasoning-parser` | `qwen3` | `qwen3` |
| thinking | on, explicit (§9) | on, explicit |
| temperature / presence_penalty | 0.7 / 1.5 (§10) | 0.7 / 1.5 |
| `--max-model-len` | 40960 text · 65536 images | same |
| `--max-num-seqs` | default (probes fan out) | **16** (§7) |
| `LD_PRELOAD` | not needed | libGLdispatch + libstdc++ (§3, §5) |
| GPU split | one card | vLLM 1,2 · eval 3 |

Results carry a `_vllm` suffix on purpose: same model name, different quantisation,
thinking on, different sampling. They are not comparable with the ollama runs and must
not be globbed together with them.

## False leads

Kept because each one cost time and each looked convincing.

| Believed | Actually | What settled it |
|---|---|---|
| GPU memory too small for 9B | driver/CUDA generation mismatch | the traceback died in `torch._C._cuda_init()`, before any allocation |
| container lacks graphics capability | EGL enumerated all 8 devices | `eglQueryDevicesEXT` returned 8 |
| sensor resolution too large for GL | unrelated — 512 failed too | swept 512/640/1024/1280, all identical |
| `CameraSensorSpec` missing `sensor_subtype` | it already defaults to `PINHOLE` | printed the spec |
| habitat built for GLX, needs a display | it links libEGL, not GLX | `ldd` on the bindings |
| torch grabbing CUDA before habitat | irrelevant | CUDA-first test passed |
| habitat GL works, something else is wrong | GL had never worked | the passing test had **no sensors** |
| more GPUs will fix the sampler OOM | KV cache grows to fill them | vLLM's own error names `max_num_seqs` |
| prefiltering broken → 0 % success | normal; 30B did it too | counted it in the old 30B log |

The pattern: the symptom named a layer, and the cause was one or two layers below it.
`ldd`, `eglQueryDevicesEXT` and the PyPI dependency metadata each settled a question
that hours of configuration changes had not.
