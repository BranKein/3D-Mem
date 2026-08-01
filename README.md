<br/>
<p align="center">
  <h1 align="center">3D-Mem: 3D Scene Memory for Embodied Exploration and Reasoning</h1>
  <p align="center">
    CVPR 2025
  </p>
  <p align="center">
    <a href="https://yyuncong.github.io/">Yuncong Yang</a>,
    <a href="https://hanyangclarence.github.io/">Han Yang</a>,
    <a href="https://www.linkedin.com/in/jiachen-zhou5/">Jiachen Zhou</a>,
    <a href="https://peihaochen.github.io/">Peihao Chen</a>,
    <a href="https://icefoxzhx.github.io/">Hongxin Zhang</a>,
    <a href="https://yilundu.github.io/">Yilun Du</a>,
    <a href="https://people.csail.mit.edu/ganchuang">Chuang Gan</a>
  </p>
  <p align="center">
    <a href="https://arxiv.org/abs/2411.17735">
      <img src='https://img.shields.io/badge/Paper-PDF-red?style=flat&logo=arXiv&logoColor=red' alt='Paper PDF'>
    </a>
    <a href='https://umass-embodied-agi.github.io/3D-Mem/' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=flat&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
    </a>
  </p>
</p>

---

This is the official repository of **3D-Mem**: 3D Scene Memory for Embodied Exploration and Reasoning.

![](assets/teaser.png)

---

## News

- [2025/03] Inference code for A-EQA and GOAT-Bench is released.
- [2025/02] 3D-Mem is accepted to CVPR 2025!
- [2024/12] [Paper](https://www.arxiv.org/abs/2411.17735) is on arXiv.

## Installation
Set up the conda environment (Linux, Python 3.9):
```bash
conda create -n 3dmem python=3.9 -y && conda activate 3dmem

pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
conda install -c conda-forge -c aihabitat habitat-sim=0.2.5 headless faiss-cpu=1.7.4 -y
conda install https://anaconda.org/pytorch3d/pytorch3d/0.7.4/download/linux-64/pytorch3d-0.7.4-py39_cu118_pyt201.tar.bz2 -y

pip install omegaconf==2.3.0 open-clip-torch==2.26.1 ultralytics==8.2.31 supervision==0.21.0 opencv-python-headless==4.10.* \
 scikit-learn==1.4 scikit-image==0.22 open3d==0.18.0 hipart==1.0.4 openai==1.35.3 httpx==0.27.2 scipy==1.11.4

```

### Numpy ABI troubleshooting

This environment runs numpy 1.x. Mixing in anything built against the **numpy 2 ABI** produces runtime errors that never mention numpy, so they are easy to misdiagnose. Symptoms seen in practice:

```
# scipy — also takes down habitat_sim, open3d, sklearn and HiPart, since they all import it
File "scipy/interpolate/_fitpack_impl.py", line 103, in <module>
    'iwrk': array([], dfitpack_int), 'u': array([], float),
TypeError

# numba — every jitted function fails on its first call, e.g. TSDFPlannerBase.vox2world
TypeError: can't unbox array from PyObject into native value.  The object maybe of a different type

# pandas (reached through ultralytics)
TypeError: Cannot convert numpy.ndarray to numpy.ndarray

# opencv
cv2.error: Overload resolution failed: src is not a numpy array, neither a scalar
```

**First check whether numpy itself is intact.** If a numpy 2.x install was ever laid on top of a numpy 1.x one (or vice versa), `site-packages/numpy/` keeps files from both, and `pip uninstall` only removes the ones its own RECORD lists. Two numpy C cores then load in the same process and every package above breaks no matter which version of it you install. In numpy 1.26 `numpy/_core/` is a pure-Python shim, so any `.so` in it is a leftover from a numpy 2 install:

```bash
python -c "import numpy, sys; print(numpy.__version__, numpy.__file__)"
ls "$(python -c 'import numpy,os;print(os.path.dirname(numpy.__file__))')/_core/"*.so 2>/dev/null && echo "LEFTOVERS -- numpy install is mixed"
```

To repair, wipe the package directory and reinstall — uninstalling alone is not enough:

```bash
pip uninstall -y numpy
rm -rf "<site-packages>/numpy"        # remove whatever survived the uninstall
rm -rf "<site-packages>"/numpy-*.dist-info
pip install "numpy==1.26.4"
```

Verify with a single loaded C core and a round of real operations:

```bash
python -c "
import numpy as np, sys, cv2, pandas, numba
print(sorted({m.__file__ for n,m in sys.modules.items() if n.startswith('numpy') and getattr(m,'__file__','') .endswith('.so') and 'multiarray_umath' in m.__file__}))
cv2.resize(np.zeros((8,8,3), np.uint8), (4,4)); pandas.DataFrame([[1,'a']], columns=['x','y'])
print('numba', numba.njit(lambda a: a.sum())(np.arange(10.)))"
```

A known-good set for numpy 1.26.4 on Python 3.9: `scipy==1.11.4`, `numba==0.58.1`, `llvmlite==0.41.1`, `pandas==2.2.2`, `opencv-python-headless==4.10.0.84`. Install only the headless OpenCV — `opencv-python` and `opencv-python-headless` share the same `cv2/` directory, so having both means whichever was installed last silently wins.


## Run Evaluation

### 1 - Preparations

#### Dataset
Please download the train and val split of [HM3D](https://aihabitat.org/datasets/hm3d-semantics/), and specify
the path in `cfg/eval_aeqa.yaml` and `cfg/eval_goatbench.yaml`. For example, if your download path is `/your_path/hm3d/` that 
contains `/your_path/hm3d/train/` and `/your_path/hm3d/val/`, you can set the `scene_data_path` in the config files as `/your_path/hm3d/`.

The test questions of A-EQA and GOAT-Bench are provided in the `data/` folder. For A-EQA, we provide two subsets of different size: `aeqa_questions-41.json` and `aeqa_questions-184.json`, where `aeqa_questions-184.json` is the official subset provided by OpenEQA and `aeqa_questions-41.json` is a smaller subset for quick evaluation.
For GOAT-Bench, we include the complete `val_unseen` split in this repository.

#### VLM Setup
The evaluation prompts a vision-language model at each step. Two backends are supported, selected with `VLM_PROVIDER` in `src/const.py` (every setting there can also be overridden with the environment variable of the same name):

**OpenAI (default, `VLM_PROVIDER = "openai"`)**: set the endpoint and API key in `src/const.py` (`END_POINT`, `OPENAI_KEY`). Leave `END_POINT` empty to use the official API. The model is `OPENAI_MODEL` (default `gpt-4o`). This backend also works with any OpenAI-compatible server (vLLM, LiteLLM, ...).

**Ollama (`VLM_PROVIDER = "ollama"`)**: runs a local vision model, no API key needed. Pull a vision model, then run the evaluation — the ollama server loads the model on the first request, nothing else to start:
```bash
ollama pull qwen2.5vl:7b
VLM_PROVIDER=ollama OLLAMA_MODEL=qwen2.5vl:7b python run_aeqa_evaluation.py -cf cfg/eval_aeqa.yaml
```
Relevant settings: `OLLAMA_END_POINT` (default `http://localhost:11434`), `OLLAMA_MODEL`, `OLLAMA_TIMEOUT`.

Two things to watch out for:
- The model **must** be a vision model, and each prompt contains many images, so the server's context length must be large. Context length is a server setting, not a per-request one: start the server with `OLLAMA_CONTEXT_LENGTH=32768 ollama serve` (or add it to the systemd unit), and check what a loaded model actually got with `curl -s localhost:11434/api/ps`. Anything beyond the context window is silently truncated. Lowering `prompt_h`/`prompt_w` and `top_k_categories` in the config keeps the prompt smaller.
- This backend uses ollama's OpenAI-compatible `/v1` API on purpose. The native `/api/chat` endpoint passes images as a separate list rather than interleaved with the text, and the model then cannot tell which image belongs to which `Snapshot i` label.

To check that the configured backend answers:
```bash
python -m src.vlm_client
```

### 2 - Run Evaluation on A-EQA

First run the following script to generate the predictions for the A-EQA dataset:

```bash
python run_aeqa_evaluation.py -cf cfg/eval_aeqa.yaml
```
To split tasks, you can add `--start_ratio` and `--end_ratio` to specify the range of tasks to evaluate. For example,
to evaluate the first half of the dataset, you can run:
```bash
python run_aeqa_evaluation.py -cf cfg/eval_aeqa.yaml --start_ratio 0.0 --end_ratio 0.5
```
After the scripts finish, the results from all splits will be automatically aggregated and saved.

To evaluate the predictions with the pipeline from OpenEQA, you can refer to [link](https://github.com/yyuncong/3D-Mem-AEQA-Eval)

### 3 - Run Evaluation on GOAT-Bench
You can directly run the following script:
```bash
python run_goatbench_evaluation.py -cf cfg/eval_goatbench.yaml
```
The results will be saved and printed after the script finishes. You can also split the task similarly by adding `--start_ratio` and `--end_ratio`.
Note that GOAT-Bench provides 10 explore episodes for each scene, and by default we only test the first episode due to the time and resource constraints.
You can also specify the episode to evaluate for each scene by setting `--split`.

### 4 - Save Visualization
The default evaluation config will save visualization results including topdown maps, egocentric views, memory snapshots, and frontier snapshots at each step. Although saving visualization is very helpful, it may slows down the evaluation process. Please make save_visualization false if you would like to run large-scale evaluation.


## Acknowledgement

The codebase is built upon [OpenEQA](https://github.com/facebookresearch/open-eqa), [Explore-EQA](https://github.com/Stanford-ILIAD/explore-eqa), and [ConceptGraph](https://github.com/concept-graphs/concept-graphs).
We thank the authors for their great work.

## Citing 3D-Mem

```tex
@InProceedings{Yang_2025_CVPR,
    author    = {Yang, Yuncong and Yang, Han and Zhou, Jiachen and Chen, Peihao and Zhang, Hongxin and Du, Yilun and Gan, Chuang},
    title     = {3D-Mem: 3D Scene Memory for Embodied Exploration and Reasoning},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {17294-17303}
}
```
