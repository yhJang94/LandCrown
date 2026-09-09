# Stage 1 — Landmark & Coarse Crown Prediction

**Stage 1 of the two-stage `LandCrown` pipeline.** Given the dentition around a
missing tooth, this stage predicts:

1. **Landmarks** of the missing tooth (`PreModel`), and
2. a **coarse 128-point crown** for it (`PreModel2`),

conditioned on four arch-form templates (Square / Ovoid / Tapered / Omega).

The predicted coarse crown is written to disk and consumed as input by
**Stage 2** ([`../fine_generator/`](../fine_generator/)), which
upsamples it to a dense point cloud. The two stages are coupled only through these
files and have **separate dependencies / environments**.

This stage is built on **GraphMAE** (THUDM, MIT) — see the repo-root `LICENSE`.

---

## Contents

```
main_inductive.py            training / validation entry point (the two models trained jointly)
configs.yml                  hyper-parameters (key: `ppi`)
requirements.txt             pinned dependencies
graphmae/
├── models/
│   ├── edcoder.py           PreModel (landmark) + PreModel2 (coarse) + attention blocks
│   ├── gat.py gcn.py gin.py dot_gat.py    graph encoders
│   └── loss_func.py
│   └── __init__.py          build_model / build_model2
├── datasets/data_util.py    CrownDataset + load_inductive_dataset
└── utils.py                 arg parsing, optimizer, logging
debug/arch_templates/
└── arch_templates_global.npz   arch-form templates (required to build the model)
```

Only `main_inductive.py` is runnable here. `main_graph.py` / `main_transductive.py`
and other datasets from upstream GraphMAE are **not** part of this release.

---

## Environment

Reference: **Python 3.8, CUDA 11.8, PyTorch 2.1.1, DGL 2.4.0**.

```bash
conda create -n landcrown python=3.8 -y
conda activate landcrown

# CUDA-specific builds first (match your CUDA toolkit; cu118 shown)
pip install torch==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install dgl==2.4.0+cu118 -f https://data.dgl.ai/wheels/torch-2.1/cu118/repo.html

pip install -r requirements.txt
```

Sanity check (no data needed):

```bash
python -c "import torch, dgl; from graphmae.models import build_model, build_model2; print('ok', torch.__version__, dgl.__version__)"
```

---

## Data

The dataset is derived from the public **Teeth3DS** benchmark
(Ben Hamadou et al., <https://github.com/abenhamadou/3DTeethSeg22_challenge>).
Set `DATA_ROOT` (default `/data/yohan`) so the following layout resolves:

```
$DATA_ROOT/
├── train.txt                     # one sample id per line:  <case>_<targetToothIdx>   (idx 0..27)
├── test.txt
├── teeth3ds_aligned_input/       # <sample>.npy  -> (28*128, 3)  : 28 teeth x 128 pts (xyz);
│                                 #   the target tooth block is zeroed = masking signal
├── teeth3ds_aligned_gt/          # <sample>.npy  -> (128, 3)     : target tooth, 128 pts, before masking
└── teeth3ds_land_aligned_npz/    # <case>.npz    -> {toothIdx: {'coord': (K,3), 'class': [K]}}
                                  #   class in {Mesial, Distal, InnerPoint, OuterPoint, FacialPoint, Cusp}
```

Point clouds are centroid-/scale-normalised per sample at load time
(`CrownDataset.normalize`, using valid points only).

The arch-form template file (`debug/arch_templates/arch_templates_global.npz`,
shape `(2 arches, 4 forms, 12 teeth, 3)`) is required to instantiate the model and is
included in the repo; override its location with `ARCH_TEMPLATE_PATH` if needed.

---

## Training

`main_inductive.py` initialises a `torch.distributed` (NCCL) process group, so launch it
with `torchrun`, one GPU per process:

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
    main_inductive.py --dataset ppi --use_cfg --device 0 --seeds 0
```

### Outputs

```
pretrain_model/<folder_name>/   model_best.pt, model2_best.pt, model_epoch_*.pt, model2_epoch_*.pt
debug/<folder_name>/            pred_*.npy, gt_*.npy, test_*_*.npy   (for offline metric computation)
```

---

## Acknowledgement / License

Built on **GraphMAE** (THUDM), MIT License — original text retained in the repo-root `LICENSE`.
