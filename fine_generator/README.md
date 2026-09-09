# Stage 2 — Fine Generator

**Stage 2 of the `LandCrown` pipeline.** Upsamples the coarse crown produced by
[Stage 1](../dual_graph/) into a dense point cloud for the missing tooth.

## Method

We adopt **DiffPMAE** (<https://github.com/TyraelDLee/DiffPMAE>) as the backbone. In the
masked-autoencoder framework the masked region is the missing tooth and the visible
region is the remaining dentition. Unlike the original implementation that uses
ground-truth coarse points, this model loads the **coarse points predicted by Stage 1**
from disk and uses them as input. The autoencoder is trained on segmented individual
teeth rather than the whole intra-oral scan.

---

## 1. Environment

Reference: conda env **`mae`** — **Python 3.9, CUDA 11.7, PyTorch 2.0.1**.

```bash
conda create -n mae python=3.9 -y
conda activate mae

pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu117

# CUDA extensions (need matching CUDA toolkit + nvcc on PATH)
pip install pointnet2-ops==3.0.0
pip install https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl

pip install -r requirements.txt
```

Sanity check (no data needed):

```bash
python -c "from model.DiffusionPretrain import Diff_Point_MAE; from metrics.evaluation_metrics import chamfer_distance_l2; print('ok')"
```

---

## 2. Data

Paths are hardcoded near the top of `train_decoder.py` (`./DiffPMAE/dataset/...`) and in
`utils/dataset.py`; edit them to point at your dataset. Expected layout:

```
<dataset_dir>/
├── train.txt / test.txt        # one sample name per line (name like  <...>_<...>_<toothIdx>)
├── data_re_12288/  <name>.npy  # full arch, sliced 1024 pts per tooth   -> input_path
├── data_re_448/    <name>.npy  # coarse arch, sliced 32 pts per tooth   -> cr_path  (Stage-1 output)
└── bbox/           <name>.npy  # per-sample bounding box
```

- The sample list file is read from `os.path.dirname(input_path)/<subset>.txt`.
- `missing_idx = int(name.split('_')[2])` selects the target tooth; the loader slices out
  its fine points (`gt`, 1024) and its coarse points (`coarse_pred`, 32), and returns the
  rest of the arch as `partial` / `coarse_partial`.
- **`data_re_448/` is the hand-off from Stage 1** — write Stage 1's predicted coarse
  crowns there (per-sample `.npy`), same sample names as `train.txt` / `test.txt`.

Point clouds are centroid-/scale-normalised per sample in `CrownDataset.normalize`.

---

## 3. Training

Plain single-process launch (`train_decoder.py` uses `nn.DataParallel(device_ids=[0])`,
**not** `torchrun`). All hyper-parameters are `argparse` flags with defaults — there is
no config file.

```bash
mkdir -p npy_result pretrain_model
CUDA_VISIBLE_DEVICES=0 python train_decoder.py
```

Main flags (defaults in `train_decoder.py`): `--batch_size 128`, `--learning_rate 1e-3`,
`--weight_decay 1e-3`, `--num_steps 1000` (diffusion), `--num_group 512`,
`--group_size 32`, `--trans_dim 384`, `--encoder_depth 12`, `--decoder_depth 4`.
Set `wand = True` at the top of the file to enable Weights & Biases logging (off by
default).

### Outputs

```
pretrain_model/   model_best.pt, model_<epoch>.pt, optimizer_<epoch>.pt, scheduler_<epoch>.pt
npy_result/       test_pred_<epoch>.npy, test_gt_<epoch>.npy   (validation dumps, every 50 epochs)
```

---

## Acknowledgement

Backbone: **DiffPMAE** (<https://github.com/TyraelDLee/DiffPMAE>).
