# MAST-KAN

Implementation of **MAST-KAN: a Domain-Adapted Spectral-Temporal Paradigm for AIS Vessel Trajectory Forecasting**, by Ye Liu, Wei Xiong, Fei Yang, Da Wang and Hui Wu.

MAST-KAN combines a causal Mamba2 encoder-decoder, a position-only two-level db4 spectral branch with Mamba1 band mixers, cross-domain attention, and four KANLinear categorical output heads. This repository provides the complete proposed model, training, evaluation, inference and Florida Gulf data preparation. AIS datasets and pretrained weights are not included.

## Installation

Use Linux or WSL2 with an NVIDIA GPU. The verified runtime uses Python 3.10, PyTorch 2.3.1 with CUDA 12.1, mamba-ssm 2.2.2 and causal-conv1d 1.4.0. Native Windows and CPU execution are not supported. Missing Mamba backends cause an error.

```bash
conda create -n mast-kan python=3.10 -y
conda activate mast-kan
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install packaging ninja wheel "setuptools>=77" numpy==2.2.6 einops==0.8.2
pip install -e . --no-build-isolation
python -m mast_kan check
```

Mamba packages may require a compatible CUDA toolkit when prebuilt wheels are unavailable. Optional CSV preparation dependencies: `pip install -e '.[data]' --no-build-isolation`.

## Data

Place the following files in a directory supplied through `--data-dir`:

```text
{dataset}_train.pkl
{dataset}_valid.pkl
{dataset}_test.pkl
```

Supported dataset identifiers are `ct_dma` and `florida_gulf_2025`. Each file contains a list of `{"mmsi": int, "traj": array}` entries. The array has shape `[points, 6]`, with columns:

```text
lat_norm, lon_norm, sog_norm, cog_norm, timestamp, mmsi
```

The first four columns are normalized to `[0, 1]`; timestamps are UTC Unix seconds at 600-second intervals. Normalize latitude and longitude using the bounds below, SOG by 30 knots, and COG by 360 degrees. The MMSI column must match its entry. Load only trusted pickle files.

| Dataset | Latitude | Longitude | Train | Validation | Test |
|---|---|---|---|---|---|
| DMA | 55.5–58.0°N | 10.3–13.0°E | Jan 1–Mar 10, 2019 | Mar 11–20, 2019 | Mar 21–31, 2019 |
| FG | 26.0–28.5°N | 85.0–82.0°W | Jan 1–Mar 31, 2025 | Apr 1–15, 2025 | Apr 16–30, 2025 |

DMA source: [Danish Maritime Authority](https://www.dma.dk/safety-at-sea/navigational-information/ais-data). The upstream processed DMA collection and preparation references are available in [TrAISformer](https://github.com/CIA-Oceanix/TrAISformer/tree/main/data/ct_dma); verify the selected version against the periods above. This repository does not implement DMA's original raw-data conversion.

FG source: [Marine Cadastre](https://marinecadastre.gov/ais/), with [NOAA 2025 AIS metadata](https://www.fisheries.noaa.gov/inport/item/77594). Download and extract daily files for January 1 through April 30, 2025, then run:

```bash
python prepare_data.py --input-dir /path/to/daily_csv --output-dir data/florida_gulf_2025
```

The CSV converter accepts `ais-YYYY-MM-DD.csv` or `AIS_YYYY_MM_DD.csv` names and standard AIS columns. Timestamps must use `YYYY-MM-DD HH:MM:SS` in UTC; standardize an ISO `T` separator to a space before conversion. The converter preserves chronological splits, 10-minute resampling, historical filtering and the default 64-bucket trajectory order. It produces three pickle files and one manifest; no maps or data are downloaded.

The model loader also removes the initial low-speed prefix (normalized SOG ≤ 0.05), rejects nonfinite or ≤36-point trajectories, clips feature values above 0.9999, and pads to 145 points. The raw collection and the model-ready collection have different counts:

| Dataset | Raw train / valid / test | Kept train / valid / test | Valid test trajectories at 15 h |
|---|---|---|---|
| DMA | 10605 / 1481 / 1593 | 9144 / 1291 / 1453 | 252 |
| FG | 5866 / 1094 / 1279 | 4145 / 783 / 932 | 346 |

Counts describe the reference experiment. Acquire data through its providers under their applicable terms; the software license does not license third-party AIS data.

## Train and evaluate

```bash
python -m mast_kan train --dataset ct_dma --data-dir /path/to/ct_dma --out-dir runs/ct_dma/train
python -m mast_kan evaluate --checkpoint runs/ct_dma/train/model.pt --data-dir /path/to/ct_dma --out-dir runs/ct_dma/evaluate
```

For FG, use `--dataset florida_gulf_2025` when training and its data directory when evaluating. Training saves `model.pt`, `config.json`, `history.json`, `train.log` and `training.json`. Evaluation restores the checkpoint configuration and saves one `evaluation.json` containing all repeat metrics, per-step counts and mean/standard deviation. A weight file without its adjacent configuration requires `--config /path/to/config.json`.

Defaults: 18 observed points, at most 90 forecast points, training seed 42005, batch size 16, AdamW learning rate 0.0006, weight decay 0.1, up to 50 epochs, early-stop patience 5, AMP and EMA decay 0.999. Complete settings live in `mast_kan/config.py`. The checkpoint is selected by validation total loss and uses EMA weights. Model dimensions and original loss, masking and learning-rate behavior are preserved.

Evaluation uses N=16, top-k=10 and positional radius 40, with seeds 92026, 93035, 94044, 95053 and 96062. At each future step, it independently selects the smallest Haversine error among N candidates. MAE is the mean of these distances; RMSE is their root mean square. ADE averages the per-step MAEs equally over available steps. Metrics at 1, 3, 5, 10 and 15 h use the valid targets at that step. Standard deviations use the five sampling evaluations of one checkpoint. Set `--eval-seeds` or `--n-samples` to change evaluation controls; `--greedy` uses deterministic argmax predictions.

## Inference and synthetic example

Inference takes JSON containing `history`, an `[18, 4]` normalized history, and an optional `dataset` identifier. It outputs sampled future trajectories in latitude degrees, longitude degrees, SOG knots and COG degrees, without using future ground truth.

```bash
python -m mast_kan infer --checkpoint runs/ct_dma/train/model.pt --input history.json --out-dir runs/prediction
```

Generate entirely synthetic train/validation/test inputs and a valid history JSON:

```bash
python -m mast_kan example --dataset ct_dma --out-dir data/synthetic
python -m mast_kan train --dataset ct_dma --data-dir data/synthetic --out-dir runs/demo --max-epochs 1 --batch-size 4
python -m mast_kan infer --checkpoint runs/demo/model.pt --input data/synthetic/history.json --steps 6 --n-samples 2 --out-dir runs/demo_prediction
```

Synthetic data only checks execution; it does not reproduce the paper's performance.

## Citation and license

See `CITATION.cff` for manuscript authors and title. This implementation is licensed under MIT.

`mast_kan/kan.py` is the unmodified [efficient-kan implementation](https://github.com/Blealtan/efficient-kan/blob/7b6ce1c87f18c8bc90c208f6b494042344216b11/src/efficient_kan/kan.py), copyright 2024 Huanqi Cao. Its MIT license is retained in `mast_kan/KAN_LICENSE`. Mamba is provided by [state-spaces/mamba](https://github.com/state-spaces/mamba).
