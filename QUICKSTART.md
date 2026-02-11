# OSR Quickstart

## 1) Install dependencies

```powershell
cd C:\Users\Maisi\Desktop\MouseWithoutBorders\SF-PGA\LPVG_OSR
pip install -r requirements.txt
```

If you plan to use CWRU:

```powershell
pip install cwru
```

## 2) Set dataset paths

- CWRU: automatic download via `cwru` package.
- BDAE: update `data_root` in [config/dataset/bdae.yaml](config/dataset/bdae.yaml).

## 3) Run a single experiment

```powershell
# CWRU
python scripts/train.py dataset=cwru

# BDAE
python scripts/train.py dataset=bdae
```

## 4) Run a preset OSR config

```powershell
python scripts/train.py dataset=osr_configs/cwru_easy_rpm1797
python scripts/train.py dataset=osr_configs/bdae_medium_domainF
```

## 5) Change model or method

```powershell
python scripts/train.py dataset=cwru model=lpvg_gnn method=openmax
python scripts/train.py dataset=bdae model=lpvg_mlp method=arpl
python scripts/train.py dataset=cwru model=lpvg_ep_dygat method=evidential
```

## 6) Common knobs

- Training: `training.num_epochs`, `training.batch_size`, `training.learning_rate` in [config/main.yaml](config/main.yaml)
- OSR split: `known_classes`, `unknown_classes`, `train_ratio`, `val_ratio`, `test_ratio` in dataset configs

## Results

- logs: `outputs/logs/`
- checkpoints: `outputs/checkpoints/`
- results: `outputs/results/`
