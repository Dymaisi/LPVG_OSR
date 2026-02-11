# LPVG_OSR: Open Set Fault Diagnosis

LPVG_OSR is an open set recognition (OSR) project for bearing fault diagnosis. It builds LPVG graphs from vibration signals and evaluates known/unknown fault discrimination on CWRU and BDAE.

## Key Entry Point

- Training/evaluation: [scripts/train.py](scripts/train.py) (Hydra entry)

## Project Layout

```
LPVG_OSR/
├── README.md
├── QUICKSTART.md
├── requirements.txt
├── setup.py
│
├── config/
│   ├── main.yaml
│   ├── dataset/
│   │   ├── cwru.yaml
│   │   ├── bdae.yaml
│   │   └── osr_configs/
│   ├── model/
│   │   ├── lpvg_mlp.yaml
│   │   ├── lpvg_gnn.yaml
│   │   └── lpvg_ep_dygat.yaml
│   └── method/
│       ├── openmax.yaml
│       ├── crosr.yaml
│       ├── arpl.yaml
│       └── evidential.yaml
│
├── data/
├── models/
├── core/
├── utils/
├── scripts/
└── outputs/
```

## Quick Usage

### 1) Install dependencies

```bash
cd LPVG_OSR
pip install -r requirements.txt
```

### 2) Dataset setup

- CWRU uses the `cwru` Python package to download data automatically.
- BDAE data should be placed at the path set in [config/dataset/bdae.yaml](config/dataset/bdae.yaml).

### 3) Run experiments

```bash
# CWRU
python scripts/train.py dataset=cwru

# BDAE
python scripts/train.py dataset=bdae
```

### 4) Switch model/method

```bash
# GNN + OpenMax
python scripts/train.py dataset=cwru model=lpvg_gnn method=openmax

# MLP + ARPL
python scripts/train.py dataset=bdae model=lpvg_mlp method=arpl

# LPVG-EP-DyGAT + Evidential
python scripts/train.py dataset=cwru model=lpvg_ep_dygat method=evidential
```

## Running OSR Task Configs (24 presets)

The directory [config/dataset/osr_configs](config/dataset/osr_configs) contains 24 OSR task definitions (CWRU 12 + BDAE 12). You can run them directly via Hydra subgroup overrides:

```bash
python scripts/train.py dataset=osr_configs/cwru_easy_rpm1797
python scripts/train.py dataset=osr_configs/bdae_medium_domainF
```

If you need to create your own task, add a new YAML under [config/dataset](config/dataset) and run:

```bash
python scripts/train.py dataset=your_config_name
```

## Where results go

- logs: outputs/logs/
- checkpoints: outputs/checkpoints/
- results: outputs/results/

## Docs

- [docs/OSR_CONFIGS_README.md](docs/OSR_CONFIGS_README.md)
- [docs/LPVG_Theory.md](docs/LPVG_Theory.md)
- [docs/OSR_Methods.md](docs/OSR_Methods.md)
- [docs/LPVG_EP_DyGAT_IMPLEMENTATION.md](docs/LPVG_EP_DyGAT_IMPLEMENTATION.md)
