# Task 2: Unsupervised Domain Adaptation

Shared PACS protocol for Tasks 2 and 3 lives in `shared/`. Method losses live in `task2/methods/`; `train.py` is the common loop. Target class labels are used only in `evaluate_final.py`.

## Data

Place PACS as:

```
data/pacs/
  art_painting/{dog,elephant,giraffe,guitar,horse,house,person}/
  cartoon/...
  photo/...
  sketch/...
```

Then write the shared source splits (seed 6304, stratified 80/20 per source domain):

```bash
python -m shared.pacs_protocol --data-root data/pacs
```

This fills `shared/splits/pacs_sketch_seed6304.json`. Reuse that file in Task 3. Do not change splits after they are generated.

## Train (source-validation selection only)

```bash
python task2/train.py --config task2/configs/source_only.yaml
python task2/train.py --config task2/configs/dan.yaml
python task2/train.py --config task2/configs/dann.yaml
python task2/train.py --config task2/configs/cdan.yaml
```

Checkpoints are selected by mean macro-F1 on the three source validation domains. Sketch labels are not used here. The source-only checkpoint is also the Task 3 ERM baseline.

BatchNorm running mean/variance stay at ImageNet values; only γ and β are trained.

## Final evaluation (Sketch labels allowed)

```bash
python task2/evaluate_final.py
```

Writes `task2/results/final_comparison.json` with source-val metrics, Sketch accuracy/macro-F1, domain separability, and per-class changes vs source-only.

## Controlled study

DAN λ_MMD or DANN max GRL strength, without using Sketch to pick the main setting:

```bash
python task2/train.py --config task2/configs/dan.yaml --lambda-mmd 0.1 --run-name dan_lambda_0.1
python task2/train.py --config task2/configs/dan.yaml --lambda-mmd 1.0 --run-name dan_lambda_1
python task2/train.py --config task2/configs/dan.yaml --lambda-mmd 10 --run-name dan_lambda_10
```

or

```bash
python task2/train.py --config task2/configs/dann.yaml --grl-max 0.25 --run-name dann_grl_0.25
python task2/train.py --config task2/configs/dann.yaml --grl-max 0.5 --run-name dann_grl_0.5
python task2/train.py --config task2/configs/dann.yaml --grl-max 1.0 --run-name dann_grl_1
```

## Dependencies

`torch`, `torchvision`, `pyyaml`, `scikit-learn`, `Pillow`, `numpy`
