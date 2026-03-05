# 2026 Setup Notes (Ant, Humanoid, Pendulum)

This repository was originally written against older Gym and TF1 APIs.
The current code includes a compatibility shim (`maxent_compat.py`) that maps:

- `Ant-v2` -> `Ant-v4`
- `Humanoid-v2` -> `Humanoid-v4`
- `Pendulum-v0` -> `Pendulum-v1`

## 1) Create environment

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements-modern.txt
```

## 2) Quick smoke test

```bash
python smoke_check.py --steps 3
```

This should create and step Pendulum, Ant, and Humanoid without API errors.

## 3) Run experiments

Ant:
```bash
python ant/ant_collect_sac.py --env="Ant-v4" --exp_name=ant_experiment --T=10000 --n=20 --l=2 --hid=300 --epochs=20 --episodes=30 --gaussian --reduce_dim=5 --geometric
```

Humanoid:
```bash
python humanoid/humanoid_collect_sac.py --env="Humanoid-v4" --exp_name=humanoid_experiment --T=50000 --n=20 --l=2 --hid=300 --epochs=30 --episodes=30 --geometric
```

Pendulum:
```bash
python base/collect_baseline.py --env="Pendulum-v1" --T=200 --train_steps=200 --episodes=200 --epochs=15 --exp_name=pendulum_test
```

## Notes

- If you have a local checkout of SpinningUp and prefer that over pip, keep it on `PYTHONPATH`.
- Video recording now uses Gym's `RecordVideo` where available.
