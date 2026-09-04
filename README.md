# Fine-tuning a Tiny LLM for Sentiment Analysis
*Started Sep 2024 — Georgia Tech, CS 7643 Deep Learning*

Group project: how far can a small model (SmolLM2-135M) go on IMDb binary sentiment analysis?

Result: ~90% accuracy after fine-tuning, vs. zero-shot / few-shot baselines.

## Contents

- `proposal/` — project proposal
- `inital-test-code/` — simple `transformers` inference script + PACE-ICE slurm example
- `environments/` — conda ymls (or just use a clean PyTorch + CUDA env with `transformers`)
- `experiments/` — two notebooks:
  - `finetune_smolLM-135.ipynb` — fine-tuning (annotated, start here)
  - `default_all_smolLM.ipynb` — zero-shot / few-shot benchmarking
- `DL-Final-Project-Report/` — final report
