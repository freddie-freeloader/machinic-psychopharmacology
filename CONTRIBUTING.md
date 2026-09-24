# Contributing

This repository is a research sketch. The numpy core is the source of truth
for the operators in `design-notes.md`; anything that talks to a real model
should sit behind the optional `hooks` extra.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Scope

- Keep the core (`operators`, `vectors`, `schedule`, `protocol`) free of
  PyTorch / transformers / vLLM.
- Treat drug names as analogies. Do not add clinical-dosing claims.
- New operators need a closed-form test: a fixed point, a no-op case, or
  a documented inequality (for example, ortho efficacy ≥ raw efficacy
  under overlap).
- Position-indexed steering belongs in `hooks.py`, matching the AISI
  vllm-lens protocol.

## Running on a real model

The intended host is [UKGovernmentBEIS/llm-self-steering](https://github.com/UKGovernmentBEIS/llm-self-steering).
Extract residual snapshots at the layer you care about, pass them through
`run_protocol`, and log `Metrics.as_dict()`. Do not vendor their 40-vector
library into this repo.
