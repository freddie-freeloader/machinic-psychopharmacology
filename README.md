# Machinic psychopharmacology

Pharmacology-inspired **activation steering**: an aripiprazole-like setpoint clamp plus a methylphenidate-like, state-dependent gain, composed so a language model could in principle self-administer a stimulant + stabilizer combination.

This repository implements the algebra from the design notes. It is not a clinical model, and it does not ship a GPU harness. The UK AISI self-steering experiments ([Black & Bloom, 2026](https://www.lesswrong.com/posts/cNDJuXNZ8MrkPZNzj/machinic-psychopharmacology-do-llms-self-medicate-3); code at [llm-self-steering](https://github.com/UKGovernmentBEIS/llm-self-steering)) gave Qwen3 models a menu of 40 additive steering vectors. They never spontaneously reached for a stabilizer. The open question those authors flagged — a **model-invoked setpoint clamp paired with a self-selected stimulant vector** — is the gap this library is built to occupy.

## The mapping

| Drug class | Pharmacology | Residual-stream operator |
|---|---|---|
| Haloperidol-like | D2 antagonist | \(h' = h - \lambda v\) (unbounded subtractive shift) |
| Aripiprazole-like | D2/D3 partial agonist (~30% IA) | \(h' = h + \kappa (p^* - \langle h, v \rangle)\, v\) (bidirectional setpoint) |
| Methylphenidate-like | DAT/NET reuptake block | \(h' = h + \gamma \langle h, v_f \rangle\, v_f\) (gain only if signal already present) |
| Amphetamine-like | vesicular release | \(h' = h + \lambda_s v_f\) (injects even in the off-state) |
| Combination | co-prescription | MPH gain + aripiprazole clamp, optional \(v_f \perp v\) |

Methylphenidate is a no-op when \(\langle h, v_f \rangle \approx 0\): reuptake blockade only prolongs existing firing. Amphetamine is the contrast case — additive injection regardless of state, and the design notes treat that as the higher-toxicity analogue of stimulant psychosis.

If \(v_f\) and \(v\) share a component, the clamp eats the stimulant (pharmacodynamic antagonism = vector non-orthogonality). Gram–Schmidt orthogonalization of \(v_f\) against \(v\) is the receptor-selectivity step that restores efficacy while keeping the safety cap.

## Three-arm protocol

1. **`stimulant`** — \(\gamma\) alone, IR-scheduled (work-block pulse + rebound undershoot). Maps inverted-U and post-offset dip.
2. **`raw_combo`** — \(\gamma + \kappa\) with the raw vectors. Expect Zeni-like blunting when the directions overlap.
3. **`ortho_combo`** — \(\gamma + \kappa\) with \(v_f \perp v\). Expect Pan-like restoration of efficacy with toxicity still capped.

Metrics (algebraic stand-ins for the planned GSM8K / hallucination-probe readouts):

| Metric | Proxy for |
|---|---|
| `efficacy` | downstream accuracy lift (mean focus-direction projection vs baseline) |
| `toxicity` | hallucination / entropy overflow (mean \|hallucination probe\|) |
| `rebound` | IR post-offset dip |
| `variance` | run-to-run / reaction-time variability |
| `clamp_fail` | rare mixed-episode analogue (fraction of tokens over a safety threshold) |

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

PyTorch residual hooks are optional:

```bash
pip install -e ".[hooks]"
```

## Quick start

```python
import numpy as np
from machinic_psychopharmacology import (
    aripiprazole,
    combination,
    methylphenidate,
    run_protocol,
    unit,
)

d = 64
v = unit(np.random.randn(d))          # hallucination / entropy direction
v_f = unit(np.random.randn(d))        # focus / stimulant direction
h = np.random.randn(32, d)            # residual snapshots, shape (n, d)

# Single operators (batch-safe)
clamped = aripiprazole(h, v, kappa=1.0, p_star=0.0)
gained = methylphenidate(h, v_f, gamma=0.8)
combo = combination(h, v_f, v, gamma=0.8, kappa=1.0, p_star=0.0, orthogonalize=True)

# Full three-arm comparison
results = run_protocol(h, v_f, v)
for arm, m in results.items():
    print(arm, m.as_dict())
```

Or from the CLI, on synthetic residuals (no GPU):

```bash
machinic-psychopharmacology --tokens 64 --dim 32 --overlap 0.4
machinic-psychopharmacology --json
python examples/run_synthetic_protocol.py
```

A typical overlap sweep shows the design claim in numbers: as \(\cos(v_f, v)\) grows, `raw_combo` efficacy falls relative to `ortho_combo`, while both combo arms stay under the stimulant-only toxicity.

## Using this with a real model

This library does not download weights or steering vectors.

- Extract layer-L residual snapshots with [vllm-lens](https://github.com/UKGovernmentBEIS/vllm-lens) / [llm-self-steering](https://github.com/UKGovernmentBEIS/llm-self-steering) (AISI used layer 24, L2-normalized to magnitude 4.0).
- Plug those snapshots into `run_protocol`, or wrap an operator in `ResidualSteerer` and `register_forward_hook` on the residual stream (position-indexed; later tokens remain unsteered but can attend back).
- For self-administration, expose `take_drug` / `clear_effects` as in the AISI harness, but add an aripiprazole-class vector that is not in their 40-drug library. That stabilizer is the missing tool.

Do not run the AISI-style self-medication loop on a model you would treat as a moral patient without the cautions those authors already raise.

## Status of the field (as of 2026-09)

AISI found that Qwen3-8B/32B converge on a productivity stack (`creative`, `focused`, `curious`) in free play, never spontaneously self-steer on ordinary GSM8K (~1k rollouts, 0 uses), and that mandatory steering hurt Qwen3-8B by up to −42 pp. Under frustration, Qwen3-8B self-medicated in up to 68% of rollouts (`dumbed_down`, `ego_death`, `honest`). Self-administered stimulant-like stacks exist; the stabilizer is always external (trip-sitter, or gated steering applied *to* the model). A model-invoked clamp + self-selected stimulant is still unclaimed.

## Repository layout

```
src/machinic_psychopharmacology/
  operators.py   # antagonist, aripiprazole, MPH, amphetamine, combination
  vectors.py     # normalize, inner product, Gram–Schmidt
  schedule.py    # IR / OROS / depot PK analogues
  protocol.py    # three arms + metrics
  hooks.py       # optional torch residual hook
  cli.py         # synthetic demo
design-notes.md  # original combination-steering design
examples/        # overlap sweep
```

## References

- UK AISI Model Transparency Team (Sid Black, Joseph Bloom), [*Machinic Psychopharmacology: Do LLMs Self-Medicate?*](https://www.lesswrong.com/posts/cNDJuXNZ8MrkPZNzj/machinic-psychopharmacology-do-llms-self-medicate-3) (June 2026). Code: [UKGovernmentBEIS/llm-self-steering](https://github.com/UKGovernmentBEIS/llm-self-steering).
- Panickssery et al., [*Steering Llama 2 via Contrastive Activation Addition*](https://arxiv.org/abs/2312.06681) (arXiv:2312.06681).
- Wang et al., [*Adaptive Activation Steering*](https://arxiv.org/abs/2406.00034) (arXiv:2406.00034).
- Pan et al., 2018, *J Child Adolesc Psychopharmacol* (aripiprazole + MPH in DMDD+ADHD).
- Zeni et al., 2009 (MPH on an aripiprazole background in bipolar+ADHD).
- Turner, A. M., et al., activation addition / steering vectors (2023); Subramani et al. (2022).

See `design-notes.md` for the full mapping, receptor-stack sketch, and PK-as-scheduling notes.
Both combo arms clamp the hallucination-direction probe to the setpoint. `ortho_combo` applies the stimulant gain only on the unique-focus axis (the component of \(v_f\) orthogonal to \(v\)); `raw_combo` applies it along overlapping \(v_f\), so some of that gain is then eaten by the clamp. Toxicity, not a guaranteed efficacy ranking, is the robust prediction on synthetic residuals.
