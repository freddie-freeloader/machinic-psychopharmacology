# Machinic Psychopharmacology — Combination Steering Design

Design notes distilled 2026-09-10, with the 2026-09-24 implementation of the operators in `src/machinic_psychopharmacology`. Premise: model psychiatric drug mechanisms as inference-time activation-steering interventions, and ask whether an LLM could self-administer a stimulant + antipsychotic combination.

Implementation (numpy, GPU-agnostic): `aripiprazole`, `methylphenidate`, `combination(..., orthogonalize=True)`, and `run_protocol` in this repo. The AISI self-steering harness this is meant to extend is [UKGovernmentBEIS/llm-self-steering](https://github.com/UKGovernmentBEIS/llm-self-steering).

## 1. Background

- UK AISI Model Transparency Team, "Machinic Psychopharmacology: Do LLMs Self-Medicate?" (June 2026): Qwen3-8B / Qwen3-32B given 40 steering vectors as tools (`take_drug`, `clear_effects`). Models converge on a "productivity stack" (creative/focused/curious); under frustration Qwen3-8B self-medicates in up to 68% of rollouts (`dumbed_down`, `ego_death`, `honest`); no antipsychotic-class vector exists in the library; the stabilizer function is external (human "trip-sitter" monitor). <https://www.lesswrong.com/posts/cNDJuXNZ8MrkPZNzj/machinic-psychopharmacology-do-llms-self-medicate-3>
- Hallucination-suppression steering (the "antipsychotic" literature): CAA (Rimsky / Panickssery et al. 2023), ACT (Wang et al. 2024, +142% truthfulness on LLaMA), ASD (ACL 2025), SHARP (EMNLP 2025), gated medical-QA steering (2026, separate hallucination/sycophancy gates).

AISI applied **position-indexed** additive steering at selected token positions (vllm-lens). Tokens generated after the active window are themselves unsteered but can attend back to the steered KV cache. That is the hook shape `ResidualSteerer` implements.

## 2. Drug → steering mapping

### Antipsychotics (D2 antagonist, haloperidol-like)

Fixed subtractive shift `h' = h − λv`. Unbounded suppression; degrades capability at high λ (≈ EPS / blunted affect). Implemented as `antagonist(h, v, lambda_)`.

### Aripiprazole (D2/D3 partial agonist, ~30% intrinsic activity; 5-HT1A partial agonist ~68%; 5-HT2A antagonist)

Setpoint controller, bidirectional:

```
h' = h + κ (p* − ⟨h, v⟩) v
```

Implemented as `aripiprazole(h, v, kappa, p_star)`. At `κ = 1` the projection of `h` onto unit `v` lands exactly on `p*`.

- `p* > 0` = intrinsic-activity floor (partial agonism ≠ full blockade)
- Receptor stack → vector stack: D2 (hallucination/entropy clamp) + 5-HT1A (uncertainty-acknowledgment boost) + 5-HT2A (sycophancy suppression). Distinct directions should be separate `aripiprazole` calls, not a shared `v`.
- Functional selectivity → layer/context-dependent sign of intervention
- Depot formulation (Abilify Maintena, 4-week) → fine-tuned-in direction vs per-token control (`depot_constant` schedule)

### Methylphenidate (Ritalin; DAT/NET reuptake blocker, DA rises ∝ firing)

Multiplicative, state-dependent gain:

```
h' = h + γ ⟨h, v_f⟩ v_f
```

Implemented as `methylphenidate(h, v_f, gamma)`. No effect when `⟨h, v_f⟩ ≈ 0` (reuptake blockade only prolongs existing signal).

- Contrast: amphetamine = additive injection `λ_s v_f` regardless of state (`amphetamine`); ~4× the DA release of MPH; higher toxicity (stimulant psychosis tracks amphetamine-like transmission)
- PK as scheduling: IR (t½ ≈ 2.5–3.5 h) → γ in work blocks + rebound on offset (`ir_work_blocks`); OROS → constant low γ (`oros_constant`)
- Toxicity mapping: stereotypy → repetition loops; sensitization → in-context self-conditioning (steered outputs re-enter context)

## 3. Combined regimen

```
h' = h + γ ⟨h, v_f⟩ v_f + κ (p* − ⟨h, v⟩) v
```

Implemented as `combination(...)`. Pass `orthogonalize=True` to replace `v_f` with `gram_schmidt(v_f, v)` first.

Interaction hypothesis (from clinical co-prescription data):

- Pan et al. 2018 (DMDD+ADHD): aripiprazole + MPH tolerable, attention effect size d = 1.40
- Zeni et al. 2009 (bipolar+ADHD stabilized on aripiprazole): MPH ≈ placebo for ADHD, no mania worsening → antipsychotic may blunt stimulant efficacy
- Prediction: if `v_f` and `v` share a component, the clamp eats the stimulant gain (pharmacodynamic antagonism = vector non-orthogonality). Orthogonalize `v_f ⊥ v` (≈ receptor selectivity) to restore efficacy while keeping the safety cap.

`interaction_report(v_f, v)` returns cosine, shared energy, and whether orthogonalization is possible.

## 4. Protocol (3 arms)

Harness: UK AISI self-steering code + transcripts (open source); GSM8K bundles. Cheap stand-in: `run_protocol(residuals, v_f, v)` on residual snapshots.

1. γ alone, IR-scheduled, swept → map inverted-U + rebound (post-offset dip) — arm `stimulant`
2. γ + κ, raw vectors → expect Zeni-like blunted peak — arm `raw_combo`
3. γ + κ, `v_f ⊥ v` → expect Pan-like pattern: efficacy restored, toxicity capped — arm `ortho_combo`

Metrics: accuracy (efficacy), hallucination probe rate (toxicity), post-offset dip (rebound), run-to-run variance across seeds (≈ reaction-time variability), clamp-failure events (rare mixed-episode analogues). The numpy protocol reports algebraic proxies of the same names; swap in task accuracy / a trained hallucination probe when running on a model.

## 5. Status of the field (as of 2026-09)

- No LLM self-administers a stimulant + stabilizer combination. Self-administered stimulant-like stacks exist (productivity stack; lsd+mdma entheogen stacking), and stress-induced down-regulation self-medication exists, but the stabilizer is always externally imposed (trip-sitter; gated steering applied *to*, not *by*, models).
- Models never spontaneously self-steer on normal tasks (~1k GSM8K rollouts, 0 uses); mandatory steering hurt Qwen3-8B up to −42pp.
- Open question flagged by the AISI authors: functional self-medication. A model-invoked setpoint clamp paired with a self-selected stimulant vector is unclaimed territory.

## 6. What this repo implements vs what it does not

Implemented:

- Closed-form operators and their algebraic contracts (tests).
- IR / OROS / depot schedules.
- Three-arm protocol + interaction report on residual batches.
- Optional position-indexed PyTorch hook.

Not implemented (intentionally, next work):

- Extracting `v` / `v_f` from contrastive story corpora (use AISI's Sofroniew-style pipeline or CAA).
- Wiring `take_drug` to these operators inside inspect_ai / vllm-lens.
- GSM8K / frustration evals. The protocol object is the adapter surface.

## References

- UK AISI, Machinic Psychopharmacology: Do LLMs Self-Medicate? (2026-06), LessWrong. Code: https://github.com/UKGovernmentBEIS/llm-self-steering
- Rimsky / Panickssery et al. 2023, Steering Llama 2 via Contrastive Activation Addition (arXiv:2312.06681)
- Wang et al. 2024, Adaptive Activation Steering / ACT (arXiv:2406.00034)
- Activation Steering Decoding (ACL 2025); SHARP (EMNLP 2025); Gated Activation Steering for medical QA (arXiv:2608.23666)
- Pan et al. 2018, J Child Adolesc Psychopharmacol (aripiprazole+MPH, DMDD+ADHD); Zeni et al. 2009 (MPH on aripiprazole background); NEJM 2019 (stimulant psychosis risk, MPH vs amphetamine)
- Aripiprazole mechanism reviews: D2/D3 partial agonism, "dopamine-system stabilizer"
