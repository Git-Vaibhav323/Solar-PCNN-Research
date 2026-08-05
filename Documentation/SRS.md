# Software Requirements Specification (SRS)

**Project:** Solar-PCNN Research — Physics-Constrained Neural Networks for Solar PV Forecasting  
**Document type:** Requirements & delivered-system specification (research software)  
**Version:** 1.0  
**Status:** Phases 1–9 complete (architecture frozen; audit 178/178 PASS)  
**Last updated:** August 2026  
**Related docs:** [`README.md`](../README.md), [`summary.md`](summary.md), [`interpret.md`](interpret.md), [`phase9_report.md`](phase9_report.md)

> **How to use this file.** This is a working SRS you can lift into a formal college / journal / thesis template. It describes what the *system was required to do*, what it *actually does today*, and how you know it works. Where classic SRS language assumes a commercial product, this document stays honest: this is a **research pipeline and evidence pack**, not a production forecasting service.

---

## 1. Introduction

### 1.1 Purpose of the document

This Software Requirements Specification records:

1. **What problem** the Solar-PCNN research system set out to solve,
2. **What the software must do** (data prep, train, evaluate, audit, report),
3. **What quality bar** counts as “done” for a publication-oriented study,
4. **What was actually delivered** across Phases 1–9.

It exists so a reader (you, a co-author, a reviewer, or an examiner) can answer: *“If I rebuild this from scratch, what am I obligated to implement, and what evidence proves it works?”*

It is **not** a user-manual for a deployed app, and it does **not** claim real-time grid dispatch capability.

### 1.2 Scope of the project

**In scope**

- Reproduce a Physics-Constrained Neural Network (PCNN) for solar PV in PyTorch.
- Diagnose where the baseline fails (especially cross-domain transfer to Renewables.ninja).
- Design and freeze an **Adaptive Residual PCNN (AR-PCNN)** with calibrated uncertainty.
- Add **cross-domain multi-task learning** so the model learns an external system (ninja kW) without rewriting the physics path.
- Validate with ablation, integrity checks, independent metric audit, λ-sensitivity, and multi-seed stability.
- Keep every headline number traceable to machine-written CSV / JSON / checkpoints — not prose alone.

**Out of scope**

- Live weather API ingestion or SCADA integration.
- Production MLOps (model registry, A/B serving, SLA dashboards).
- Claiming near-unity NSRDB R² as a forecasting breakthrough (labels are physics-derived).
- Numeric alignment to the source paper’s absolute MAE (e.g. 0.0213) when unit/capacity are ambiguous.

**Primary scientific claim the software must support**

> Masked multitask learning lifts Renewables.ninja R² from ~0.5 → ~0.86 (val-holdout ~0.81).  
> Five-seed mean at λ_ext = 1.0: **0.8772 ± 0.0140**.

**Explicit non-claim**

> NSRDB R² ≈ 1.0 is a **physics-fidelity consistency check**, not the lead result.

### 1.3 Intended audience

| Audience | Why they read this |
|----------|-------------------|
| Project authors / students | Build or extend the pipeline; write thesis/SRS chapters |
| Academic supervisors / examiners | See requirements, constraints, and acceptance evidence |
| Co-authors / reviewers | Trace claims to artifacts and phases |
| Future maintainers | Understand frozen architecture and “do not break Phase 3–8” rule |

### 1.4 Definitions, acronyms, and abbreviations

| Term | Meaning |
|------|---------|
| **PCNN** | Physics-Constrained Neural Network (paper baseline, dual-branch) |
| **AR-PCNN** | Adaptive Residual PCNN — `ŷ = clamp(α · P_physics + r)` |
| **CD / Multitask** | Cross-domain model: shared encoder + external kW head |
| **NSRDB** | NREL National Solar Radiation Database (hourly weather + derived physics target) |
| **Ninja / Renewables.ninja** | External PV electricity series (2019, 10 kW system) |
| **GHI / DNI / DHI** | Global / Direct / Diffuse Horizontal Irradiance |
| **SZA** | Solar Zenith Angle |
| **α (gate)** | Learned trust weight on the analytical physics prior (0…1) |
| **r (residual)** | Learned correction added to gated physics |
| **λ_external / λ_ext** | Weight on masked ninja MSE in the multitask loss |
| **PICP** | Prediction Interval Coverage Probability (target ≈ 0.95) |
| **MPIW** | Mean Prediction Interval Width |
| **MC Dropout** | Monte Carlo Dropout for epistemic uncertainty |
| **Conformal UQ** | Validation-set scaling of MC intervals to hit target PICP |
| **MAE / RMSE / R²** | Mean Absolute Error / Root Mean Squared Error / coefficient of determination |
| **MAPE / sMAPE / nRMSE** | Relative-error metrics; prefer sMAPE/nRMSE near zero irradiance |
| **TS250** | Tata Power Solar 250 W panel model used for physics prior |
| **VITCC** | Paper site citation (~12.84°N, 80.15°E); our NSRDB point is ~25 km NE |
| **Feasible** | Fraction of predictions inside [0, capacity]; **1.0 by hard clamp** (architecture, not a learned score) |
| **Phase 3–8 frozen** | Baseline checkpoints and metrics must not be overwritten by Phase 9 experiments |
| **SRS** | Software Requirements Specification (this document) |

### 1.5 References

1. Source paper PDF: [`Solar.pdf`](Solar.pdf) — *Physics-constrained neural networks for reliable solar power forecasting with uncertainty quantification* (Results in Engineering, 2026) — reproduction target.
2. Project README: [`../README.md`](../README.md)
3. Living status: [`summary.md`](summary.md)
4. How to read numbers: [`interpret.md`](interpret.md)
5. Phase reports: `phase4_limitations.md` … `phase9_report.md`
6. Configs: `configs/preprocess.yaml`, `configs/train_pcnn.yaml`, `configs/train_ar_pcnn.yaml`, `configs/train_ar_pcnn_cd.yaml`
7. Dependencies: [`requirements.txt`](requirements.txt)
8. Raw evidence: `artifacts/`, `data/processed/`, `data/figures/`

### 1.6 Document overview

| Section | Contents |
|---------|----------|
| **2** | Big-picture product description, users, environment, constraints |
| **3** | Functional, non-functional, interface, and data requirements (with phase mapping) |
| **4** | Quality attributes (reliability through interoperability) |
| **5** | Project constraints (time, stack, scientific honesty rules) |
| **6** | Acceptance criteria — what “done” looked like for Phases 1–9 |
| **7** | Appendices: glossary, delivered inventory, known open issues |

---

## 2. Overall Description

### 2.1 Product perspective

This system sits in a **research toolchain**, not inside a utility SCADA stack.

```text
[NSRDB hourly files] ──┐
                       ├──► Phase 2 pipeline ──► processed splits + scaler + physics target
[Renewables.ninja CSV]─┘                              │
                                                      ▼
                         Phase 3–6 training / eval / UQ / Case-1
                                                      │
                                                      ▼
                         Phase 7 ablation + figures ←── frozen checkpoints
                                                      │
                                                      ▼
                         Phase 8 independent metric audit
                                                      │
                                                      ▼
                         Phase 9 analyses (λ, seeds, gate, regimes, …)
                              └──► Documentation/*.md + artifacts/phase9/
```

Upstream: public weather / PV datasets.  
Downstream: reports, figures, and tables intended for a journal-style paper — not a REST API for end customers.

### 2.2 Product functions (what the software does)

At a human level, the product must let a researcher:

1. **Ingest & prepare** NSRDB + ninja data with chronological splits and train-only scaling.
2. **Train** baseline PCNN, AR-PCNN, and cross-domain multitask / finetune variants from YAML configs.
3. **Evaluate** on NSRDB test hours and on ninja (Case-1 / full-2019 / val-holdout), with clear **W vs kW** units.
4. **Quantify uncertainty** via MC Dropout + conformal calibration toward PICP ≈ 0.95.
5. **Ablate** models A–E under one protocol so each contribution is isolated.
6. **Audit** published-looking numbers by recomputing metrics from saved prediction arrays.
7. **Stress-test** λ_external, random seeds, regimes, noise, and feature importance — writing only into `artifacts/phase9/`.
8. **Document** results so claims can be verified from CSV/JSON, including `lambda_external` per run.

### 2.3 User classes and characteristics

| User class | Skill assumed | Typical actions |
|------------|---------------|-----------------|
| **Research developer** | Python, PyTorch, basic ML | Train, debug losses, regenerate reports |
| **Analyst / co-author** | Interpreting metrics & figures | Read `Documentation/`, check CSVs, write paper text |
| **Examiner / reviewer** | Domain + methods literacy | Trace claims → artifacts; check caveats |
| **Future student** | Can follow README setup | Reproduce Phase 7–9 with frozen checkpoints |

There is **no** end-customer “dashboard user.” CLI + markdown reports are the interface.

### 2.4 Operating environment

| Layer | Requirement / delivered choice |
|-------|--------------------------------|
| OS | Windows 10/11 (primary dev), portable to Linux with path tweaks |
| Language | Python 3.x |
| Deep learning | PyTorch ≥ 2.1 (prefer project `.venv`; CUDA GPU when available) |
| Data / viz | pandas, numpy, scikit-learn, matplotlib, seaborn, scipy, PyYAML, joblib, tqdm |
| Shell | PowerShell examples in docs; module entry points (`python -m src.…`) |
| Storage | Local repo: `data/`, `artifacts/`, `Documentation/` |

**Known environment constraint:** System / Microsoft Store Python often breaks Torch (`torchgen.model` errors). Requirement: use the **project virtual environment**.

### 2.5 Design and implementation constraints

1. **Reproduce first, invent later** — Phase 3 baseline frozen before AR-PCNN / CD changes.
2. **Architecture frozen** after Phase 5/6 (encoder 14→128→64→32, gate + residual; CD adds external head only).
3. **Phase 9 isolation** — new training writes under `artifacts/phase9/` only; do not overwrite Phase 3–8 checkpoints.
4. **Chronological splits** — no shuffle; no future leakage into train.
5. **Scaler fit on train only.**
6. **Units discipline** — NSRDB absolute errors in **W** (1× TS250); ninja in **kW** (10 kW / 40×250 W). Every table row must label units.
7. **No guessed conversion** of paper MAE ≈ 0.0213 into our tables (capacity/unit not reconcilable).
8. **Site disclosure** — NSRDB at 13.06°N, 80.25°E; paper VITCC ~12.84°N, 80.15°E (~25 km); ninja file coords slightly different again.
9. **Feasibility = 1.0 by clamp** must be disclosed as architectural, not a learned achievement.
10. Prefer **sMAPE / nRMSE** over MAPE for near-zero irradiance hours.

### 2.6 User documentation

| Document | Role |
|----------|------|
| `README.md` | Architecture, how to run, figure gallery, verification commands |
| `Documentation/summary.md` | Plain-language living status |
| `Documentation/interpret.md` | How to read final numbers without fooling yourself |
| `Documentation/phase4…9_*.md` | Phase-level evidence |
| `Documentation/SRS.md` | This requirements specification |
| Config YAML | Machine-readable training / preprocess contracts |
| `artifacts/**/summary_*.json`, `*.csv` | Ground-truth run records |

### 2.7 Assumptions and dependencies

**Assumptions**

- NSRDB and Renewables.ninja files are available under the configured `dataset/` / `data/processed/` paths.
- Analytical PV physics (TS250-style) is an acceptable *prior*, knowing NSRDB targets are themselves physics-bounded.
- A single Chennai-area site is enough for the study design (site mismatch vs paper is disclosed, not “fixed away”).
- GPU is preferred but CPU training is allowed (slower).

**Dependencies**

- External datasets (NSRDB, Renewables.ninja) and their licenses / citation norms.
- PyTorch + scientific Python stack ([`requirements.txt`](requirements.txt)).
- Source paper methods as the reproduction baseline ([`Solar.pdf`](Solar.pdf)).

**Dependency risk already accepted**

- Paper site name/coords (VITCC vs Aralvaimozhi-style citations) and MAE unit ambiguity remain **open for methods disclosure**, not silently patched in code.

---

## 3. Specific Requirements

Requirements are tagged **FR** (functional), **NFR** (non-functional), **IR** (interface), **DR** (data).  
**Status** reflects delivery as of Phases 1–9.

### 3.1 Functional requirements

#### 3.1.1 Data preparation (Phase 1–2)

| ID | Requirement | Status |
|----|-------------|--------|
| FR-01 | Ingest NSRDB hourly data for years 2016–2020 at configured site | Done |
| FR-02 | Ingest Renewables.ninja 2019 PV electricity series | Done |
| FR-03 | Clean, merge, and engineer 14 features (12 met + hour_sin/hour_cos) | Done |
| FR-04 | Compute analytical physics prior / bounded target (`target_power`, `P_theoretical`) for TS250 | Done |
| FR-05 | Chronological train/val/test split (~28,062 / 7,016 / 8,770); shuffle disabled | Done |
| FR-06 | Fit z-score scaler on **train only**; persist scaler for later eval | Done |
| FR-07 | Persist processed tables under `data/processed/` | Done |

#### 3.1.2 Baseline PCNN (Phase 3)

| ID | Requirement | Status |
|----|-------------|--------|
| FR-08 | Implement dual-branch PCNN in PyTorch per paper spirit | Done |
| FR-09 | Train with MSE + capacity + non-negativity penalties from YAML | Done |
| FR-10 | Evaluate NSRDB test metrics (MAE/RMSE/R²/…) in Watts | Done |
| FR-11 | Run Case-1 ninja evaluation (kW, 10 kW system) | Done |
| FR-12 | Provide MC Dropout uncertainty baseline | Done |
| FR-13 | Freeze baseline checkpoint for fair later comparison | Done |

#### 3.1.3 Limitation diagnosis (Phase 4)

| ID | Requirement | Status |
|----|-------------|--------|
| FR-14 | Analyse and document physics-label copying risk | Done |
| FR-15 | Document weak ninja transfer and over-wide UQ | Done |
| FR-16 | Produce a limitations memo used as redesign input | Done (`phase4_limitations.md`) |

#### 3.1.4 AR-PCNN + calibrated UQ (Phase 5)

| ID | Requirement | Status |
|----|-------------|--------|
| FR-17 | Implement AR-PCNN: shared encoder, residual head, gate α, capacity clamp | Done |
| FR-18 | Train AR-PCNN; keep architecture freezable | Done |
| FR-19 | Implement conformal calibration of MC intervals on validation | Done |
| FR-20 | Achieve calibrated test PICP near 0.95 (delivered ≈ 0.94) | Done |
| FR-21 | Compare PCNN vs AR-PCNN in unit-labeled tables | Done |

#### 3.1.5 Cross-domain learning (Phase 6)

| ID | Requirement | Status |
|----|-------------|--------|
| FR-22 | Add external kW head on shared encoder | Done |
| FR-23 | Multitask loss = physics loss + λ_external × masked ninja MSE | Done |
| FR-24 | Support warm-start from Phase-5 AR-PCNN | Done |
| FR-25 | Implement frozen-encoder finetune strategy (negative-result path) | Done |
| FR-26 | Report full-2019 and val-holdout ninja R² | Done (~0.86 / ~0.81 multitask; finetune fails) |
| FR-27 | Preserve NSRDB physics-path fidelity as consistency check | Done |

#### 3.1.6 Validation, ablation, audit (Phases 7–8)

| ID | Requirement | Status |
|----|-------------|--------|
| FR-28 | Ablation models A–E under identical protocol | Done |
| FR-29 | Integrity checks (no leakage, train-only scaler, ninja not in Phase-5 loss, etc.) | Done (11/11) |
| FR-30 | Publication figures (architecture, ablation, UQ, regimes, interpretability) | Done |
| FR-31 | Independent recalculation of metrics from saved arrays | Done |
| FR-32 | Magnitude-aware audit tolerances; ledger of pass/fail | Done (**178/178 PASS**) |
| FR-33 | Prefer sMAPE/nRMSE where MAPE is pathological | Done |

#### 3.1.7 Publication analyses (Phase 9)

| ID | Requirement | Status |
|----|-------------|--------|
| FR-34 | λ_external sweep {0, 0.05, 0.1, 0.25, 0.5, 1.0} with frozen architecture | Done |
| FR-35 | Multi-seed stability (seeds 42, 123, 456, 999, 2026) at recorded λ | Done (0.8772 ± 0.0140) |
| FR-36 | Persist `lambda_external` in launch YAML, train JSON, and runs CSV | Done |
| FR-37 | Gate / residual / importance / embedding / regime / noise / runtime analyses | Done |
| FR-38 | Isolate all Phase-9 outputs under `artifacts/phase9/` | Done |
| FR-39 | Assemble `phase9_report.md` from per-experiment notes | Done |

### 3.2 Non-functional requirements

| ID | Requirement | Notes / delivery |
|----|-------------|------------------|
| NFR-01 | **Reproducibility** — same config + seed → comparable metrics; artifacts retained | Seeds logged; summaries JSON |
| NFR-02 | **Traceability** — headline numbers must map to CSV/JSON, not markdown alone | Exp4 λ verified via YAML+JSON (7/24) vs CSV backfill (7/26) |
| NFR-03 | **Scientific honesty** — disclose circular labels, clamps, site offset, unit mismatch | Caveats in README / audit / interpret |
| NFR-04 | **Performance (research)** — full Phase-9 train suite completes on GPU in ~tens of minutes; `--skip-train` analysis faster | ~25 min full / ~4 min skip-train (observed) |
| NFR-05 | **Modularity** — models, training, analysis as separate packages under `src/` | Delivered layout |
| NFR-06 | **Non-destruction** — Phase 9 must not clobber Phase 3–8 artifacts | Separate `artifacts/phase9/` tree |
| NFR-07 | **Clarity of units** — every absolute metric labeled W or kW | Tables + generators |
| NFR-08 | **Usability for researchers** — runnable via `python -m …` and documented PowerShell steps | README §10 |

### 3.3 External interface requirements

#### 3.3.1 User interfaces

- **CLI / module runners** (primary): e.g. `python -m src.training.train_ar_pcnn_cd`, `python -m src.analysis.phase8_audit`, `python -m src.analysis.phase9.run_all`.
- **Markdown reports** under `Documentation/` and `artifacts/phase9/reports/`.
- **Figures** under `data/figures/` (PNG).
- No GUI / web dashboard required.

#### 3.3.2 Hardware interfaces

- Optional NVIDIA CUDA GPU via PyTorch `device: auto`.
- No specialized sensors; all inputs are files.

#### 3.3.3 Software interfaces

| Interface | Direction | Format |
|-----------|-----------|--------|
| NSRDB raw | In | CSV / dataset folder as configured |
| Renewables.ninja | In | CSV |
| Preprocess / train configs | In | YAML |
| Scaler | In/Out | `joblib` |
| Checkpoints | In/Out | PyTorch `.pt` |
| Metrics / predictions | Out | JSON, NPZ, CSV |
| Reports | Out | Markdown |

#### 3.3.4 Communication interfaces

- None required beyond local filesystem and optional Git remote for source control.
- Not a networked service.

### 3.4 Data requirements

| ID | Requirement | Detail |
|----|-------------|--------|
| DR-01 | NSRDB coverage | 2016–2020 hourly; site 13.06°N, 80.25°E (disclosed vs paper VITCC) |
| DR-02 | Ninja coverage | 2019 electricity; system 10 kW (40 × 250 W) |
| DR-03 | Feature vector | 14 inputs listed in train YAMLs |
| DR-04 | Targets | NSRDB: `target_power` (W); ninja: system kW with mask for missing hours |
| DR-05 | Split sizes | Train 28,062 / val 7,016 / test 8,770 (chronological) |
| DR-06 | Test integrity | 2020 test has **zero** ninja labels |
| DR-07 | Physics params | `p_rated_w=250`, capacity margin 1.1, gamma/NOCT per preprocess config |
| DR-08 | Artifact retention | Checkpoints, metrics, Phase-9 configs/summaries kept for audit |

---

## 4. Quality Requirements

### 4.1 Reliability

- Training must support early stopping and checkpointing of best val model.
- Evaluation scripts must load the frozen best checkpoint, not an arbitrary last epoch.
- Audit must fail loudly if recomputed metrics disagree beyond magnitude-aware tolerance.
- **Delivered:** Phase 8 **178/178** checks PASS; multi-seed all five seeds succeed on ninja (no silent averaging over failures).

### 4.2 Availability

- Research tool: “available” means a developer can recreate the environment and re-run analyses offline.
- No 24/7 uptime SLA.
- **Delivered:** documented venv setup; module entry points; `--skip-train` path when checkpoints exist.

### 4.3 Maintainability

- Clear package boundaries: `src/data`, `src/models`, `src/training`, `src/analysis`.
- Phase reports regenerable from code (compare_phase5/6, phase7_run, phase8_audit, phase9.report).
- Architecture freeze rule reduces accidental redesign during analysis phases.
- **Delivered:** Documentation index + interpret guide for future readers.

### 4.4 Security

- No authentication subsystem (local research use).
- Requirement: do not commit secrets; datasets may be large / separately obtained.
- Requirement: treat this as non-safety-critical offline analytics — still avoid executing untrusted pickles from unknown sources (`joblib`/checkpoint hygiene).
- **Delivered:** standard local-dev posture; no public API attack surface by design.

### 4.5 Safety

- Not a safety-critical control system.
- Soft requirement: do not present physics-clamp feasibility or NSRDB R²≈1 as operational guarantees for real plants.
- **Delivered:** explicit caveats in README, audit, and interpret docs.

### 4.6 Portability

- Python + PyTorch codebase; Windows primary, Linux feasible.
- Paths centralized in YAML; some absolute paths may appear inside generated JSON (machine-local) — portable *code*, machine-specific *artifact metadata*.
- **Delivered:** relative project layout; note absolute checkpoint paths in some summaries.

### 4.7 Interoperability

- Interoperates with standard scientific file formats (CSV, JSON, PNG, PyTorch).
- Does not require proprietary runtimes beyond the open Python ecosystem.
- Dataset providers (NSRDB, Renewables.ninja) are external; software consumes their exports.

---

## 5. Constraints

### 5.1 Budget limitations

- Academic / student research budget: rely on public datasets and local or institutional GPU.
- No paid SaaS ML platform required.
- Constraint accepted: single-site study; no multi-continent data purchase.

### 5.2 Time constraints

- Phased delivery (1→9) with freeze points so later work does not invalidate earlier comparisons.
- Phase 9 full retrain is slower (~25 min GPU); analysis-only mode required for iteration.
- Paper writing remains downstream of software acceptance (methods/results still to be authored).

### 5.3 Regulatory / ethical / scientific requirements

- Cite dataset providers and the source paper appropriately in any publication.
- Disclose site coordinate mismatch (~25 km) and unit/protocol differences vs published tables.
- Do not fabricate λ-sweep or multi-seed results; claims must match raw artifacts.
- Prefer val-holdout ninja R² when discussing generalization (full-2019 overlaps early train hours).

### 5.4 Technology stack restrictions

| Allowed | Restricted / avoided |
|---------|----------------------|
| Python, PyTorch, sklearn, pandas, matplotlib/seaborn | Random undocumented framework swaps mid-study |
| YAML configs for train/preprocess | Hard-coded magic metrics in markdown as sole source of truth |
| Project `.venv` | Store Python for Torch training |
| Frozen Phase 3–8 artifacts | Overwriting baselines during Phase 9 |

---

## 6. Acceptance Criteria

These are the bars that were used (or should be used) to call the *software system* complete for research submission prep.

### 6.1 Functional completeness

- [x] All FR-01 … FR-39 implemented and runnable from documented commands.
- [x] Models A–E exist; multitask CD is the primary cross-domain path; finetune retained as negative result.
- [x] Phase reports + README reflect Ninja-first framing.

### 6.2 Validation / “UAT” (researcher acceptance)

In this project, UAT means: *a second reader can reproduce the story from artifacts without trusting chat history.*

- [x] `Documentation/interpret.md` explains how to read each major number.
- [x] Phase 7 integrity 11/11.
- [x] Phase 8 audit 178/178 PASS.
- [x] λ and seed claims backed by `artifacts/phase9/configs/*.yaml` and `metrics/summary_*.json`.
- [ ] Formal examiner walkthrough / viva (process outside the repo).

### 6.3 Performance benchmarks (research KPIs)

| KPI | Target | Delivered |
|-----|--------|-----------|
| Ninja multitask R² (full 2019) | ≫ baseline ~0.5 | ~**0.86** |
| Ninja multitask R² (val holdout) | Report as cleaner estimate | ~**0.81** |
| Multi-seed mean ± std (λ=1.0) | Stable across seeds | **0.8772 ± 0.0140** |
| Calibrated PICP (AR / CD) | ≈ 0.95 | ~0.94–0.96 |
| Phase 8 audit | All checks pass | **178/178** |
| λ=0 ninja R² | Negative control (untrained head) | ~**−0.51** |
| λ peak (ninja) | Identified on grid | ~**0.25** (R²≈0.871) |

### 6.4 Security / integrity testing completed

- [x] Leakage / scaler / split integrity checks (Phase 7).
- [x] Independent metric recomputation (Phase 8).
- [x] λ logging gap on multi-seed CSV identified and closed; YAML/JSON remain the training-day record.
- [ ] Paper PDF site-name / 0.0213 unit ambiguity: **disclosed as unresolved** — acceptance = honest disclosure, not forced numeric match.

### 6.5 Definition of Done (software)

The research software is **Done** when:

1. Architecture is frozen and documented,
2. Primary claim (ninja multitask lift) is measurable from frozen artifacts,
3. Audit passes,
4. Caveats (circular labels, clamps, units, site offset, paper MAE non-reconcilable) are written where a reader will see them,
5. Phase 9 evidence pack exists without destroying Phase 3–8.

**Paper manuscript acceptance is separate** and still in progress (methods/results writing, figure selection).

---

## 7. Appendices

### Appendix A — Glossary (extended)

See §1.4. Additional plain-language notes:

- **Physics prior** — a formula-based power estimate from irradiance/temperature; the network may trust it (high α) or down-weight it.
- **Masked loss** — ninja MSE only on hours that have ninja labels; other hours do not punish the external head.
- **Warm start** — load Phase-5 weights before multitask training.
- **Val-holdout ninja** — evaluation slice that avoids treating early-2019 train-overlap hours as pure test glory.

### Appendix B — Delivered phase inventory

| Phase | Deliverable highlights |
|-------|------------------------|
| 1–2 | Raw + processed data, scaler, physics target |
| 3 | Baseline PCNN checkpoint + Case-1 + raw UQ |
| 4 | Limitations memo |
| 5 | AR-PCNN + conformal UQ + comparison tables |
| 6 | CD multitask / finetune + comparison tables |
| 7 | Ablation A–E, integrity, publication figures |
| 8 | Audit ledger 178/178 |
| 9 | λ-sweep, 5-seed, gate/residual/importance/regimes/noise/runtime |

### Appendix C — Model catalogue (requirements view)

| Model | Requirement role |
|-------|------------------|
| A PCNN | Reproduction baseline |
| B Residual-only | Ablation: residual without learned gate |
| C AR-PCNN | Core adaptive residual + gate |
| D C + conformal | Same weights as C; UQ calibration only |
| E Multitask CD | Cross-domain primary result |

### Appendix D — Key numeric evidence (for SRS → results chapter)

Use these only with units and caveats attached:

- Baseline ninja R² ≈ 0.54 (PCNN); AR-PCNN alone ≈ 0.49.
- Multitask ninja ≈ 0.86 full / 0.81 holdout.
- Multi-seed @ λ=1.0: 0.8772 ± 0.0140 (seeds 42…2026).
- λ grid peak ≈ 0.25; λ=0 → ninja R² ≈ −0.51 (control).
- NSRDB residual models ≈ R² 1.0 + tiny MAE (W) — consistency, not forecasting claim.
- Audit: 178/178 PASS.

### Appendix E — Open issues to carry into the written paper (not software blockers)

1. **Seed paper PDF:** resolve or explicitly narrate Aralvaimozhi / VITCC naming and coordinate text.
2. **0.0213 MAE:** keep “not reconcilable” unless the PDF states unit + capacity clearly.
3. **CSV timestamps** for Exp 02/03/06/08/09 — verify with PowerShell (no bash braces):

```powershell
Get-ChildItem artifacts\phase9\csv\exp0*.csv |
  Where-Object { $_.Name -match '^exp0(2|3|6|8|9)' } |
  Select-Object Name, LastWriteTime |
  Format-Table -AutoSize
```

4. Optional future work: metered (non-physics) PV targets if available.

### Appendix F — References (repeat)

1. [`Solar.pdf`](Solar.pdf) — source paper.  
2. [`../README.md`](../README.md) — system overview.  
3. [`summary.md`](summary.md), [`interpret.md`](interpret.md), phase reports.  
4. NSRDB / Renewables.ninja dataset documentation (cite in final paper).  
5. IEEE / institutional SRS templates — map sections 1–7 of this file into your college format as needed.

### Appendix G — Suggested mapping into a college SRS template

| Their heading | Use from this file |
|---------------|--------------------|
| Introduction | §1 |
| Overall description | §2 |
| Specific requirements | §3 (copy FR/NFR tables) |
| Quality requirements | §4 |
| Constraints | §5 |
| Acceptance criteria | §6 + Appendix D numbers |
| Appendices | §7 |

When you paste into Word/LaTeX, keep the **honest research tone**: this software *enables* a study; it does not *operate* a solar farm.

---

## Document control

| Field | Value |
|-------|-------|
| Title | Software Requirements Specification — Solar-PCNN Research |
| Version | 1.0 |
| Based on delivered work | Phases 1–9 (complete) |
| Primary claim supported | Ninja multitask transfer ~0.86 / holdout ~0.81 |
| Audit status | 178/178 PASS |
| Next human task | Methods/results prose + figure selection for the paper |

*End of SRS.md*
