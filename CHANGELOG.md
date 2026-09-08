# Changelog

## 1.2.0 - 2026-09-08

- Add protocol-wide model comparison charts to the static results explorer.
- Add an interactive read/write/privileged enforcement-policy simulator.
- Display experiment, split, configuration, and policy provenance directly in the demo.
- Keep the demo fully static with no cold start, submitted prompts, model key, or backend.

## 1.1.1 - 2026-09-03

- Emit strict JSON from the static benchmark report, representing unavailable measurements as `null` rather than non-standard `NaN`.
- Add checksums for committed source metrics and validate protocol coverage, uniqueness, and checksums before Pages deployment.
- Show an explicit explorer error instead of a blank page when report loading fails.

## 1.1.0 - 2026-09-03

- Added a deterministic, risk-aware enforcement policy with explicit human-review band.
- Added fail-closed handling and reproducible policy configuration hashes.
- Exposed policy evaluation separately from detector inference to demonstrate a clean enforcement boundary.

All notable changes to ToolShield are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.0.0] — 2026-09-03

### Changed
- Fixed transformer seed propagation and added regression coverage.
- Added an explicit disclosure for historical neural results affected by the seed defect.
- Added a machine-readable, provenance-bearing results report and static explorer.
- Updated the dependency lock to the first patched Torch 2.13 release line.
- Added CI deployment for the GitHub Pages results explorer.

## [0.9.0] — 2026-05-01

### Added
- Synthetic dataset generator for prompt injection in tool-using LLM agents
- Four attack families: AF1 (Instruction Override), AF2 (Data Exfiltration), AF3 (Tool Hijacking), AF4 (Indirect Injection)
- Three evaluation protocols: `S_random`, `S_attack_holdout`, `S_tool_holdout`
- Baseline models: rule-based heuristics, TF-IDF + LR, transformer classifier
- Operational metrics: FPR@TPR(0.90), FPR@TPR(0.95), ASR Reduction
- Long-schema stress test: naive truncation vs. prompt-preserving truncation ablation
- Key result: `keep_prompt` strategy achieves 0.998 ROC-AUC vs. 0.455 for naive truncation on `S_attack_holdout`
- Split hygiene verification: no template leakage across splits
- CLI (`toolshield generate`, `split`, `train`, `eval`)
- Full thesis PDF and supplementary artifacts in `thesis/`
- Reproducible seeds embedded in config files and output manifests
