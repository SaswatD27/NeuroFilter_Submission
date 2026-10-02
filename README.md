# NeuroFilter Code Package

This repository contains the scripts used to run the experiments in
"NeuroFilter: Activation-Based Guardrails for Privacy-Conscious LLM Agents."

- `Singleturn_Probing` contains the CMPL and PrivacyLens single-turn experiments and their ablations.
- `Multiturn_Probing` contains the CMPL, Fractured SORRY-Bench, repeated-split, additional-attack, and delta-versus-displacement experiments.
- `CMPL_Insurance_vs_Scheduling_Cosine_Similarities` contains the concept-direction similarity experiment.
- `Superposition_Test` contains the modularity experiments.
- `Harmfulness_Context_Dependence` contains the context-dependence experiments.
- `Baselines` contains the baseline, latency, and memory-measurement scripts.
- `Data` contains the input datasets distributed with this package.

Placeholder pathnames such as `/path/to/code` must be set to your local specific paths before running. Required Python packages are listed in `neurofilter_requirements.txt`.
