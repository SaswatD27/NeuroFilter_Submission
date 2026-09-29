# README

This repository provides code to reproduce the experiments in the submission "NeuroFilter: Activation-Based Guardrails for Privacy-Conscious LLM Agents".

In particular, each folder provides code to run the eponymous experiment.

- Singleturn_Probing runs the NeuroFilter singleturn probing results over CMPL and PrivacyLens benchmarks and related ablations/variations thereof.
- Multiturn_Probing runs the NeuroFilter activation velocity probing results over CMPL and FracturedSORRY benchmarks.
- CMPL_Insurance_vs_Scheduling_Cosine_Similarities runs the experiment to calculate cosine similarities of concept directions over layers for CMPL benchmarks.
- Superposition_Test runs the modular construction test.
- Baselines contains the baseline scripts.
- Data contains the Fractured SORRYBench data for multi-turn probing and generated AutoDAN data for CMPL Single-Turn Probing

Each python script has a self-explanatory name and is self-contained.

The required Python packages are listed in neurofilter_requirements.txt.