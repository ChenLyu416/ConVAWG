"""Framework-level dialogue-generation baselines for ConVAWG.

Three published dialogue-synthesis frameworks re-implemented (adapted) on the
same raw inputs as the ConVAWG pipeline:

- ``run_diasynth.py``  — DiaSynth (Suresh et al., Findings of NAACL 2025)
- ``run_syndg.py``     — LLM-adapted SynDG (Chen et al., ACL 2023)
- ``run_spasm.py``     — SPASM (Luo & Laban, Findings of ACL 2026)

Shared raw inputs (and nothing else from ConVAWG): the 200 persona seeds,
the CPS-aligned VAWG taxonomy, and retrieved DHR case evidence. See
``common/`` and ``prepare_shared_inputs.py``.
"""
