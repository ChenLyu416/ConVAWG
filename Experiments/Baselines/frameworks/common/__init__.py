"""Shared infrastructure for the framework-level baselines.

Centralises the sys.path setup needed to import sibling repo modules
(`baseline_generate`, `llm_backend`, `retrieval`, `prompt_templates`) so the
framework scripts and common modules can import them directly.
"""

import sys
from pathlib import Path

FRAMEWORKS_DIR = Path(__file__).resolve().parents[1]
BASELINES_DIR = FRAMEWORKS_DIR.parent
REPO_ROOT = BASELINES_DIR.parents[1]
PERSONA_GENERATE_DIR = REPO_ROOT / "persona_generate"
LLM_JUDGE_DIR = REPO_ROOT / "LLM_judge"

for _p in (BASELINES_DIR, PERSONA_GENERATE_DIR, LLM_JUDGE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
