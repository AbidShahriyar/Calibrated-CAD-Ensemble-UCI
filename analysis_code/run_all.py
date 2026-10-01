"""Run the complete CAD analysis workflow in order."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ANALYSIS_ROOT = Path(__file__).resolve().parent
SCRIPTS = [
    ANALYSIS_ROOT / "scripts" / "01_prepare_data.py",
    ANALYSIS_ROOT / "scripts" / "02_model_analysis.py",
    ANALYSIS_ROOT / "scripts" / "03_generate_figures.py",
    ANALYSIS_ROOT / "scripts" / "04_shap_analysis.py",
    ANALYSIS_ROOT / "scripts" / "05_zero_deletion_sensitivity.py",
    ANALYSIS_ROOT / "scripts" / "06_cohort_transport_sensitivity.py",
    ANALYSIS_ROOT / "scripts" / "07_extended_models_and_ensembles.py",
]


def main() -> None:
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/cad-analysis-matplotlib")
    env.setdefault("LOKY_MAX_CPU_COUNT", "4")
    for script in SCRIPTS:
        print(f"\n=== Running {script.name} ===", flush=True)
        subprocess.run([sys.executable, str(script)], check=True, env=env)


if __name__ == "__main__":
    main()
