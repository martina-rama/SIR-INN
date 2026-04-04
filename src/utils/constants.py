# Make repository importable when running as script or from notebook
import sys
from pathlib import Path
# Add repository root to Python path
repo_root = Path().resolve().parent
sys.path.append(str(repo_root))

import numpy as np

# ------------------------------------------------------------------
## Epidemiological time indexing

# Define the time grids
domain = [0, 600]
times =  np.linspace(int(domain[0]), int(domain[1]), int(domain[1]) - int(domain[0]) + 1) # Continuous time grid for evaluation

# Epidemiological weeks of an influenza season (weeks 42–52, 01–17)
epi_weeks = np.array(list(range(42, 53)) + list(range(1, 18)))
t0_range = range(3, len(epi_weeks))

# Continuous ordered time index (used by the model)
t_epi = np.arange(1, len(epi_weeks) + 1)

# String labels for plotting
epi_weeks_str = np.array([f"{w:02d}" for w in epi_weeks])
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# Define the quantiles of interest for the probabilistic forecasts

quantiles = [0.01, 0.02, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25,
             0.30, 0.35, 0.40, 0.50, 0.60, 0.65, 0.70, 0.75,
             0.80, 0.85, 0.90, 0.95, 0.975, 0.99]
rename_map = {
    0.01: 'q01',
    0.02: 'q02',
    0.025: 'q025',
    0.05: 'q05',
    0.10: 'q10',
    0.15: 'q15',
    0.20: 'q20',
    0.25: 'q25',
    0.30: 'q30',
    0.35: 'q35',
    0.40: 'q40',
    0.50: 'q50',
    0.60: 'q60',
    0.65: 'q65',
    0.70: 'q70',
    0.75: 'q75',
    0.80: 'q80',
    0.85: 'q85',
    0.90: 'q90',
    0.95: 'q95',
    0.975: 'q975',
    0.99: 'q99'
}
# ------------------------------------------------------------------ 