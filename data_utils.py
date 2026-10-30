"""Dataset generation, loading and validation helpers.

IMPORTANT: The bundled dataset is SYNTHETIC. It is produced by a physical
simulation of four ceiling LEDs (Lambertian model) and is NOT real
experimental data. Results obtained on it demonstrate that the software works;
they are not a claim about real-world positioning accuracy.
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
DEFAULT_CSV = DATA_DIR / "lifi_synthetic_dataset.csv"

# ---- Simulation settings (used only for the synthetic dataset) ----
ROOM_W, ROOM_L, ROOM_H = 5.0, 5.0, 3.0      # metres
RX_HEIGHT = 0.85                            # receiver (desk/hand height), metres
LAMBERT_ORDER = 1.0                         # Lambertian order m (half-power angle 60 deg)
PEAK_POWER_UW = 100.0                       # power directly beneath an LED, in micro-watts
LED_XY = [(1.25, 1.25), (1.25, 3.75), (3.75, 1.25), (3.75, 3.75)]  # L1..L4
SYN_FEATURES = ["L1", "L2", "L3", "L4"]
SYN_TARGETS = ["X", "Y"]


def simulate_power(x, y, led_xy=LED_XY, h=ROOM_H - RX_HEIGHT, m=LAMBERT_ORDER):
    """Line-of-sight Lambertian received power, normalised so that the value
    directly under an LED equals PEAK_POWER_UW:  P = P0 * (h/d)^(m+3)."""
    cols = []
    for lx, ly in led_xy:
        d = np.sqrt((x - lx) ** 2 + (y - ly) ** 2 + h ** 2)
        cols.append(PEAK_POWER_UW * (h / d) ** (m + 3))
    return np.column_stack(cols)


def generate_synthetic_dataset(n_samples=5000, seed=42, path=DEFAULT_CSV):
    """Create and save the synthetic fingerprint dataset (reproducible)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, ROOM_W, n_samples)
    y = rng.uniform(0, ROOM_L, n_samples)
    clean = simulate_power(x, y)
    noisy = clean * rng.lognormal(0.0, 0.03, clean.shape)      # ~3 % multiplicative noise
    noisy = noisy + rng.normal(0.0, 0.30, clean.shape)         # 0.3 uW additive noise
    noisy = np.clip(noisy, 0.0, None)
    df = pd.DataFrame(np.round(noisy, 4), columns=SYN_FEATURES)
    df["X"] = np.round(x, 4)
    df["Y"] = np.round(y, 4)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def validate_dataframe(df, features, targets):
    """Check structure, coerce to numeric, drop missing/duplicate rows.
    Returns (clean_df, report_dict). Raises ValueError on structural problems."""
    missing_cols = [c for c in list(features) + list(targets) if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Required column(s) missing from dataset: {missing_cols}. "
                         f"Columns found: {list(df.columns)}")
    work = df[list(features) + list(targets)].apply(pd.to_numeric, errors="coerce")
    report = {
        "rows_raw": int(len(df)),
        "missing_values_raw": int(work.isna().sum().sum()),
        "duplicate_rows_raw": int(work.duplicated().sum()),
    }
    work = work.dropna().drop_duplicates().reset_index(drop=True)
    report["rows_clean"] = int(len(work))
    if len(work) < 50:
        raise ValueError(f"Only {len(work)} usable rows after cleaning; need at least 50.")
    if (work[list(features)] < 0).any().any():
        raise ValueError("Signal features contain negative values; expected non-negative signal strength.")
    return work, report
