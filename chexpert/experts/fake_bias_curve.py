import csv
import os

import numpy as np


def _load_expert_curve_from_csv(path: str, seq_len: int) -> list:
    """Read column `p` (or second column) from CSV; pad/truncate to seq_len."""
    values = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No header row in {path}")
        p_key = "p" if "p" in reader.fieldnames else None
        if p_key is None:
            names = [n for n in reader.fieldnames if n is not None]
            if len(names) >= 2:
                p_key = names[1]
            else:
                raise ValueError(f"Expected column 'p' in {path}, got {reader.fieldnames}")
        for row in reader:
            if row.get(p_key) in (None, ""):
                continue
            values.append(float(row[p_key]))
    if not values:
        raise ValueError(f"No numeric rows loaded from {path}")
    if len(values) < seq_len:
        values = values + [values[-1]] * (seq_len - len(values))
    elif len(values) > seq_len:
        values = values[:seq_len]
    return values


def _curve_fixed_len(base: list, seq_len: int) -> list:
    if len(base) >= seq_len:
        return list(base[:seq_len])
    last = base[-1]
    return list(base) + [last] * (seq_len - len(base))


# 21-step non-monotone expert accuracy curve for CheXpert (seq_len=21, step_size=25).
#
# Derived from error counts [1,0,3,3,3,2,2,0,1,5,0,2,4,5,4,4,2,3,2,2,1] (n=25 per step):
#   raw_acc  = 1 - error/25  -> range [0.80, 1.00]
#   scaled   = 0.68 + 1.10 * (raw_acc - 0.80)  -> range [0.68, 0.90]
# Target: classifier AUC_cls ≈ 0.78 is strictly between min (0.68) and max (0.90),
# creating meaningful delegation opportunities throughout the sequence.
expert_acc_curve = [
    0.856, 0.900, 0.768, 0.768, 0.768,   # t=0..4  (errors: 1,0,3,3,3)
    0.812, 0.812, 0.900, 0.856, 0.680,   # t=5..9  (errors: 2,2,0,1,5)
    0.900, 0.812, 0.724, 0.680, 0.724,   # t=10..14 (errors: 0,2,4,5,4)
    0.724, 0.812, 0.768, 0.812, 0.812,   # t=15..19 (errors: 4,2,3,2,2)
    0.856,                                # t=20    (error: 1)
]

class ExpertModelCurveBiased:
    def __init__(self, confounding_class, seq_len, num_classes, curve_csv=None):
        """
        seq_len         : total number of timesteps
        num_classes     : number of prediction classes (K)
        curve_csv       : optional path to CSV with column `p` (per-timestep expert accuracy);
                          if None, uses module default expert_acc_curve.
        """
        self.seq_len = seq_len
        self.K = num_classes
        self.confounding_class = confounding_class
        if curve_csv is not None:
            if not os.path.isfile(curve_csv):
                raise FileNotFoundError(curve_csv)
            self.expert_acc_curve = _load_expert_curve_from_csv(curve_csv, seq_len)
        else:
            self.expert_acc_curve = _curve_fixed_len(expert_acc_curve, seq_len)

    def predict(self, y_batch, timestep):
        """
        y_batch           : numpy array [B, K] binary ground truth labels
        confound_indicator: numpy array [B], 1 if confounded, 0 if not
        timestep          : int, current timestep (starts at 0)
        returns           : numpy array [B, K] binary predictions
        """
        B, K = y_batch.shape
        acc_matrix = np.zeros((B, K))

        t = min(max(int(timestep), 0), len(self.expert_acc_curve) - 1)
        acc_t = self.expert_acc_curve[t]
        for i in range(B):
            for k in range(K):
                acc_matrix[i, k] = max(0.0001, acc_t)

        rand = np.random.rand(B, K)
        flip_mask = rand > acc_matrix
        pred = np.where(flip_mask, 1 - y_batch, y_batch)
        return pred