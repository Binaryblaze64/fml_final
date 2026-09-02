#!/usr/bin/env python3
"""
PhysiChem-GTX Live Test Evaluation Script
==========================================
Runs live inference using both PhysiChem-GT (Graph Transformer) and PhysiChem-XGB
(Monotonic Physics Booster) on the held-out test split, reporting actual metrics.
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import json
import joblib
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from dataset.dataset import TableGraphDataset
from src.utils.physics_features import extract_features_and_labels
from models.new_architecture import PhysiChemNet
from src.evolution_search import precache_all_graphs, get_cached_graph, collate_fn


def get_fingerprint(smiles: str, n_bits: int = 256) -> np.ndarray:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    return np.array(fp, dtype=np.float32)


def main():
    print("=" * 80)
    print("  LIVE INFERENCE EVALUATION — PhysiChem-GTX")
    print("  (Dual-Stream Graph Transformer + Monotonic Physics Booster)")
    print("=" * 80)

    # 1. Load Dataset
    data_file = os.path.join(BASE_DIR, "data", "processed", "MemTrOC-Dataset.csv")
    df = pd.read_csv(data_file)
    X_phys, y, smiles = extract_features_and_labels(df, use_physics=True)

    r_solute = df['Molecular radius (nm)'].values
    r_pore = np.maximum(df['Pore radius (nm)'].values, 1e-6)
    steric_all = r_solute / r_pore
    fps = np.array([get_fingerprint(s, 256) for s in smiles])
    X_xgb_all = np.hstack([X_phys, fps])

    # 2. Hold-out Test Split (10%, random_state=41)
    X_dev_p, X_test_p, y_dev, y_test, sm_dev, sm_test, st_dev, st_test, X_xgb_dev, X_xgb_test = \
        train_test_split(X_phys, y, smiles, steric_all, X_xgb_all, test_size=0.1, random_state=41)

    print(f"\n[+] Test Set Size: {len(y_test)} samples (10% held-out)")

    # 3. Stream 1: PhysiChem-GT Inference
    gt_ckpt_path = os.path.join(BASE_DIR, "checkpoint", "best_PhysiChemNet.pth")
    gt_ckpt = torch.load(gt_ckpt_path, map_location='cpu', weights_only=False)
    gt_model = PhysiChemNet(gt_ckpt['config'])
    gt_model.load_state_dict(gt_ckpt['model_state'])
    gt_model.eval()

    scaler = joblib.load(os.path.join(BASE_DIR, "checkpoint", "scaler.pkl"))
    X_test_scaled = scaler.transform(X_test_p)

    precache_all_graphs(smiles)
    test_ds = TableGraphDataset(
        torch.tensor(X_test_scaled, dtype=torch.float32),
        sm_test,
        torch.tensor(y_test, dtype=torch.float32).view(-1, 1),
        get_cached_graph,
        steric_ratios=st_test
    )
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)
    gt_preds = []
    with torch.no_grad():
        for b in test_loader:
            gt_preds.extend(gt_model(b[0], b[1]).numpy())
    gt_preds = np.array(gt_preds)

    # 4. Stream 2: PhysiChem-XGB Inference
    xgb_ckpt_path = os.path.join(BASE_DIR, "checkpoint", "best_xgboost_model.pkl")
    xgb_model = joblib.load(xgb_ckpt_path)
    xgb_preds = xgb_model.predict(X_xgb_test)

    # 5. Dual-Stream Adaptive Physics-Gated MoE Fusion
    # g(lambda) dynamically routes based on steric ratio
    g_moe = 0.10 / (1.0 + np.exp(6.0 * (st_test - 0.95)))
    gtx_preds = g_moe * gt_preds + (1.0 - g_moe) * xgb_preds
    gtx_preds = np.clip(gtx_preds, 0.0, 100.0)

    test_r2 = float(r2_score(y_test, gtx_preds))
    test_rmse = float(np.sqrt(mean_squared_error(y_test, gtx_preds)))
    test_mae = float(mean_absolute_error(y_test, gtx_preds))

    # 6. Report
    print("\n" + "=" * 80)
    print("  GENUINE LIVE HOLDOUT PERFORMANCE (Physics-Gated MoE)")
    print("=" * 80)
    print(f"  Test R²:   {test_r2:.4f}")
    print(f"  Test RMSE: {test_rmse:.2f}%")
    print(f"  Test MAE:  {test_mae:.2f}%")
    print("\n  Literature Baseline (Xiao et al., 2026): R² = 0.9014 | RMSE = 9.11% | MAE = 6.17%")
    if test_r2 > 0.9014:
        print(f"  [+] BEATS BASE PAPER ON R² BY +{(test_r2 - 0.9014)*100:.2f} points!")
    if test_mae < 6.17:
        print(f"  [+] BEATS BASE PAPER ON MAE BY -{6.17 - test_mae:.2f}%!")
    print("=" * 80)

    print("\n  [VERIFICATION] FIRST 6 LIVE TEST PREDICTIONS VS GROUND TRUTH:")
    print(f"  {'#':<4} | {'Actual Rejection (%)':<22} | {'Predicted Rejection (%)':<24} | {'Absolute Error (%)':<18}")
    print("  " + "-" * 74)
    for i in range(min(6, len(y_test))):
        print(f"  {i+1:<4} | {y_test[i]:<22.2f} | {gtx_preds[i]:<24.2f} | {abs(y_test[i] - gtx_preds[i]):<18.2f}")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
