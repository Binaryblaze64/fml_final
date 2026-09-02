#!/usr/bin/env python3
"""
Final Evaluation for PhysiChem-GT (Fixed — Live Metrics)
==========================================================
Evaluates the winning PhysiChem-GT architecture with LIVE inference:
  1. Loads the saved checkpoint and scaler (no hardcoded numbers)
  2. Reconstructs the exact same train/test split used in training
  3. Runs forward pass on the held-out test set
  4. Reports genuinely computed R², RMSE, MAE
"""

import sys
import os
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import json
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader

# Add project root to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from dataset.dataset import TableGraphDataset
from src.utils.physics_features import extract_features_and_labels
from models.new_architecture import PhysiChemNet
from src.evolution_search import precache_all_graphs, get_cached_graph, collate_fn


def main():
    print("=" * 80)
    print("  FINAL LIVE EVALUATION — PhysiChem-GT")
    print("  (All metrics computed by actual model inference, not hardcoded)")
    print("=" * 80)

    # 1. Load full dataset + steric ratios
    data_file = os.path.join(BASE_DIR, "data", "processed", "MemTrOC-Dataset.csv")
    df = pd.read_csv(data_file)
    X_all, y_all, smiles_all = extract_features_and_labels(df, use_physics=True)

    r_solute   = df['Molecular radius (nm)'].values
    r_pore     = np.maximum(df['Pore radius (nm)'].values, 1e-6)
    steric_all = r_solute / r_pore

    print(f"\n[+] Total Samples: {len(X_all)}, Features: {X_all.shape[1]}-D")

    # 2. Recreate the exact same held-out test split used in training
    #    (random_state=41 matches evolution_search.py)
    X_dev, X_test, y_dev, y_test, smiles_dev, smiles_test, steric_dev, steric_test = \
        train_test_split(X_all, y_all, smiles_all, steric_all,
                         test_size=0.1, random_state=41)
    print(f"[+] Test split: {len(X_test)} samples (10% holdout, seed=41)")

    # 3. Load checkpoint
    ckpt_path = os.path.join(BASE_DIR, "checkpoint", "best_PhysiChemNet.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt   = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    config = ckpt['config']
    print(f"[+] Checkpoint loaded: {ckpt_path}")

    # 4. Load the saved scaler (Fix 3 — never refit at inference)
    scaler_path = os.path.join(BASE_DIR, "checkpoint", "scaler.pkl")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print("[+] Loaded saved scaler from checkpoint/scaler.pkl")
    else:
        import warnings
        warnings.warn(
            "scaler.pkl not found — refitting on dev set. "
            "Re-run evolution_search.py to generate a saved scaler."
        )
        scaler = MinMaxScaler()
        scaler.fit(X_dev)

    X_test_scaled = scaler.transform(X_test)

    # 5. Build model and load weights
    model = PhysiChemNet(config)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"[+] Model loaded: {model.count_parameters():,} parameters")

    # 6. Pre-cache all graphs and build test DataLoader
    precache_all_graphs(smiles_all)
    test_ds = TableGraphDataset(
        torch.tensor(X_test_scaled, dtype=torch.float32),
        smiles_test,
        torch.tensor(y_test, dtype=torch.float32).view(-1, 1),
        get_cached_graph,
        steric_ratios=steric_test
    )
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

    # 7. Live inference — Fix 5: actually run the model (no hardcoded values)
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            x, graph_data, y = batch[0], batch[1], batch[2]
            pred = model(x, graph_data)
            all_preds.extend(pred.numpy())
            all_targets.extend(y.squeeze().numpy())

    all_preds   = np.array(all_preds)
    all_targets = np.array(all_targets)

    live_r2   = float(r2_score(all_targets, all_preds))
    live_rmse = float(np.sqrt(mean_squared_error(all_targets, all_preds)))
    live_mae  = float(mean_absolute_error(all_targets, all_preds))

    # 8. Report
    print("\n" + "=" * 80)
    print("  LIVE HOLDOUT PERFORMANCE (actually computed)")
    print("=" * 80)
    print(f"  Test R²:   {live_r2:.4f}")
    print(f"  Test RMSE: {live_rmse:.4f}%")
    print(f"  Test MAE:  {live_mae:.4f}%")
    print("\n  Base Paper (Xiao et al.): R² = 0.9014 | RMSE = 9.1118% | MAE = 6.1691%")
    if live_r2 > 0.9014:
        print(f"  [+] BEATS PAPER on R2 by {(live_r2 - 0.9014)*100:.2f} points!")
    else:
        print(f"  [-] Below paper R2 by {(0.9014 - live_r2)*100:.2f} points")
    if live_mae < 6.1691:
        print(f"  [+] BEATS PAPER on MAE by {6.1691 - live_mae:.4f}%")
    print("=" * 80)

    # 9. Save to Benchmark JSON
    benchmark_data = {
        "model_name": "PhysiChem-GT (Physics-Informed Chemical Graph Transformer)",
        "note": "All metrics live-computed by actual model inference on held-out test set.",
        "single_split": {
            "test_r2":   live_r2,
            "test_rmse": live_rmse,
            "test_mae":  live_mae
        },
        "base_paper_comparison": {
            "paper_r2":       0.9014,
            "paper_rmse":     9.1118,
            "paper_mae":      6.1691,
            "r2_gain":        live_r2  - 0.9014,
            "rmse_reduction": 9.1118  - live_rmse,
            "mae_reduction":  6.1691  - live_mae
        }
    }

    results_json = os.path.join(BASE_DIR, "results", "final_physichemnet_benchmark.json")
    with open(results_json, 'w') as f:
        json.dump(benchmark_data, f, indent=2)
    print(f"\n[+] Benchmark JSON saved to: {results_json}")


if __name__ == '__main__':
    main()
