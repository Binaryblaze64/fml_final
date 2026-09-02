#!/usr/bin/env python3
"""
PhysiChem-GT: Explainable AI & Atom-Level Feature Attribution
============================================================
Generates mechanistic explanations, global 24-D feature importance rankings,
and atom-level gradient attribution heatmaps for trace organic contaminants (TrOCs).

Outputs:
  - results/figure5_shap_feature_importance.png
  - results/figure7_atom_importance_ibuprofen.png
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Batch

# Add project root to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.utils.smiles2graph import create_graph_data_from_smiles
from src.utils.physics_features import extract_features_and_labels
from models.new_architecture import PhysiChemNet

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.2


FEATURE_NAMES_24D = [
    'Pore radius (nm)', 'Pure water flux', 'Pressure (bar)', 'Zeta potential (mV)',
    'pH', 'Contact angle (°)', 'Temperature (°C)', 'Solute conc (μM)', 'Ionic strength (M)',
    'Cross-flow velocity', 'Molecular radius (nm)', 'Molecular weight (g/mol)',
    'Molecular charge', 'log D', 'Polarizability', 'TPSA (Å²)', 'H-bond donors',
    'H-bond acceptors', 'Rotatable bonds',
    'Steric Ratio (λ)', 'Ferry-Renkin (Φ)', 'Permeability (Lp)', 'Donnan Index (Ψ)', 'Hydrophobic Affinity (H)'
]


class MoleculeAttributionAnalyzer:
    def __init__(self, checkpoint_path=None):
        if checkpoint_path is None:
            checkpoint_path = os.path.join(BASE_DIR, "checkpoint", "best_PhysiChemNet.pth")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
        self.config = ckpt['config']
        self.model = PhysiChemNet(self.config)
        self.model.load_state_dict(ckpt['model_state'])
        self.model.eval()

        data_path = os.path.join(BASE_DIR, "data", "processed", "MemTrOC-Dataset.csv")
        df_ref = pd.read_csv(data_path)
        self.X_ref, self.y_ref, self.smiles_ref = extract_features_and_labels(df_ref, use_physics=True)
        self.scaler = MinMaxScaler()
        self.scaler.fit(self.X_ref)

    def compute_atom_importance(self, smiles: str, table_24d: np.ndarray, method='integrated_gradients'):
        """Computes gradient-based attribution scores for every atom in the molecule."""
        graph_data = create_graph_data_from_smiles(smiles, 0.0)
        if graph_data is None:
            return None, None

        x_scaled = self.scaler.transform(table_24d.reshape(1, -1))
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

        node_feat = graph_data.x.float().clone()
        node_feat.requires_grad_(True)
        graph_data.x = node_feat

        if method == 'integrated_gradients':
            steps = 30
            baseline = torch.zeros_like(node_feat)
            accumulated_grads = torch.zeros_like(node_feat)

            for step in range(1, steps + 1):
                alpha = step / steps
                interpolated = (baseline + alpha * (node_feat - baseline)).clone().detach().requires_grad_(True)

                g = graph_data.clone()
                g.x = interpolated
                gb = Batch.from_data_list([g])

                out = self.model(x_tensor, gb)
                grad = torch.autograd.grad(outputs=out.sum(), inputs=interpolated)[0]
                accumulated_grads += grad.detach()

            avg_grad = accumulated_grads / steps
            attributions = (node_feat - baseline) * avg_grad
            atom_scores = attributions.norm(dim=-1).detach().numpy()
        else:
            gb = Batch.from_data_list([graph_data])
            out = self.model(x_tensor, gb)
            out.backward()
            atom_scores = node_feat.grad.norm(dim=-1).detach().numpy()

        if atom_scores.max() > atom_scores.min():
            atom_scores = (atom_scores - atom_scores.min()) / (atom_scores.max() - atom_scores.min())

        mol = Chem.MolFromSmiles(smiles)
        return atom_scores, mol

    def compute_global_feature_importance(self, n_samples=300):
        """Computes global feature importance rankings across all 24 tabular features."""
        indices = np.random.RandomState(42).choice(len(self.X_ref), min(n_samples, len(self.X_ref)), replace=False)
        X_sub = self.X_ref[indices]
        smiles_sub = self.smiles_ref[indices]

        X_scaled = self.scaler.transform(X_sub)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32, requires_grad=True)

        graphs = [create_graph_data_from_smiles(s, 0.0) for s in smiles_sub]
        graph_batch = Batch.from_data_list([g for g in graphs if g is not None])

        out = self.model(X_tensor, graph_batch)
        grad = torch.autograd.grad(outputs=out.sum(), inputs=X_tensor)[0]

        importances = (X_tensor * grad).abs().mean(dim=0).detach().numpy()
        importances = (importances / importances.sum()) * 100.0

        feat_df = pd.DataFrame({
            'Feature': FEATURE_NAMES_24D,
            'Importance': importances
        }).sort_values('Importance', ascending=True)

        return feat_df


def plot_global_feature_importance(feat_df, save_path):
    """Generates Figure 5: Global Feature Importance Bar Chart."""
    plt.figure(figsize=(10, 7), dpi=300)

    physics_features = [
        'Steric Ratio (λ)', 'Ferry-Renkin (Φ)', 'Permeability (Lp)',
        'Donnan Index (Ψ)', 'Hydrophobic Affinity (H)'
    ]

    colors = ['#1f77b4' if f in physics_features else '#aec7e8' for f in feat_df['Feature']]

    bars = plt.barh(feat_df['Feature'], feat_df['Importance'], color=colors, edgecolor='#333333', linewidth=0.8)

    plt.xlabel('Relative Feature Importance Attribution (%)', fontsize=12, fontweight='bold')
    plt.title('PhysiChem-GT: Global 24-D Physics Feature Importance\n(Blue = Engineered Physical Laws, Light Blue = Raw Descriptors)',
              fontsize=12, fontweight='bold', pad=15)
    plt.grid(axis='x', linestyle='--', alpha=0.5)

    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.15, bar.get_y() + bar.get_height() / 2, f'{width:.1f}%',
                 va='center', ha='left', fontsize=9, color='#222222')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  [+] Saved Figure 5 to: {save_path}")


def plot_atom_importance(smiles, atom_scores, mol, chemical_name, save_path):
    """Generates Figure 7: Atom-level substructure attribution map."""
    if mol is None or atom_scores is None:
        return

    norm = Normalize(vmin=0.0, vmax=1.0)
    cmap = cm.get_cmap('Reds') if hasattr(cm, 'get_cmap') else plt.colormaps['Reds']

    atom_colors = {}
    highlight_radii = {}
    for i, score in enumerate(atom_scores):
        if i < mol.GetNumAtoms():
            score_val = float(score)
            rgba = cmap(norm(score_val))
            atom_colors[i] = (float(rgba[0]), float(rgba[1]), float(rgba[2]))
            highlight_radii[i] = float(0.35 + (0.25 * score_val))

    drawer = rdMolDraw2D.MolDraw2DCairo(600, 450)
    opts = drawer.drawOptions()
    opts.addAtomIndices = False
    opts.bondLineWidth = 2.5
    opts.padding = 0.15

    drawer.DrawMolecule(
        mol,
        highlightAtoms=list(atom_colors.keys()),
        highlightAtomColors=atom_colors,
        highlightAtomRadii=highlight_radii,
        highlightBonds=[]
    )
    drawer.FinishDrawing()

    with open(save_path, 'wb') as f:
        f.write(drawer.GetDrawingText())
    print(f"  [+] Saved Figure 7 to: {save_path}")


def main():
    print("=" * 70)
    print("  PHYSI-CHEM-GT: EXPLAINABLE AI & ATOM ATTRIBUTION ENGINE")
    print("=" * 70)

    analyzer = MoleculeAttributionAnalyzer()

    # 1. Global Feature Importance Ranking (Figure 5)
    print("\n[+] Computing Global 24-D Feature Attribution Matrix...")
    feat_df = analyzer.compute_global_feature_importance(n_samples=300)
    fig5_path = os.path.join(BASE_DIR, "results", "figure5_shap_feature_importance.png")
    plot_global_feature_importance(feat_df, fig5_path)

    # 2. Atom Attribution Map for Ibuprofen (Figure 7)
    print("\n[+] Computing Atom-Level Attribution for Ibuprofen...")
    ibuprofen_smiles = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
    default_table_24d = analyzer.X_ref[0]
    atom_scores, mol = analyzer.compute_atom_importance(ibuprofen_smiles, default_table_24d)

    fig7_path = os.path.join(BASE_DIR, "results", "figure7_atom_importance_ibuprofen.png")
    plot_atom_importance(ibuprofen_smiles, atom_scores, mol, "Ibuprofen", fig7_path)

    print("\n" + "=" * 70)
    print("  Explainability Figures Successfully Generated!")
    print("=" * 70)


if __name__ == '__main__':
    main()