#!/usr/bin/env python3
"""
Single Molecule Analysis Example (PhysiChem-GT)
==============================================
Runs atomic attribution analysis on custom chemical SMILES.
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import numpy as np

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.molecule_feature_importance import MoleculeAttributionAnalyzer


def main():
    print("=" * 70)
    print("  PHYSI-CHEM-GT: SINGLE MOLECULE ATOM ATTRIBUTION EXAMPLE")
    print("=" * 70)
    
    analyzer = MoleculeAttributionAnalyzer()
    
    example_compounds = {
        "Acetaminophen (Paracetamol)": "CC(=O)Nc1ccc(O)cc1",
        "Caffeine": "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
        "Ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
    }

    default_table_24d = analyzer.X_ref[0]

    for name, smiles in example_compounds.items():
        print(f"\nAnalyzing: {name}")
        print(f"SMILES:    {smiles}")
        atom_scores, mol = analyzer.compute_atom_importance(smiles, default_table_24d)
        print(f"Atom Count: {len(atom_scores)}")
        print(f"Top 3 Most Influential Atom Indices: {np.argsort(atom_scores)[::-1][:3].tolist()}")
        print(f"Attribution Range: [{atom_scores.min():.4f}, {atom_scores.max():.4f}]")

    print("\n" + "=" * 70)
    print("  Analysis Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()