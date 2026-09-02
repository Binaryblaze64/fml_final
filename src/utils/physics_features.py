#!/usr/bin/env python3
"""
Physics-Informed Feature Engineering Engine (24-D)
==================================================
Computes 5 governing dimensionless physical parameters from 19 raw descriptors:
1. Steric Sieve Ratio (lambda = r_s / r_p)
2. Ferry-Renkin Factor (Phi)
3. Hydraulic Permeability (L_p = Flux / Pressure)
4. Donnan Electrostatic Index (Psi = Charge * Zeta / pH)
5. Hydrophobic Affinity (H = logD * cos(theta))
"""

import numpy as np
import pandas as pd


def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes 5 dimensionless physics-informed membrane hydrodynamic 
    and electrostatic coupling parameters.
    """
    df_out = df.copy()

    # 1. Physical quantities
    r_solute = df_out['Molecular radius (nm)'].values
    r_pore = np.maximum(df_out['Pore radius (nm)'].values, 1e-6)

    flux = df_out['Pure\u2009water flux (L\u00b7m-2\u00b7h-1)'].values if 'Pure\u2009water flux (L\u00b7m-2\u00b7h-1)' in df_out.columns else df_out.iloc[:, 4].values
    pressure = np.maximum(df_out['Pressure (bar)'].values, 1e-6)

    charge = df_out['Molecular charge'].values
    zeta = df_out['Zeta potential (mV)'].values
    ph = np.maximum(df_out['pH'].values, 1e-6)

    log_d = df_out['log D '].values if 'log D ' in df_out.columns else df_out.iloc[:, 22].values
    contact_angle = df_out['Contact angle (\u00b0)'].values if 'Contact angle (\u00b0)' in df_out.columns else df_out.iloc[:, 19].values

    # 2. Physics Equations
    lambda_steric = r_solute / r_pore
    sieve_term = np.maximum(0.0, 1.0 - lambda_steric)
    phi_ferry = (sieve_term ** 2) * (2.0 - (sieve_term ** 2))
    permeability = flux / pressure
    donnan_electro = (charge * zeta) / ph
    theta_rad = np.radians(contact_angle)
    hydrophobic_affinity = log_d * np.cos(theta_rad)

    # 3. Add to DataFrame
    df_out['Steric_Ratio'] = lambda_steric
    df_out['Ferry_Renkin_Factor'] = phi_ferry
    df_out['Hydraulic_Permeability'] = permeability
    df_out['Donnan_Electro_Index'] = donnan_electro
    df_out['Hydrophobic_Affinity'] = hydrophobic_affinity

    return df_out


def extract_features_and_labels(df: pd.DataFrame, use_physics: bool = True):
    """
    Extracts numerical feature matrix X and target labels y.
    If use_physics is True: returns (19 raw + 5 physics) = 24-D X.
    If use_physics is False: returns 19-D raw X.
    """
    if use_physics:
        df_proc = add_physics_features(df)
        raw_cols = df_proc.iloc[:, 4:23].values
        physics_cols = df_proc[[
            'Steric_Ratio', 'Ferry_Renkin_Factor', 'Hydraulic_Permeability',
            'Donnan_Electro_Index', 'Hydrophobic_Affinity'
        ]].values
        X = np.hstack([raw_cols, physics_cols])
    else:
        X = df.iloc[:, 4:23].values

    y = df.iloc[:, 23].values
    smiles = df.iloc[:, 3].values
    return X, y, smiles
