import pandas as pd
import numpy as np

def extract_tables():
    # Load processed dataset
    df = pd.read_csv('data/processed/MemTrOC-Dataset.csv')
    
    # 1. Unique membranes and their properties
    mem_cols = ['Name of Membranes', 'Type of Membranes', 'MWCO (Da)', 'Pore radius (nm)', 'Contact angle (°)', 'Zeta potential (mV)']
    mem_df = df.groupby('Name of Membranes').agg({
        'Type of Membranes': 'first',
        'MWCO (Da)': 'mean',
        'Pore radius (nm)': 'mean',
        'Contact angle (°)': 'mean',
        'Zeta potential (mV)': ['min', 'max']
    }).reset_index()
    
    print(f"Total unique membranes in dataset: {len(mem_df)}")
    
    # 2. Extract 169 unique TrOCs
    troc_cols = ['NAME of TrOCs', 'SMILEs', 'MW (Da)', 'log Kow', 'Molecular charge', 'log D ']
    # Group by SMILES to get unique chemical entities
    trocs = df.groupby('SMILEs').agg({
        'NAME of TrOCs': 'first',
        'MW (Da)': 'first',
        'log Kow': 'first',
        'Molecular charge': 'first',
        'log D ': 'first'
    }).reset_index()
    
    print(f"Total unique TrOC SMILES in dataset: {len(trocs)}")
    
    # Sort alphabetically by compound name
    trocs = trocs.sort_values(by='NAME of TrOCs', key=lambda col: col.str.lower()).reset_index(drop=True)
    
    # Save extracted compound data to CSV for easy inspection and TeX generation
    trocs.to_csv('results/paper_figures/si_troc_catalog.csv', index=False)
    print("Saved si_troc_catalog.csv")
    
    return mem_df, trocs

if __name__ == '__main__':
    extract_tables()
