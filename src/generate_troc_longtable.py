import pandas as pd

def format_smiles_breakable(smiles):
    smiles = str(smiles).replace('\xa0', '').strip()
    tokens = []
    for c in smiles:
        if c == '\\':
            tokens.append(r'\textbackslash{}')
        elif c == '#':
            tokens.append(r'\#')
        elif c == '%':
            tokens.append(r'\%')
        elif c == '&':
            tokens.append(r'\&')
        elif c == '_':
            tokens.append(r'\_')
        elif c in r'{}~^$':
            tokens.append('\\' + c)
        else:
            tokens.append(c)
    # Join tokens with \allowbreak so TeX breaks cleanly at any character within the cell
    return r'\allowbreak '.join(tokens)

def clean_compound_name(name):
    name = str(name).replace('\xa0', ' ').strip()
    # Map Greek symbols to standard LaTeX math
    name = name.replace('\u03b1', r'$\alpha$').replace('\u03b2', r'$\beta$')
    name = name.replace('&', r'\&').replace('%', r'\%').replace('_', ' ')
    return name

def generate_table_s3():
    trocs = pd.read_csv('results/paper_figures/si_troc_catalog.csv')
    
    # Sort alphabetically by name
    trocs = trocs.sort_values(by='NAME of TrOCs', key=lambda col: col.str.lower()).reset_index(drop=True)
    
    lines = []
    lines.append(r"\begin{footnotesize}")
    lines.append(r"\setlength{\tabcolsep}{3.5pt}")
    lines.append(r"\begin{longtable}{>{\raggedright\arraybackslash}p{3.5cm} >{\raggedright\arraybackslash\ttfamily}p{5.8cm} c c c c}")
    lines.append(r"\caption{\textbf{Comprehensive physicochemical library of all 169 micropollutants} compiled in the MemTrOC benchmark corpus. Compounds are listed alphabetically with canonical SMILES representations, molecular weight (MW), octanol-water partition coefficient ($\log K_{ow}$), operational pH distribution coefficient ($\log D$), and formal net charge ($z_s$).}\label{tab:s3_trocs} \\")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Compound Name} & \multicolumn{1}{l}{\textbf{Canonical SMILES}} & \begin{tabular}{@{}c@{}}\textbf{MW}\\\textbf{(Da)}\end{tabular} & $\mathbf{\log K_{ow}}$ & $\mathbf{\log D}$ & $\mathbf{z_s}$ \\")
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\multicolumn{6}{c}{{\bfseries Table \thetable\ continued from previous page}} \\")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Compound Name} & \multicolumn{1}{l}{\textbf{Canonical SMILES}} & \begin{tabular}{@{}c@{}}\textbf{MW}\\\textbf{(Da)}\end{tabular} & $\mathbf{\log K_{ow}}$ & $\mathbf{\log D}$ & $\mathbf{z_s}$ \\")
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{6}{r}{{\textit{Continued on next page...}}} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\endfoot")
    lines.append(r"\bottomrule")
    lines.append(r"\endlastfoot")
    
    for idx, row in trocs.iterrows():
        name = clean_compound_name(row['NAME of TrOCs'])
        smiles_fmt = format_smiles_breakable(row['SMILEs'])
        mw = f"{float(row['MW (Da)']):.1f}" if pd.notnull(row['MW (Da)']) else "-"
        logkow = f"{float(row['log Kow']):.2f}" if pd.notnull(row['log Kow']) else "-"
        logd = f"{float(row['log D ']):.2f}" if pd.notnull(row['log D ']) else "-"
        charge = f"{int(round(float(row['Molecular charge'])))}" if pd.notnull(row['Molecular charge']) else "0"
        
        lines.append(f"{name} & {smiles_fmt} & {mw} & {logkow} & {logd} & {charge} \\\\")
        
    lines.append(r"\end{longtable}")
    lines.append(r"\end{footnotesize}")
    
    out_tex = "\n".join(lines)
    with open('results/paper_figures/table_s3_troc_catalog.tex', 'w', encoding='utf-8') as f:
        f.write(out_tex)
    print(f"Generated table_s3_troc_catalog.tex with {len(trocs)} compounds successfully!")

if __name__ == '__main__':
    generate_table_s3()
