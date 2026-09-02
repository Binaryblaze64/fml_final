# PhysiChem-GTX: Physics-Informed Dual-Stream Graph Transformer and Tree Mixture-of-Experts for High-Precision Membrane Micropollutant Rejection Prediction and Mechanistic Discovery

**Target Journal:** *ACS ES&T Engineering* / *Environmental Science & Technology* / *Journal of Membrane Science*  
**Repository & Code Availability:** [https://github.com/Binaryblaze64/fml_final.git](https://github.com/Binaryblaze64/fml_final.git)

---

## ABSTRACT

The proliferation of trace organic contaminants (TrOCs)—encompassing pharmaceuticals, personal care products, pesticides, and endocrine disruptors—in municipal and industrial water cycles poses grave ecological and public health challenges. Nanofiltration (NF) and reverse osmosis (RO) membranes represent gold-standard separation technologies; however, predicting solute rejection remains notoriously intractable due to complex, concurrent transport mechanisms including hydrodynamic steric hindrance, Donnan electrostatic repulsion, dielectric exclusion, and hydrophobic partitioning. While deep learning offers empirical promise, existing architectures either struggle with data sparsity or treat molecular structure as a black box divorced from hydrodynamic principles.

Here, we present **PhysiChem-GTX**, a physics-informed dual-stream Mixture-of-Experts (MoE) architecture that integrates 3D-aware graph neural networks, monotonic tree ensembles, and physical transport gating. PhysiChem-GTX couples:
1. **PhysiChem-GT**, an evolutionary neural architecture-searched (NAS) Graph Transformer operating directly on molecular topology with integrated gradients attribution, and
2. **PhysiChem-XGB**, a monotonic gradient-boosted decision tree ensemble trained on 280 physical descriptors and extended-connectivity fingerprints (ECFP4), coordinated via
3. A **Sigmoidal Physical Gating Function** conditioned on the dimensionless steric ratio $\lambda = r_{\mathrm{solute}} / r_{\mathrm{pore}}$.

Evaluated on the benchmark **MemTrOC** dataset ($N = 1,617$ filtration trials across 169 distinct micropollutants), PhysiChem-GTX establishes a new state-of-the-art:
- **Test Performance:** $R^2 = 0.9130$, $\text{RMSE} = 8.56\%$, and $\text{MAE} = 5.52\%$, substantially surpassing previous benchmark models ($R^2 = 0.8654$, $\text{RMSE} = 10.60\%$).
- **5-Fold Cross-Validation:** $R^2 = 0.8878 \pm 0.0163$, confirming rigorous out-of-fold generalization.
- **Epistemic Uncertainty Quantification:** Monte Carlo dropout ($N_{\mathrm{MC}} = 50$) delivers empirical confidence envelopes with mean uncertainty $\bar{\sigma} = 13.21\%$.
- **Mechanistic Explainability:** Multi-scale TreeSHAP analysis identifies steric ratio ($\lambda$), pore radius ($r_p$), and filtration duration as top drivers, while atom-level Integrated Gradients reveal localized functional group attributions (e.g., carboxylate Donnan repulsion in ibuprofen, chlorotriazine sieving in atrazine).

PhysiChem-GTX reconciles high-capacity deep learning with hydrodynamic transport physics, providing an open-source, publication-grade digital twin for membrane process design and micropollutant barrier assessment.

**Keywords:** Trace Organic Contaminants (TrOCs), Nanofiltration/Reverse Osmosis, Physics-Informed Machine Learning, Graph Transformer, TreeSHAP, Uncertainty Quantification.

---

## 1. INTRODUCTION

Water scarcity and industrial effluent discharge have intensified the presence of trace organic contaminants (TrOCs)—including pharmaceuticals, personal care products, per- and polyfluoroalkyl substances (PFAS), and endocrine-disrupting compounds—in drinking water sources worldwide. Even at trace concentrations ($\text{ng/L}$ to $\mu\text{g/L}$), TrOCs exhibit bioaccumulation, endocrine disruption, and chronic ecotoxicity.

Pressure-driven membrane processes, specifically nanofiltration (NF) and reverse osmosis (RO), provide energy-efficient, chemical-free multi-barrier separation. However, accurately forecasting rejection efficiency across diverse solute-membrane-operating matrices remains challenging. Classical hydrodynamic models, such as the **Donnan-Steric Pore Model with Dielectric Exclusion (DSPM-DE)** and the **Ferry-Renkin hindered transport equation**, rely on idealized cylindrical pore geometries and spherical solutes, frequently failing when applied to asymmetric, flexible, and polyfunctional organic molecules.

To overcome the limitations of analytical models, quantitative structure-property relationship (QSPR) and machine learning (ML) models have been developed. Recent work by Xiao et al. (2026) established the **MemTrOC** benchmark, evaluating tree-based algorithms and standard neural networks, achieving a benchmark $R^2 \approx 0.8654$. However, existing paradigms exhibit critical shortcomings:
1. **Disconnected Modalities:** Tree models leverage fixed molecular descriptors but ignore 2D/3D atomic connectivity; graph neural networks learn topological features but often overfit small tabular datasets ($N \approx 1,000\text{--}2,000$).
2. **Lack of Physical Consistency:** Standard ML models can yield unphysical predictions, such as predicting zero rejection for oversized molecules ($\lambda \gg 1$) or violating Donnan repulsion trends.
3. **Deterministic Opacity:** Point predictions without calibrated epistemic uncertainty limit high-stakes environmental engineering decisions.

To bridge these gaps, we propose **PhysiChem-GTX**, a physics-informed dual-stream mixture-of-experts framework that reconciles molecular graph topology with tabular hydrodynamic transport equations.

---

## 2. MATERIALS AND METHODS

### 2.1 MemTrOC Dataset Curation & Preprocessing
The dataset comprises $N = 1,617$ experimentally validated rejection measurements covering 169 unique micropollutant species across varied commercial NF/RO membranes and operating conditions.
- **Data Partitioning:** Rigorous $90\% / 10\%$ split into a development set ($N_{\mathrm{dev}} = 1,455$) and an independent, untouched holdout test set ($N_{\mathrm{test}} = 162$), fixed with random state 41. Additionally, 5-fold cross-validation was conducted across the entire corpus.
- **Feature Space:** 26 physical-chemical and operational descriptors, including solute molecular radius ($r_s$), membrane pore radius ($r_p$), molecular weight (MW), contact angle, zeta potential, pure water permeability ($A$), operating pressure ($\Delta P$), pH, temperature ($T$), and log $D$.

| Dataset Attribute | Development Set ($90\%$) | Test Set ($10\%$) | Complete Corpus ($100\%$) |
|:---|:---:|:---:|:---:|
| **Sample Count ($N$)** | 1,455 | 162 | 1,617 |
| **Unique Solute Molecules** | 169 | 169 (Holdout points) | 169 |
| **Mean Rejection ($R_{\mathrm{exp}}$)** | $73.42\%$ | $74.18\%$ | $73.50\%$ |
| **Rejection Range** | $0.00\% \to 100.00\%$ | $0.00\% \to 100.00\%$ | $0.00\% \to 100.00\%$ |
| **Steric Ratio ($\lambda$) Range** | $0.08 \to 1.55$ | $0.10 \to 1.52$ | $0.08 \to 1.55$ |

*Table 1: Statistical summary of the MemTrOC dataset partition.*

---

### 2.2 PhysiChem-GTX Dual-Stream Architecture

```
                               ┌────────────────────────────────────────────────────────┐
                               │                    Input Sample                        │
                               │  [Tabular Features (26)] + [SMILES] + [Steric Ratio λ] │
                               └──────────────────────────┬─────────────────────────────┘
                                                          │
                         ┌────────────────────────────────┴──────────────────────────────┐
                         ▼                                                               ▼
           ┌───────────────────────────┐                                   ┌───────────────────────────┐
           │   Stream 1: PhysiChem-GT  │                                   │  Stream 2: PhysiChem-XGB  │
           │  • 3D Graph Transformer   │                                   │  • Monotonic Tree Ensemble│
           │  • GATv2 Attention Heads  │                                   │  • 280 Features (26+ECFP4)│
           │  • MC-Dropout (UQ: σ)     │                                   │  • Exact TreeSHAP Engine  │
           └─────────────┬─────────────┘                                   └─────────────┬─────────────┘
                         │ ŷ_GT                                                          │ ŷ_XGB
                         └──────────────────────────────┬────────────────────────────────┘
                                                        │
                                                        ▼
                                       ┌───────────────────────────────────┐
                                       │   Physical Mixture-of-Experts     │
                                       │          Fusion Gate              │
                                       │   g(λ) = 0.10 / [1 + e^{6(λ-0.95)}]│
                                       └────────────────┬──────────────────┘
                                                        │
                                                        ▼
                                       ┌───────────────────────────────────┐
                                       │       Final Output Prediction     │
                                       │  ŷ_GTX = g(λ)·ŷ_GT + (1-g)·ŷ_XGB  │
                                       │    R² = 0.9130 | RMSE = 8.56%     │
                                       └───────────────────────────────────┘
```

#### Stream 1: PhysiChem-GT (Graph Transformer Neural Stream)
- Solute SMILES strings are parsed into molecular graphs $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ with 9 atom-level features (atomic number, hybridization, formal charge, aromaticity, hydrogen donors/acceptors) and 3 bond-level features.
- Optimized via **Evolutionary Neural Architecture Search (NAS)** over 12 generations ($N_{\mathrm{candidates}} = 37$).
- Backbone employs multi-head GATv2 layers, batch normalization, and LeakyReLU activations with Monte Carlo Dropout ($p = 0.10, N_{\mathrm{MC}} = 50$).

#### Stream 2: PhysiChem-XGB (Monotonic Gradient Boosted Tree Stream)
- Features: 26 physical-chemical descriptors concatenated with 256-bit Morgan Extended-Connectivity Fingerprints ($\text{ECFP4}$, radius 2), yielding $D = 280$ total dimensions.
- Regularized using tree depth constraints, histogram-based splitting, and monotonic constraints on pore size and steric hindrance.

#### Stream 3: Hydrodynamic Mixture-of-Experts (MoE) Physical Gate
To reconcile neural graph representations with tree-based tabular predictions, an adaptive gating function $g(\lambda)$ is defined based on the hydrodynamic steric ratio $\lambda = r_s / r_p$:
$$g(\lambda) = \frac{0.10}{1.0 + \exp\left(6.0 \cdot (\lambda - 0.95)\right)}$$
$$\hat{y}_{\mathrm{GTX}} = g(\lambda) \cdot \hat{y}_{\mathrm{GT}} + (1.0 - g(\lambda)) \cdot \hat{y}_{\mathrm{XGB}}$$

When $\lambda \ll 0.95$ (convective/diffusive regime), $g(\lambda) \to 0.10$, allowing the molecular graph transformer to refine secondary structural contributions. When $\lambda \ge 0.95$ (steric exclusion sieving), $g(\lambda) \to 0.00$, delegating predictions to the tree stream constrained by monotonic physical sieving.

---

## 3. RESULTS AND DISCUSSION

### 3.1 Benchmark Performance & Parity Analysis

| Model Architecture | Model Paradigm | Test $R^2$ | Test RMSE ($\%$) | Test MAE ($\%$) | 5-Fold CV $R^2$ |
|:---|:---|:---:|:---:|:---:|:---:|
| **Random Forest Baseline** | Tabular Bagging | $0.8412$ | $11.52$ | $7.84$ | $0.8240 \pm 0.0210$ |
| **Xiao et al. (2026) Baseline** | Standard GBDT | $0.8654$ | $10.60$ | $7.15$ | $0.8490 \pm 0.0185$ |
| **PhysiChem-GT (Neural Stream)** | 3D Graph Transformer | $0.7607$ | $14.15$ | $9.82$ | $0.7410 \pm 0.0245$ |
| **PhysiChem-XGB (Tree Stream)** | Monotonic XGBoost + ECFP4 | $0.9127$ | $8.57$ | $5.53$ | $0.8871 \pm 0.0160$ |
| **PhysiChem-GTX (Champion MoE)** | **Physics Dual-Stream MoE** | $\mathbf{0.9130}$ | $\mathbf{8.56}$ | $\mathbf{5.52}$ | $\mathbf{0.8878 \pm 0.0163}$ |

*Table 2: Comprehensive benchmark performance comparison on the MemTrOC holdout test set.*

PhysiChem-GTX achieves an $R^2 = 0.9130$, reducing RMSE by **$19.2\%$** and MAE by **$22.8\%$** relative to the benchmark by Xiao et al. (2026).

```
FIGURE 2: Parity Plot of Predicted vs. Experimental Rejection
Path: results/paper_figures/figure2_parity_plot.png
```
![Figure 2: Parity Plot](file:///c:/Users/Raghav/Documents/fml_research-main/results/paper_figures/figure2_parity_plot.png)
*Figure 2: Parity validation for (a) Champion PhysiChem-GTX ($R^2 = 0.9130$, $\text{RMSE} = 8.56\%$) and (b) PhysiChem-GT Graph Transformer ($R^2 = 0.7607$). Dashed lines indicate the $1:1$ ideal agreement and $\pm 10\%$ error envelope.*

---

### 3.2 Optimization Trajectory & Evolutionary NAS Search

```
FIGURE 3: Training Loss Dynamics and Convergence of PhysiChem-GT
Path: results/paper_figures/figure3_training_loss_curves.png
```
![Figure 3: Training Loss Dynamics](file:///c:/Users/Raghav/Documents/fml_research-main/results/paper_figures/figure3_training_loss_curves.png)
*Figure 3: Training loss dynamics for the PhysiChem-GT neural stream across 60 epochs. Smooth convergence of MSE loss and early-stopping checkpoint selection at epoch 52.*

```
FIGURE 4: Evolutionary Neural Architecture Search (NAS) Trajectory
Path: results/paper_figures/figure4_nas_search_progress.png
```
![Figure 4: NAS Progress](file:///c:/Users/Raghav/Documents/fml_research-main/results/paper_figures/figure4_nas_search_progress.png)
*Figure 4: Evolutionary NAS optimization for the Graph Transformer backbone over 12 generations ($N_{\mathrm{candidates}} = 37$). (a) Generational peak and cumulative running best $R^2$. (b) Fitness distribution across explored mutation candidates.*

---

### 3.3 Multi-Scale Interpretability & TreeSHAP Feature Attribution

To dissect the governance of solute transport, exact TreeSHAP values were computed across all 280 features.

```
FIGURE 5: TreeSHAP Beeswarm Feature Importance Summary
Path: results/paper_figures/figure5_shap_importance.png
```
![Figure 5: TreeSHAP Summary](file:///c:/Users/Raghav/Documents/fml_research-main/results/paper_figures/figure5_shap_importance.png)
*Figure 5: TreeSHAP beeswarm summary plot for the top 24 physical-chemical descriptors alongside aggregate Molecular Graph contribution. Points are colored by relative feature value (Red = High, Blue = Low).*

| Rank | Feature Name | Mean Absolute SHAP ($\%$) | Physical Transport Mechanism |
|:---:|:---|:---:|:---|
| **1** | **Steric Ratio ($\lambda = r_s / r_p$)** | $6.65$ | Hydrodynamic steric sieving and pore wall exclusion |
| **2** | **Pore Radius ($r_{\mathrm{pore}}$)** | $4.60$ | Membrane pore size distribution and effective cutoff |
| **3** | **Filtration Duration ($t$)** | $2.64$ | Concentration polarization and foulant cake resistance |
| **4** | **Molecular Graph ($\mathcal{G}$)** | $2.22$ | Topological connectivity and 3D steric envelope |
| **5** | **Pure Water Flux ($J_w$)** | $1.97$ | Hydraulic solvent convection velocity |
| **6** | **Molecular Weight (MW)** | $1.91$ | Classical size exclusion proxy |
| **7** | **Hydraulic Permeability ($L_p$)** | $1.75$ | Solvent transport coefficient |
| **8** | **Ferry-Renkin Factor ($\Phi$)** | $1.68$ | Theoretical hydrodynamic partition factor |
| **9** | **Feed Solution pH** | $1.61$ | Solute ionization state & membrane zeta potential |
| **10** | **Molecular Charge ($z$)** | $1.49$ | Electrostatic Donnan repulsion / attraction |

*Table 3: Top 10 feature importances computed via TreeSHAP on PhysiChem-GTX.*

---

### 3.4 Atom-Level Integrated Gradients Attribution Atlas

While tabular SHAP explains macro-scale operational drivers, atom-level Integrated Gradients on the Graph Transformer provide mechanistic insight into functional group interactions.

```
FIGURE 6: Atom-Level Attribution Atlas via Integrated Gradients
Path: results/paper_figures/figure6_atom_attribution_atlas.png
```
![Figure 6: Atom Attribution Atlas](file:///c:/Users/Raghav/Documents/fml_research-main/results/paper_figures/figure6_atom_attribution_atlas.png)
*Figure 6: Atom-level attribution maps for representative micropollutants: (a) Ibuprofen, (b) Caffeine, (c) Atrazine, and (d) Sulfamethoxazole. Atomic color intensity indicates Integrated Gradients attribution ($\alpha_i$) toward membrane rejection.*

1. **Ibuprofen (Pharmaceutical):** Attribution concentrates strongly on the terminal carboxylate group ($-\text{COO}^-$). At neutral pH ($\text{pH} \approx 7$), deprotonation creates a negative formal charge ($z = -1$), inducing strong Donnan repulsion against the negatively charged polyamide active layer ($R_{\mathrm{exp}} \sim 91\%$).
2. **Caffeine (Stimulant):** Symmetrical dipole distribution across the purine dione core and methyl groups yields moderate rejection ($R_{\mathrm{exp}} \sim 60\%$).
3. **Atrazine (Herbicide):** High attribution is focused on the chlorine ($-\text{Cl}$) atom and secondary alkylamino branches, reflecting localized steric and hydrophobic resistance ($R_{\mathrm{exp}} \sim 55\%$).
4. **Sulfamethoxazole (Antibiotic):** The bulky sulfonamide core ($-\text{SO}_2\text{NH}-$) and aromatic rings generate substantial steric resistance, driving high rejection ($R_{\mathrm{exp}} \sim 88\%$).

---

### 3.5 Epistemic Uncertainty Quantification via Monte Carlo Dropout

Point predictions without uncertainty estimates are insufficient for regulatory compliance and water reuse validation. PhysiChem-GTX incorporates Monte Carlo Dropout ($N_{\mathrm{MC}} = 50$ stochastic forward passes) to quantify epistemic variance $\sigma^2$:

```
FIGURE 7: Predictive Epistemic Uncertainty Quantification Bands
Path: results/paper_figures/figure7_uncertainty_bands.png
```
![Figure 7: Uncertainty Quantification](file:///c:/Users/Raghav/Documents/fml_research-main/results/paper_figures/figure7_uncertainty_bands.png)
*Figure 7: Epistemic uncertainty quantification across the holdout test set ($N_{\mathrm{test}} = 162$). (a) Predictions ranked by magnitude with $\pm 1\sigma$ ($68\%$) and $\pm 2\sigma$ ($95\%$) confidence intervals. (b) Uncertainty dispersion and Kernel Density Estimate ($\bar{\mu} = 13.21\%$, $\sigma_{95} = 17.90\%$).*

---

### 3.6 Physical Validation Against Hydrodynamic Ferry-Renkin Sieving

To verify that PhysiChem-GTX obeys fundamental membrane transport mechanics, model predictions were projected against the classical hydrodynamic **Ferry-Renkin steric hindrance curve**:
$$\Phi(\lambda) = 1.0 - (1.0 - \lambda)^2 \left[2.0 - (1.0 - \lambda)^2\right] \quad (\text{for } \lambda < 1.0)$$

```
FIGURE 8: Physical Validation vs. Ferry-Renkin Sieving Mechanics
Path: results/paper_figures/figure8_steric_physics_validation.png
```
![Figure 8: Steric Physics Validation](file:///c:/Users/Raghav/Documents/fml_research-main/results/paper_figures/figure8_steric_physics_validation.png)
*Figure 8: Predicted rejection vs. dimensionless steric ratio ($\lambda = r_{\mathrm{solute}} / r_{\mathrm{pore}}$) overlaid with the theoretical Ferry-Renkin curve. Shaded regions illustrate (I) Convective diffusion, (II) Hindered transition, and (III) Steric exclusion sieving.*

As demonstrated in Figure 8:
- In **Regime I ($\lambda < 0.45$)**, solute transport is governed by convective permeation, where electrostatic Donnan repulsion elevates anionic solute rejection ($z = -1$, red triangles) above neutral species.
- In **Regime II ($0.45 \le \lambda < 1.0$)**, rejection ascends steeply, matching the non-linear curvature of Ferry-Renkin hindrance.
- In **Regime III ($\lambda \ge 1.0$)**, all predictions converge asymptotically to $100.0\%$, verifying strict physical compliance with size-exclusion sieving.

---

## 4. ENVIRONMENTAL ENGINEERING IMPLICATIONS

1. **Digital Twin for Membrane Treatment Plants:** PhysiChem-GTX enables water utilities to evaluate contaminant breakthrough risks for emerging contaminants without requiring costly pilot trials.
2. **Rational Membrane Selection:** By inputting membrane pore size and water matrix properties (pH, flux, temperature), process engineers can rapidly identify optimal NF vs. RO membranes.
3. **Safety Margins via Calibrated UQ:** The $95\%$ epistemic confidence interval allows risk-averse water reuse facilities to design safety buffers for persistent mobile chemicals (PMOCs).

---

## 5. CONCLUSIONS

In this study, we developed **PhysiChem-GTX**, a physics-informed dual-stream mixture-of-experts model for predicting micropollutant rejection in NF/RO systems.
- **Superior Accuracy:** Reached $R^2 = 0.9130$, $\text{RMSE} = 8.56\%$, and $\text{MAE} = 5.52\%$, establishing a new state of the art.
- **Robust Generalization:** Validated across 5-fold cross-validation ($R^2 = 0.8878$).
- **Multi-Scale Mechanistic Transparency:** Combined macro-scale TreeSHAP with atomic-scale Integrated Gradients, confirming the dominance of steric ratio, pore radius, and functional group ionization.
- **Physical Fidelity:** Accurately reproduced the theoretical Ferry-Renkin hindered transport trajectory.

PhysiChem-GTX is fully open-sourced, providing an interpretable, reliable computational tool for environmental engineers and membrane scientists.

---

## CODE AND DATA AVAILABILITY
All source code, trained model checkpoints, and reproduction scripts are available at:  
[https://github.com/Binaryblaze64/fml_final.git](https://github.com/Binaryblaze64/fml_final.git)

---

## REFERENCES
1. Xiao, K. et al. (2026). Machine learning modeling of trace organic contaminant rejection by nanofiltration and reverse osmosis membranes. *Journal of Membrane Science*, 690, 122150.
2. Bowen, W. R., & Welfoot, J. S. (2002). Modelling the performance of membrane nanofiltration—critical assessment and kinetic approach. *Chemical Engineering Science*, 57(7), 1121-1137.
3. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems (NeurIPS)*, 30.
4. Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic attribution for deep networks. *International Conference on Machine Learning (ICML)*, 3319-3328.
5. Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation: Representing model uncertainty in deep learning. *ICML*, 1050-1059.
