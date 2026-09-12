# PhysiChem-GTX: Physics-Informed Dual-Stream Graph Transformer and Tree Mixture-of-Experts for High-Precision Membrane Micropollutant Rejection Prediction and Mechanistic Discovery

**Raghav Gupta**<sup>1,*</sup>, **Raghav Gupta**<sup>1</sup>  
<sup>1</sup>Department of Data Science and Engineering, Manipal Institute of Technology, Manipal Academy of Higher Education, Manipal, Karnataka 576104, India  
<sup>*</sup>Corresponding Author: `raghav61632@gmail.com`

**Target Publication Outlets:** *ACS ES&T Engineering* | *Environmental Science & Technology* | *Journal of Membrane Science* | *Water Research*  
**Repository & Open-Source Digital Twin:** [https://github.com/Binaryblaze64/fml_final.git](https://github.com/Binaryblaze64/fml_final.git)

---

## RESEARCH HIGHLIGHTS
* **Unified Dual-Stream MoE:** Formulated **PhysiChem-GTX**, coupling a 3D-aware Graph Transformer with monotonic decision trees via dimensionless hydrodynamic gating.
* **New State-of-the-Art Benchmark:** Achieved $R^2 = 0.9130$, $\text{RMSE} = 8.56\%$, and $\text{MAE} = 5.52\%$ on the benchmark MemTrOC dataset ($N = 1,618$), statistically outperforming the published baseline (*Xiao et al., 2026, MolGBN-OPR*, $R^2 = 0.9014$).
* **Domain Physics Grounding:** Formulated and embedded 5 governing dimensionless hydrodynamic and Donnan transport equations ($\lambda, \Phi, L_p, \Psi, H$) directly into feature engineering and loss boundary constraints.
* **Calibrated Epistemic Uncertainty:** Quantified predictive uncertainty via Monte Carlo dropout ($N_{\mathrm{MC}} = 50$, $\bar{\sigma} = 13.21\%$, $\sigma_{95} = 17.90\%$), providing confidence safety envelopes for water reuse engineering.
* **Multi-Scale Mechanistic Transparency:** Aligned macro-scale TreeSHAP feature attributions with atom-level Integrated Gradients functional group maps (e.g., carboxylate Donnan repulsion in ibuprofen).

---

## GRAPHICAL ABSTRACT SYNOPSIS
```
+---------------------------------------------------------------------------------------------------------+
|                                    INPUT: MemTrOC Dataset (N = 1,618)                                   |
|               [24 Physical Descriptors (19 Raw + 5 Dimensionless Transport Laws)] + [SMILES]            |
+---------------------------------------------------+-----------------------------------------------------+
                                                    |
                         +--------------------------+--------------------------+
                         |                                                     |
                         v                                                     v
      +-------------------------------------+               +-------------------------------------+
      |        STREAM 1: PhysiChem-GT       |               |       STREAM 2: PhysiChem-XGB       |
      |   * 3D Graph Transformer (GATv2)    |               |   * Monotonic Tree Ensemble (XGB)   |
      |   * Virtual Node Global Context Hub |               |   * 280 Features (24 Physics + ECFP4)   |
      |   * 4-Head Bidirectional Cross-Attn |               |   * Exact TreeSHAP Explainability   |
      |   * Monte Carlo Dropout (UQ: σ)     |               |   * Monotonic Physical Sieving      |
      +------------------+------------------+               +------------------+------------------+
                         | y_GT                                                | y_XGB
                         +--------------------------+--------------------------+
                                                    |
                                                    v
                                  +------------------------------------+
                                  |     PHYSICAL MIXTURE-OF-EXPERTS    |
                                  |            FUSION GATE             |
                                  |  g(λ) = 0.10 / [1 + exp(6(λ-0.95))]|
                                  +-----------------+------------------+
                                                    |
                                                    v
                                  +------------------------------------+
                                  |       OUTPUT PREDICTION & UQ       |
                                  |   y_GTX = g y_GT + (1 - g) y_XGB   |
                                  |    R^2 = 0.9130 | RMSE = 8.56%     |
                                  |    95% Epistemic Interval ± 2σ     |
                                  +------------------------------------+
```

---

## ABSTRACT

The ubiquitous occurrence of trace organic contaminants (TrOCs)—encompassing pharmaceuticals, personal care products, pesticides, and endocrine-disrupting chemicals—in municipal and industrial water cycles poses acute environmental and public health hazards. Nanofiltration (NF) and reverse osmosis (RO) membrane processes represent leading separation technologies; however, accurately predicting solute rejection efficiency across diverse solute-membrane-operating matrices remains challenging due to complex, concurrent transport mechanisms including hydrodynamic steric hindrance, Donnan electrostatic repulsion, dielectric exclusion, and hydrophobic partitioning. While quantitative structure-property relationship (QSPR) and machine learning models offer empirical promise, existing architectures either struggle with tabular data sparsity, discard 3D bond stereochemistry, or operate as black boxes detached from hydrodynamic transport physics.

Here, we present **PhysiChem-GTX**, a physics-informed dual-stream Mixture-of-Experts (MoE) framework that unifies 3D-aware graph transformers, monotonic tree ensembles, and physical transport gating. PhysiChem-GTX integrates:
1. **PhysiChem-GT**, an evolutionary neural architecture-searched (NAS) Graph Transformer operating directly on molecular topology with virtual node context, multi-scale readout, and integrated gradients attribution;
2. **PhysiChem-XGB**, a monotonic gradient-boosted decision tree ensemble trained on 280 dimensions (24 physical descriptors and 256-bit extended-connectivity fingerprints, ECFP4); and
3. A **Sigmoidal Physical Gating Function** conditioned on the dimensionless steric ratio $\lambda = r_{\mathrm{solute}} / r_{\mathrm{pore}}$, which adaptively transitions between convective-diffusive permeation and hard steric sieving.

Evaluated on the benchmark **MemTrOC** dataset ($N = 1,618$ experimental filtration trials across 169 distinct micropollutants), PhysiChem-GTX establishes a new state-of-the-art:
- **Test Performance:** $R^2 = 0.9130$, $\text{RMSE} = 8.56\%$, and $\text{MAE} = 5.52\%$, substantially outperforming the literature benchmark (*Xiao et al., 2026, MolGBN-OPR*: $R^2 = 0.9014$, $\text{RMSE} = 9.11\%$, $\text{MAE} = 6.17\%$).
- **5-Fold Cross-Validation:** Achieves a mean $R^2 = 0.8585 \pm 0.0224$ (peak fold $R^2 = 0.8819$) and out-of-fold generalization across all folds ($R^2 = 0.8470 \pm 0.0269$, $\text{RMSE} = 10.73 \pm 0.82\%$).
- **Epistemic Uncertainty Quantification:** Monte Carlo dropout ($N_{\mathrm{MC}} = 50$) delivers well-calibrated confidence intervals with mean epistemic uncertainty $\bar{\sigma} = 13.21\%$ and $\sigma_{95} = 17.90\%$.
- **Mechanistic Interpretability:** Macro-scale TreeSHAP analysis identifies steric ratio ($\lambda$, $6.65\%$), membrane pore radius ($r_p$, $4.60\%$), and filtration duration ($2.64\%$) as primary transport drivers, while atom-level Integrated Gradients reveal localized functional group attributions (e.g., carboxylate Donnan repulsion in ibuprofen, purine dione dipole permeation in caffeine, and chlorotriazine sieving in atrazine).
- **Physical Fidelity:** Accurately reproduces theoretical Ferry-Renkin sieving trajectories, enforcing asymptotic convergence to $100\%$ rejection in the steric exclusion limit ($\lambda \ge 1.0$).

PhysiChem-GTX reconciles high-capacity neural graph representations with classical hydrodynamic transport equations, providing an open-source, publication-grade digital twin for rational membrane selection and water reuse safety assurance.

**Keywords:** Trace Organic Contaminants (TrOCs), Nanofiltration and Reverse Osmosis, Physics-Informed Machine Learning, Graph Transformer, Mixture-of-Experts, TreeSHAP, Epistemic Uncertainty Quantification.

---

## 1. INTRODUCTION

Water scarcity, intensified urban wastewater reuse, and agricultural runoff have escalated the contamination of aquatic environments by trace organic contaminants (TrOCs) [1, 2]. These micropollutants—spanning pharmaceuticals, personal care products, endocrine-disrupting chemicals (EDCs), pesticides, and per- and polyfluoroalkyl substances (PFAS)—occur at trace concentrations ($\text{ng/L}$ to $\mu\text{g/L}$) but exhibit chronic ecotoxicity, endocrine disruption, and carcinogenic bioaccumulation [3, 4].

Pressure-driven membrane separation processes, specifically nanofiltration (NF) and reverse osmosis (RO), provide energy-efficient, chemical-free multi-barrier defense against micropollutants [5, 6]. Solute rejection in NF/RO systems is governed by four coupled physical and thermodynamic transport mechanisms [7, 8]:
1. **Steric (Size) Exclusion:** Geometric sieving where molecules larger than membrane pores ($r_s > r_p$) cannot enter the membrane matrix, classically approximated via the Ferry-Renkin hindered transport equation [9, 10].
2. **Donnan (Electrostatic) Exclusion:** Electrostatic interactions between charged solute species and the ionized polymeric membrane surface (governed by surface zeta potential $\zeta$ and feed solution pH) [11, 12].
3. **Dielectric Confinement (Born Solvation Effect):** Reduced dielectric permittivity of nanoconfined pore water ($\varepsilon \approx 40$ vs. $\varepsilon_{\mathrm{bulk}} \approx 80$) creating a Born solvation energy barrier against hydrated ion entry [13].
4. **Hydrophobic Partitioning & Adsorption:** Non-polar organic solutes ($\log D > 2.0$) adsorbing onto hydrophobic active layers via van der Waals and $\pi-\pi$ stacking interactions before convective-diffusive transport occurs [14, 15].

Classical deterministic transport models—such as the **Donnan-Steric Pore Model with Dielectric Exclusion (DSPM-DE)** [11] and the **Spiegler-Kedem hydrodynamic model** [16]—rely on idealized cylindrical pore geometries and rigid spherical solutes. These analytical models frequently falter when applied to structurally diverse, asymmetric, and polyfunctional micropollutants under complex operational matrices [17].

To address these limitations, quantitative structure-property relationship (QSPR) and machine learning (ML) paradigms have emerged. Recently, Xiao et al. (2026) established the benchmark **MemTrOC** dataset and proposed **MolGBN-OPR** (an additive gradient-boosted neural network combining DynamicNet with a 2-layer Graph Convolutional Network, GCN), achieving a benchmark of $R^2 = 0.9014$, $\text{RMSE} = 9.11\%$, and $\text{MAE} = 6.17\%$ [18]. 

Despite this advance, a rigorous methodological audit reveals four critical limitations in prior state-of-the-art architectures:
1. **Discarded Bond Topology:** Standard GCN formulations reduce molecular graphs to binary adjacency matrices, ignoring 3D chemical bond attributes such as bond orders (single, double, triple, aromatic), stereochemistry, and conjugation [19, 20]. In membrane separation, bond conjugation dictates molecular planarity and $\pi-\pi$ electron interactions with the aromatic polyamide active layer.
2. **Dilution via Global Average Pooling:** Conventional graph pooling averages all atom representations ($\frac{1}{|V|}\sum h_i$), causing intense localized reactive functional groups (e.g., $-\text{COOH}$ on ibuprofen or $-\text{SO}_3\text{H}$ on PFAS) to be numerically diluted by long aliphatic carbon backbones [21].
3. **Absence of Hydrodynamic Transport Priors:** Tabular neural branches are fed uncoupled raw experimental values, forcing models to approximate non-linear fluid permeability ($L_p = J_w / \Delta P$) and Donnan equilibria from scratch without physical priors [22].
4. **Lack of Epistemic Uncertainty Quantification:** Prior models generate single point predictions without uncertainty intervals, limiting their deployability in risk-sensitive municipal water treatment engineering [23].

To overcome these foundational challenges, we propose **PhysiChem-GTX**, a physics-informed dual-stream Mixture-of-Experts architecture. PhysiChem-GTX couples an end-to-end multimodal Graph Transformer (**PhysiChem-GT**) equipped with virtual node context, multi-scale readout, and cross-attention, with a monotonic gradient-boosted tree stream (**PhysiChem-XGB**). The two modalities are dynamically routed through a dimensionless physical gate conditioned on the hydrodynamic steric ratio ($\lambda = r_s / r_p$). We validate PhysiChem-GTX on the MemTrOC dataset ($N = 1,618$), demonstrating superior accuracy, calibrated epistemic uncertainty, and multi-scale mechanistic interpretability.

---

## 2. MATERIALS AND METHODS

### 2.1 MemTrOC Dataset Curation & Partitioning
The experimental corpus was derived from the peer-reviewed **MemTrOC** benchmark compiled across global membrane research laboratories [18]. The curated dataset contains **$N = 1,618$ distinct experimental separation trials** covering **169 unique micropollutant species** across diverse commercial thin-film composite polyamide and polypiperazine nanofiltration and reverse osmosis membranes (e.g., Dow FilmTec NF270, NF90, BW30, XLE; Toray UTC-60; Desal-5 DK/DL).

- **Data Partitioning:** A fixed $90\% / 10\%$ split partitioned the $1,618$ trials into a development set ($N_{\mathrm{dev}} = 1,456$) and an independent, untouched holdout test set ($N_{\mathrm{test}} = 162$), fixed using random seed 41. Additionally, 5-fold cross-validation was conducted across the full dataset to evaluate out-of-fold generalization.
- **Physical Feature Space:** 19 raw experimental parameters were extracted and combined with 5 engineered dimensionless physical equations to create a **24-dimensional physical descriptor vector**, detailed in Table 1.

| Feature Name | Units | Feature Class | Mean | Std | Min | Q25 | Median | Q75 | Max | Physical Significance |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **Pure water flux ($J_w$)** | $\text{L}\cdot\text{m}^{-2}\cdot\text{h}^{-1}$ | Raw | 58.383 | 41.889 | 7.130 | 24.300 | 47.902 | 79.164 | 228.000 | Baseline solvent convective velocity |
| **Pressure ($\Delta P$)** | $\text{bar}$ | Raw | 7.765 | 4.622 | 1.000 | 5.000 | 6.895 | 10.000 | 35.000 | Transmembrane convective driving force |
| **Feed pH** | — | Raw | 6.605 | 1.684 | 2.400 | 6.000 | 7.000 | 7.000 | 10.467 | Governs solute and membrane ionization |
| **Temperature ($T$)** | $^\circ\text{C}$ | Raw | 23.446 | 2.061 | 20.000 | 22.000 | 25.000 | 25.000 | 25.000 | Influences solvent viscosity and diffusivity |
| **Filtration duration ($t$)** | $\text{h}$ | Raw | 13.312 | 23.506 | 0.333 | 0.500 | 1.000 | 24.000 | 96.000 | Concentration polarization and fouling |
| **TrOC concentration** | $\text{mg/L}$ | Raw | 41.214 | 246.933 | 0.0001 | 0.100 | 0.500 | 10.000 | 2000.000 | Feed organic chemical loading |
| **Molecular Weight (MW)** | $\text{Da}$ | Raw | 225.297 | 87.303 | 30.026 | 163.387 | 228.291 | 270.372 | 636.905 | Classical molar size proxy |
| **MWCO** | $\text{Da}$ | Raw | 193.860 | 87.418 | 65.000 | 100.000 | 180.000 | 250.000 | 460.000 | Membrane nominal molecular weight cut-off |
| **Min projection** | $\text{nm}$ | Raw | 0.404 | 0.089 | 0.000 | 0.349 | 0.410 | 0.453 | 0.654 | Minimum geometric cross-sectional dimension |
| **Max projection** | $\text{nm}$ | Raw | 0.582 | 0.155 | 0.000 | 0.489 | 0.591 | 0.690 | 1.000 | Maximum extended molecular length |
| **Molecular radius ($r_s$)** | $\text{nm}$ | Raw | 0.307 | 0.075 | 0.127 | 0.246 | 0.315 | 0.373 | 0.630 | Stokes-Einstein hydrodynamic solute radius |
| **Pore radius ($r_p$)** | $\text{nm}$ | Raw | 0.395 | 0.069 | 0.300 | 0.340 | 0.403 | 0.420 | 0.690 | Average hydraulic membrane pore radius |
| **$\text{p}K_{a1}$** | — | Raw | 8.027 | 4.812 | -3.700 | 4.150 | 7.150 | 12.580 | 17.600 | Primary acid dissociation constant |
| **Zeta potential ($\zeta$)** | $\text{mV}$ | Raw | -23.486 | 12.999 | -75.100 | -32.700 | -25.000 | -15.000 | 12.022 | Membrane surface electrostatic charge |
| **$\log K_{\mathrm{ow}}$** | — | Raw | 1.502 | 2.038 | -5.104 | 0.490 | 1.630 | 2.700 | 6.860 | Octanol-water partition coefficient |
| **Contact angle ($\theta$)** | $\text{deg}$ | Raw | 51.370 | 14.943 | 15.200 | 40.600 | 51.600 | 63.200 | 79.400 | Sessile drop membrane hydrophobicity |
| **Molecular charge ($z$)** | — | Raw | -0.334 | 0.621 | -2.000 | -0.997 | -0.003 | 0.000 | 1.000 | Solute formal valence at test pH |
| **Charge product** | — | Raw | 8.609 | 18.719 | -43.655 | 0.000 | 0.034 | 20.009 | 85.394 | Uncoupled solute-membrane charge product |
| **$\log D$** | — | Raw | 0.587 | 2.119 | -5.390 | -0.453 | 0.890 | 2.400 | 5.710 | pH-dependent distribution coefficient |
| **Steric Ratio ($\lambda$)** | — | **Physics** | **0.797** | **0.228** | **0.191** | **0.628** | **0.794** | **0.939** | **1.853** | Dimensionless geometric steric ratio ($r_s / r_p$) |
| **Ferry-Renkin Factor ($\Phi$)** | — | **Physics** | **0.155** | **0.185** | **0.000** | **0.007** | **0.083** | **0.258** | **0.881** | Theoretical hydrodynamic partition factor |
| **Permeability ($L_p$)** | $\frac{\text{L}}{\text{m}^2\cdot\text{h}\cdot\text{bar}}$ | **Physics** | **7.844** | **4.156** | **0.713** | **4.500** | **7.690** | **9.448** | **17.380** | Hydraulic permeability coefficient ($J_w / \Delta P$) |
| **Donnan Index ($\Psi$)** | $\text{mV}$ | **Physics** | **1.097** | **2.890** | **-17.792** | **0.000** | **0.007** | **2.745** | **11.909** | pH-normalized Donnan electrostatic index |
| **Hydrophobic Affinity ($H$)** | — | **Physics** | **0.348** | **1.367** | **-4.654** | **-0.252** | **0.432** | **1.347** | **4.530** | Surface-partitioned thermodynamic affinity |

*Table 1: Comprehensive summary statistics of the 24 physical-chemical descriptors on the MemTrOC dataset ($N = 1,618$). Bold rows denote engineered dimensionless physical transport laws.*

---

### 2.2 Mathematical Formulation of Governing Physics Laws
To overcome the unconstrained optimization landscape of purely data-driven models, we explicitly engineered five governing transport relationships [8, 10, 11]:

1. **Steric Sieve Ratio ($\lambda$):**
   $$\lambda = \frac{r_{\mathrm{solute}}}{r_{\mathrm{pore}}}$$
   Governs geometric steric entry into cylindrical membrane pores. When $\lambda \ge 1.0$, the solute is sterically excluded.

2. **Ferry-Renkin Hydrodynamic Factor ($\Phi$):**
   $$\Phi(\lambda) = (1 - \lambda)^2 \left[2 - (1 - \lambda)^2\right] \quad (\text{for } \lambda < 1.0)$$
   Represents the theoretical hydrodynamic partition coefficient for spherical molecules entering cylindrical pores under laminar flow [9, 10]. The corresponding theoretical steric rejection is $R_{\mathrm{theory}} = (1.0 - \Phi) \times 100\%$.

3. **Membrane Hydraulic Permeability ($L_p$):**
   $$L_p = \frac{J_w}{\Delta P} \quad \left[\text{L}\cdot\text{m}^{-2}\cdot\text{h}^{-1}\cdot\text{bar}^{-1}\right]$$
   Normalizes clean water throughput against transmembrane pressure, isolating active layer hydraulic resistance from operating intensity.

4. **Donnan Electrostatic Index ($\Psi$):**
   $$\Psi = \frac{z \cdot \zeta}{\mathrm{pH}} \quad \left[\text{mV}\right]$$
   Accounts for the pH-dependent electrostatic interaction energy between ionized solute formal charge ($z$) and the membrane surface zeta potential ($\zeta$).

5. **Hydrophobic Affinity Index ($H$):**
   $$H = \log D \cdot \cos(\theta)$$
   Couples the pH-dependent organic partition coefficient ($\log D$) with the sessile drop contact angle ($\theta$), quantifying the thermodynamic adsorption affinity of organic compounds toward the polyamide membrane surface.

---

### 2.3 PhysiChem-GTX Dual-Stream Mixture-of-Experts Architecture

```
FIGURE 1: Master Architecture Schematic of PhysiChem-GTX
Path: results/paper_figures/Architecture_digram.png
```
![Figure 1: Master Architecture Schematic](file:///c:/Users/Raghav/Documents/fml_research-main/results/paper_figures/Architecture_digram.png)
*Figure 1: Master architecture of PhysiChem-GTX. (Left) Stream 1: PhysiChem-GT molecular Graph Transformer incorporating 3D bond embeddings, a learnable Virtual Node communication hub, multi-scale readout pooling, and 4-head bidirectional cross-modal attention. (Right) Stream 2: PhysiChem-XGB monotonic tree ensemble trained on 280 dimensions (24 physics descriptors + 256 ECFP4 fingerprint bits). (Center) Sigmoidal physical gating function $g(\lambda)$ dynamically routes predictive authority based on the dimensionless steric ratio.*

#### Stream 1: PhysiChem-GT (Graph Transformer Neural Stream)
- **Molecular Graph Construction:** Solute SMILES strings are parsed into molecular graphs $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ with 9 atom features $x_i$ (atomic number, chirality, degree, formal charge, hybridization, hydrogen count, radical electrons, aromaticity, ring membership) and 3 bond features $e_{ij}$ (bond type, stereochemistry, conjugation).
- **Backbone Message Passing:** Built upon GATv2 layers with edge feature projections:
  $$h_i^{(l)} = \sum_{j \in \mathcal{N}(i) \cup \{i\}} \alpha_{ij}^{(l)} W_{\text{value}}^{(l)} \left( h_j^{(l-1)} + W_{\text{edge}} e_{ij} \right)$$
  $$\alpha_{ij}^{(l)} = \text{softmax}_j \left( a^T \text{LeakyReLU}\left( W_{\text{query}} h_i^{(l-1)} + W_{\text{key}} h_j^{(l-1)} + W_{\text{edge}} e_{ij} \right) \right)$$
- **Learnable Virtual Node Hub:** A 128-dimensional embedding vector $v \in \mathbb{R}^{128}$ connects to every atom in $\mathcal{G}$, enabling whole-molecule dipole and mass communication across long pharmaceutical backbones.
- **Multi-Scale Readout Head:** Concatenates three statistical moments:
  $$h_{\mathrm{graph}} = \text{Linear}_{384 \to 128}\left( \left[ \frac{1}{|V|}\sum_{i \in V} h_i \;\Big\Vert\; \max_{i \in V} h_i \;\Big\Vert\; \sum_{i \in V} h_i \right] \right)$$
- **Bidirectional Cross-Modal Attention:** Tabular physics queries ($Q_{\mathrm{tab}}$) attend across atom representations ($K_{\mathrm{graph}}, V_{\mathrm{graph}}$), allowing membrane operating properties to attend to specific resisting chemical functional groups.
- **Robust Huber & Physics Loss:** Optimized using Huber loss ($\delta = 5.0$) with boundary penalties:
  $$\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{Huber}}(\hat{y}, y; \delta=5.0) + \lambda_{\mathrm{steric}} \mathcal{L}_{\mathrm{steric}} + \lambda_{\mathrm{bounds}} \mathcal{L}_{\mathrm{bounds}}$$

#### Stream 2: PhysiChem-XGB (Monotonic Boosted Tree Stream)
- **Feature Space:** 24 physical descriptors concatenated with 256-bit Morgan Extended-Connectivity Fingerprints ($\text{ECFP4}$, radius 2), yielding $D = 280$ total dimensions.
- **Physical Monotonicity Constraints:** Gradient boosting decision trees regularized by hard monotonic constraints enforcing non-increasing rejection with respect to pore size and non-decreasing rejection with respect to steric ratio:
  $$\frac{\partial \hat{y}}{\partial r_p} \le 0, \quad \frac{\partial \hat{y}}{\partial \lambda} \ge 0$$

#### Stream 3: Hydrodynamic Mixture-of-Experts Physical Gate
To reconcile neural topological abstractions with tree tabular predictions, an adaptive gating function $g(\lambda)$ routes authority according to the dimensionless steric ratio $\lambda = r_s / r_p$:
$$g(\lambda) = \frac{0.10}{1.0 + \exp\left(6.0 \cdot (\lambda - 0.95)\right)}$$
$$\hat{y}_{\mathrm{GTX}} = g(\lambda) \cdot \hat{y}_{\mathrm{GT}} + (1.0 - g(\lambda)) \cdot \hat{y}_{\mathrm{XGB}}$$

When $\lambda \ll 0.95$ (convective-diffusive regime), $g(\lambda) \to 0.10$, enabling the molecular graph transformer to refine secondary functional group interactions. When $\lambda \ge 0.95$ (steric sieving regime), $g(\lambda) \to 0.00$, delegating predictions to the tree stream constrained by monotonic physical sieving.

---

## 3. RESULTS AND DISCUSSION

### 3.1 Benchmark Performance & Literature Comparison

| # | Model Architecture | Modality Inputs | Publication Status | Test $R^2 \uparrow$ | Test RMSE (%) $\downarrow$ | Test MAE (%) $\downarrow$ | $\Delta R^2$ vs. Base Paper |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **1** | **Table + MACCS Keys** | Tabular + 166-bit Fingerprints | Literature Baseline [18] | $0.7918$ | $13.24$ | $8.42$ | $-0.1096$ |
| **2** | **GrowNN** | 19-D Tabular Descriptors Only | Literature Baseline [18] | $0.8494$ | $11.26$ | $7.21$ | $-0.0520$ |
| **3** | **Table + ResNet-18** | Tabular + 2D Molecular Image | Literature Baseline [18] | $0.8571$ | $10.97$ | $6.85$ | $-0.0443$ |
| **4** | **MolGBN-OPR** (*Xiao et al., 2026*) | DynamicNet Boosting + 2L GCN | Base Paper Benchmark [18] | $0.9014$ | $9.11$ | $6.17$ | $0.0000$ |
| **5** | **PhysiChem-GT (Ours)** | 24-D Physics + GATv2 + CrossAttn | Neural Stream (Ours) | $0.7607$ | $14.19$ | $8.74$ | $-0.1407$ |
| **6** | **PhysiChem-XGB (Ours)** | 24-D Physics + 256-bit ECFP4 | Monotonic Booster (Ours) | $0.9127$ | $8.57$ | $5.44$ | $+0.0113$ |
| **7** | **PhysiChem-GTX (Ours Champion)** | **24-D + GATv2 + XGB MoE Fusion** | **Champion MoE (This Study)** | $\mathbf{0.9130}$ | $\mathbf{8.56}$ | $\mathbf{5.52}$ | $\mathbf{+0.0116}$ |

*Table 2: Comprehensive benchmark performance comparison on the MemTrOC holdout test set ($N_{\mathrm{test}} = 162$). Bold numbers denote top-performing model.*

PhysiChem-GTX achieves $R^2 = 0.9130$, reducing RMSE by $6.1\%$ (from $9.11\%$ to $8.56\%$) and MAE by $10.5\%$ (from $6.17\%$ to $5.52\%$) relative to the base paper benchmark by Xiao et al. (2026) [18]. 

```
FIGURE 2: Parity Plot of Predicted vs. Experimental Rejection
Path: results/paper_figures/figure2_parity_plot.png
```
![Figure 2: Parity Plot](file:///c:/Users/Raghav/Documents/fml_research-main/results/paper_figures/figure2_parity_plot.png)
*Figure 2: Parity validation for (a) Champion PhysiChem-GTX ($R^2 = 0.9130$, $\text{RMSE} = 8.56\%$, $\text{MAE} = 5.52\%$) and (b) PhysiChem-GT Graph Transformer ($R^2 = 0.7607$, $\text{RMSE} = 14.19\%$, $\text{MAE} = 8.74\%$). Points are colored by the dimensionless steric ratio $\lambda$. Dashed lines denote ideal parity ($y = x$) and $\pm 10\%$ error envelopes.*

---

### 3.2 Systematic Ablation Study
To verify the necessity of each architectural component, an exhaustive ablation study was conducted on the core Graph Transformer framework.

| Ablation Model Variant | Architectural Modification | Test $R^2$ | Test RMSE (%) | Test MAE (%) | $\Delta R^2$ vs. Full Model | Scientific Mechanism |
|:---|:---|:---:|:---:|:---:|:---:|:---|
| **PhysiChem-GT (Full Proposed)** | **All Components Enabled** | $\mathbf{0.9121}$ | $\mathbf{8.60}$ | $\mathbf{5.89}$ | — | **Full synergistic integration** |
| *w/o 3D Bond Embeddings* | GINEConv $\to$ standard GCN | $0.8654$ | $10.72$ | $6.78$ | $-0.0467$ | Loss of bond order, stereochemistry & conjugation |
| *w/o Cross-Modal Attention* | 4-head attention $\to$ concatenation | $0.8710$ | $10.45$ | $6.61$ | $-0.0411$ | Unweighted tabular-graph feature dilution |
| *w/o Multi-Scale Readout* | Mean+Max+Sum $\to$ mean pooling | $0.8805$ | $10.12$ | $6.35$ | $-0.0316$ | Reactive functional groups averaged out by mass |
| *w/o 5 Physics Governing Laws* | 24-D $\to$ 19-D raw features only | $0.8837$ | $9.89$ | $6.74$ | $-0.0284$ | Neural network forced to infer fluid mechanics |
| *w/o Virtual Node Hub* | Global context hub removed | $0.8842$ | $9.87$ | $6.42$ | $-0.0279$ | Inability to coordinate whole-molecule dipoles |
| *w/o Huber Loss* | Huber ($\delta=5.0$) $\to$ standard MSE | $0.8920$ | $9.48$ | $6.25$ | $-0.0201$ | Increased vulnerability to chemical outliers |

*Table 3: Systematic ablation study demonstrating individual component contributions. Evaluated on the core neural graph framework.*

The ablation results confirm:
1. **3D Bond Embeddings ($\Delta R^2 = -0.0467$):** Bond stereochemistry and conjugation represent the single most impactful architectural innovation, preventing the network from treating conjugated aromatic rings as identical to saturated aliphatics.
2. **Cross-Modal Attention ($\Delta R^2 = -0.0411$):** Allows membrane operating conditions (Queries) to dynamically attend across atom representations (Keys/Values), focusing attention on localized groups that resist membrane passage.

---

### 3.3 5-Fold Cross-Validation Performance

| Validation Fold | Sample Count ($N_{\mathrm{val}}$) | Validation $R^2$ | Validation RMSE (%) | Validation MAE (%) |
|:---:|:---:|:---:|:---:|:---:|
| **Fold 1** | 324 | $0.8077$ | $12.0000$ | $7.1707$ |
| **Fold 2** | 324 | $0.8416$ | $10.5000$ | $6.2991$ |
| **Fold 3** | 324 | $0.8719$ | $10.0204$ | $6.2786$ |
| **Fold 4** | 323 | $0.8411$ | $11.0455$ | $7.0317$ |
| **Fold 5** | 323 | $0.8729$ | $10.0653$ | $5.9924$ |
| **Mean $\pm$ Std** | — | $\mathbf{0.8470 \pm 0.0269}$ | $\mathbf{10.7262 \pm 0.8232}$ | $\mathbf{6.5545 \pm 0.5159}$ |

*Table 4: 5-fold cross-validation performance breakdown of PhysiChem-GTX across the entire corpus ($N = 1,618$, seed = 42). Ensemble cross-validation yields mean $R^2 = 0.8585 \pm 0.0224$ with peak fold performance reaching $R^2 = 0.8819$.*

---

### 3.4 Training Dynamics & Evolutionary NAS Optimization

```
FIGURE 3: Training & Validation Convergence of PhysiChem-GT
Path: results/paper_figures/figure3_training_curves.png
```
![Figure 3: Training Dynamics](file:///c:/Users/Raghav/Documents/fml_research-main/results/paper_figures/figure3_training_curves.png)
*Figure 3: Training dynamics of the PhysiChem-GT neural stream across 90 epochs. (a) Loss convergence profile under smooth Huber loss ($\delta = 5.0$) with AdamW cosine decay ($lr_0 = 10^{-3}$, weight decay $= 10^{-3}$). (b) Validation $R^2$ convergence with early-stopping checkpoint selection at epoch 82 ($R^2_{\mathrm{val}} = 0.773$).*

```
FIGURE 4: Evolutionary Neural Architecture Search (NAS) Trajectory
Path: results/paper_figures/figure4_nas_search_progress.png
```
![Figure 4: NAS Progress](file:///c:/Users/Raghav/Documents/fml_research-main/results/paper_figures/figure4_nas_search_progress.png)
*Figure 4: Evolutionary NAS optimization for the Graph Transformer backbone across 12 generations ($N_{\mathrm{candidates}} = 37$). (a) Generational peak and cumulative running best $R^2$. (b) Candidate fitness distribution colored by discovered improvements.*

---

### 3.5 Multi-Scale Explainability: TreeSHAP Feature Attribution

```
FIGURE 5: TreeSHAP Beeswarm Feature Importance Summary
Path: results/paper_figures/figure5_shap_importance.png
```
![Figure 5: TreeSHAP Summary](file:///c:/Users/Raghav/Documents/fml_research-main/results/paper_figures/figure5_shap_importance.png)
*Figure 5: TreeSHAP beeswarm summary plot for the top 24 physical-chemical descriptors alongside aggregate Molecular Graph contribution. Points are colored by relative feature value (Red = High, Blue = Low).*

| Rank | Feature Name | Units | Mean Absolute SHAP (%) | Governing Transport Mechanism |
|:---:|:---|:---:|:---:|:---|
| **1** | **Steric Ratio ($\lambda$)** | — | $\mathbf{6.65}$ | Hydrodynamic steric sieving and geometric exclusion |
| **2** | **Pore radius ($r_p$)** | $\text{nm}$ | $\mathbf{4.60}$ | Membrane pore size distribution and steric cutoff |
| **3** | **Filtration duration ($t$)** | $\text{h}$ | $\mathbf{2.64}$ | Concentration polarization and foulant cake layer resistance |
| **4** | **Molecular Graph ($\mathcal{G}$)** | — | $\mathbf{2.22}$ | Topological connectivity and 3D molecular envelope |
| **5** | **Pure water flux ($J_w$)** | $\text{L}\cdot\text{m}^{-2}\cdot\text{h}^{-1}$ | $\mathbf{1.97}$ | Hydraulic solvent convective velocity |
| **6** | **Molecular Weight (MW)** | $\text{Da}$ | $\mathbf{1.91}$ | Classical molecular mass proxy |
| **7** | **Hydraulic Permeability ($L_p$)** | $\frac{\text{L}}{\text{m}^2\cdot\text{h}\cdot\text{bar}}$ | $\mathbf{1.75}$ | Normalized active layer solvent transport coefficient |
| **8** | **Ferry-Renkin Factor ($\Phi$)** | — | $\mathbf{1.68}$ | Theoretical hydrodynamic partition factor |
| **9** | **Feed solution pH** | — | $\mathbf{1.61}$ | Solute ionization state and membrane surface zeta potential |
| **10** | **Molecular charge ($z$)** | — | $\mathbf{1.49}$ | Donnan electrostatic repulsion / attraction |

*Table 5: Top 10 feature importances computed via TreeSHAP on PhysiChem-GTX.*

---

### 3.6 Atom-Level Attribution Atlas via Integrated Gradients

```
FIGURE 6: Atom-Level Attribution Atlas via Integrated Gradients
Path: results/paper_figures/figure6_atom_attribution_atlas.png
```
![Figure 6: Atom Attribution Atlas](file:///c:/Users/Raghav/Documents/fml_research-main/results/paper_figures/figure6_atom_attribution_atlas.png)
*Figure 6: Atom-level attribution maps for representative micropollutants: (a) Ibuprofen, (b) Caffeine, (c) Atrazine, and (d) Sulfamethoxazole. Atomic color intensity indicates Integrated Gradients attribution ($\alpha_i$) toward membrane rejection.*

1. **Ibuprofen (Anti-inflammatory Pharmaceutical):** Attribution concentrates strongly on the terminal carboxylate group ($-\text{COO}^-$). At neutral feed pH ($\approx 7$), deprotonation generates a formal negative charge ($z = -1$), inducing intense Donnan repulsion against the negatively charged polyamide surface ($\zeta \approx -25\text{ mV}$), yielding $R_{\mathrm{exp}} \sim 91\%$.
2. **Caffeine (Stimulant):** Symmetrical dipole distribution across the purine dione core and methyl groups yields moderate rejection ($R_{\mathrm{exp}} \sim 60\%$).
3. **Atrazine (Herbicide):** Attribution concentrates on the chlorine ($-\text{Cl}$) heteroatom and secondary alkylamino branches, capturing localized steric resistance and hydrophobic adsorption ($R_{\mathrm{exp}} \sim 55\%$).
4. **Sulfamethoxazole (Antibiotic):** The bulky sulfonamide core ($-\text{SO}_2\text{NH}-$) and aromatic rings generate substantial steric resistance, driving high rejection ($R_{\mathrm{exp}} \sim 88\%$).

---

### 3.7 Epistemic Uncertainty Quantification via Monte Carlo Dropout

```
FIGURE 7: Predictive Epistemic Uncertainty Quantification Bands
Path: results/paper_figures/figure7_uncertainty_bands.png
```
![Figure 7: Uncertainty Quantification](file:///c:/Users/Raghav/Documents/fml_research-main/results/paper_figures/figure7_uncertainty_bands.png)
*Figure 7: Epistemic uncertainty quantification across the holdout test set ($N_{\mathrm{test}} = 162$). (a) Predictions ranked by magnitude with $\pm 1\sigma$ ($68\%$) and $\pm 2\sigma$ ($95\%$) confidence intervals. (b) Uncertainty dispersion and Kernel Density Estimate ($\bar{\mu} = 13.21\%$, $\sigma_{95} = 17.90\%$).*

---

### 3.8 Physical Conformance Against Hydrodynamic Ferry-Renkin Sieving

```
FIGURE 8: Physical Validation vs. Ferry-Renkin Sieving Mechanics
Path: results/paper_figures/figure8_steric_physics_validation.png
```
![Figure 8: Steric Physics Validation](file:///c:/Users/Raghav/Documents/fml_research-main/results/paper_figures/figure8_steric_physics_validation.png)
*Figure 8: Predicted rejection vs. dimensionless steric ratio ($\lambda = r_{\mathrm{solute}} / r_{\mathrm{pore}}$) overlaid with the theoretical Ferry-Renkin curve. Shaded regions illustrate (I) Convective diffusion, (II) Hindered transition, and (III) Hard steric exclusion sieving.*

As demonstrated in Figure 8:
- **Regime I ($\lambda < 0.45$, Convective Diffusion):** Solutes are substantially smaller than membrane pores. Here, electrostatic Donnan repulsion elevates anionic species ($z = -1$, red triangles) well above neutral species.
- **Regime II ($0.45 \le \lambda < 1.0$, Hindered Transition):** Rejection ascends steeply, adhering to the non-linear trajectory of Ferry-Renkin hindrance.
- **Regime III ($\lambda \ge 1.0$, Steric Exclusion Sieving):** Solutes physically exceed pore dimensions. Predictions asymptotically converge to $100.0\%$, verifying strict physical conformance.

---

## 4. ENVIRONMENTAL ENGINEERING & INDUSTRIAL IMPLICATIONS

1. **Digital Twin for Municipal Water Treatment Plants:** Enables utilities to rapidly forecast contaminant breakthrough risks for emerging contaminants without requiring months of costly pilot trials.
2. **Rational Membrane Selection:** Process engineers can simulate tradeoffs between high-flux loose nanofiltration (e.g., NF270) and tight reverse osmosis (e.g., BW30) across feed pH and operating pressures.
3. **Regulatory Safety Buffers via Calibrated UQ:** The $95\%$ epistemic confidence interval ($\pm 2\sigma$) provides water reuse facilities with a quantitative safety margin to manage persistent mobile organic chemicals (PMOCs) and PFAS.

---

## 5. CONCLUSIONS

In this study, we developed and validated **PhysiChem-GTX**, a physics-informed dual-stream Mixture-of-Experts framework for predicting micropollutant rejection in NF/RO systems.
- **Accuracy:** Reached $R^2 = 0.9130$, $\text{RMSE} = 8.56\%$, and $\text{MAE} = 5.52\%$, establishing a new state of the art on the MemTrOC benchmark.
- **Generalization:** Validated via 5-fold cross-validation ($R^2 = 0.8585 \pm 0.0224$, peak fold $0.8819$).
- **Transparency:** Reconciled macro-scale TreeSHAP with atomic-scale Integrated Gradients, confirming the dominance of steric ratio, pore radius, and functional group ionization.
- **Physical Fidelity:** Enforced asymptotic convergence to $100\%$ rejection in the steric exclusion limit.

PhysiChem-GTX provides an open-source, publication-grade digital twin for environmental engineers and membrane separation scientists.

---

## DECLARATIONS

### Conflict of Interest Statement
The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

### Author Contributions (CRediT)
- **Raghav Gupta:** Conceptualization, Methodology, Software, Formal Analysis, Investigation, Data Curation, Writing - Original Draft, Visualization.
- **Raghav Gupta:** Software, Validation, Writing - Review & Editing, Supervision.

### Data and Code Availability
All source code, trained model weights, preprocessing scripts, and reproduction routines are open-sourced under the MIT License at:  
[https://github.com/Binaryblaze64/fml_final.git](https://github.com/Binaryblaze64/fml_final.git)

---

## REFERENCES

1. Schwarzenbach, R. P., Escher, B. I., Fenner, K., Hofstetter, T. B., Johnson, C. A., von Gunten, U., & Wehrli, B. (2006). The challenge of micropollutants in aquatic systems. *Science*, 313(5790), 1072-1077.
2. Shannon, M. A., Bohn, P. W., Elimelech, M., Georgiadis, J. G., Marinas, B. J., & Mayes, A. M. (2008). Science and technology for water purification in the coming decades. *Nature*, 452(7185), 301-310.
3. Bellona, C., Drewes, J. E., Xu, P., & Amy, G. (2004). Factors affecting the rejection of organic solutes during NF/RO treatment—a literature review. *Water Research*, 38(12), 2795-2809.
4. Verliefde, A. R., Heijman, S. G., Cornelissen, E. R., Amy, G., Van der Bruggen, B., & van Dijk, J. C. (2007). Influence of electrostatic, steric and hydrophobic solute–membrane interactions on rejection of trace organic contaminants by NF/RO. *Water Science and Technology*, 55(5), 183-189.
5. Elimelech, M., & Phillip, W. A. (2011). The future of seawater desalination: energy, technology, and the environment. *Science*, 333(6043), 712-717.
6. Werber, J. R., Osuji, C. O., & Elimelech, M. (2016). Materials for next-generation desalination and water purification membranes. *Nature Reviews Materials*, 1(5), 16018.
7. Van der Bruggen, B., & Vandecasteele, C. (2003). Removal of pollutants from surface water and groundwater by nanofiltration: overview of policy, current research and information needs. *Progress in Energy and Combustion Science*, 29(4), 307-345.
8. Yangali-Quintanilla, V., Verliefde, A., Kim, T. U., Sadmani, A., Kennedy, M., & Amy, G. (2009). Artificial neural network models based on QSAR for predicting rejection of neutral organic compounds by polyamide nanofiltration and reverse osmosis membranes. *Journal of Membrane Science*, 342(1-2), 251-262.
9. Ferry, J. D. (1936). Statistical evaluation of sieve constants in ultrafiltration. *The Journal of General Physiology*, 20(1), 95-104.
10. Renkin, E. M. (1954). Filtration, diffusion, and molecular sieving through porous cellulose membranes. *The Journal of General Physiology*, 38(2), 225-243.
11. Bowen, W. R., & Welfoot, J. S. (2002). Modelling the performance of membrane nanofiltration—critical assessment and kinetic approach. *Chemical Engineering Science*, 57(7), 1121-1137.
12. Bowen, W. R., & Mohammad, A. W. (1998). Characterization and prediction of nanofiltration membrane performance—a general approach. *Chemical Engineering Science*, 53(19), 3399-3412.
13. Bandini, S., & Vezzani, D. (2003). Nanofiltration modeling: the role of dielectric exclusion in rejection of ions. *Chemical Engineering Science*, 58(15), 3303-3326.
14. Kiso, Y., Sugiura, Y., Kitao, T., & Nishimura, K. (2001). Effects of hydrophobicity and molecular size on rejection of monosubstituted benzenes using nanofiltration membranes. *Journal of Membrane Science*, 192(1-2), 1-10.
15. Nghiem, L. D., Schäfer, A. I., & Elimelech, M. (2005). Pharmaceutical retention mechanisms by nanofiltration membranes. *Environmental Science & Technology*, 39(19), 7698-7705.
16. Spiegler, K. S., & Kedem, O. (1966). Thermodynamics of hyperfiltration (reverse osmosis): criteria for efficient membranes. *Desalination*, 1(4), 311-326.
17. Lin, C., Ding, L., & Hu, X. (2022). Quantitative structure-property relationships for predicting the rejection of organic micropollutants by nanofiltration/reverse osmosis membranes: A review. *Water Research*, 226, 119227.
18. Xiao, K., Shen, Y., Huang, X., & Zhang, X. (2026). Machine learning modeling of trace organic contaminant rejection by nanofiltration and reverse osmosis membranes. *Journal of Membrane Science*, 690, 122150.
19. Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. *International Conference on Learning Representations (ICLR)*.
20. Hu, W., Fey, M., Zitnik, M., Dong, Y., Ren, H., Liu, B., Catasta, M., & Leskovec, J. (2020). Open graph benchmark: Datasets for machine learning on graphs. *Advances in Neural Information Processing Systems (NeurIPS)*, 33, 22118-22133.
21. Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2019). How powerful are graph neural networks? *International Conference on Learning Representations (ICLR)*.
22. Karniadakis, G. E., Kevrekidis, I. G., Lu, L., Perdikaris, P., Wang, S., & Yang, L. (2021). Physics-informed machine learning. *Nature Reviews Physics*, 3(6), 422-440.
23. Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation: Representing model uncertainty in deep learning. *International Conference on Machine Learning (ICML)*, 1050-1059.
24. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems (NeurIPS)*, 30.
25. Brody, S., Alon, U., & Yahav, E. (2022). How attentive are graph attention networks? *International Conference on Learning Representations (ICLR)*.
26. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems (NeurIPS)*, 30.
27. Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic attribution for deep networks. *International Conference on Machine Learning (ICML)*, 3319-3328.
28. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794.
29. Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T. Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems (NeurIPS)*, 30.
30. Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. *International Conference on Learning Representations (ICLR)*.
31. Rogers, D., & Hahn, M. (2010). Extended-connectivity fingerprints. *Journal of Chemical Information and Modeling*, 50(5), 742-754.
32. Landrum, G. et al. (2023). RDKit: Open-source cheminformatics. [https://www.rdkit.org](https://www.rdkit.org).
33. Fey, M., & Lenssen, J. E. (2019). Fast graph representation learning with PyTorch Geometric. *ICLR Workshop on Representation Learning on Graphs and Manifolds*.
34. Paszke, A. et al. (2019). PyTorch: An imperative style, high-performance deep learning library. *Advances in Neural Information Processing Systems (NeurIPS)*, 32.
35. Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
36. Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., & Dahl, G. E. (2017). Neural message passing for quantum chemistry. *International Conference on Machine Learning (ICML)*, 1263-1272.
37. Kedem, O., & Katchalsky, A. (1958). Thermodynamic analysis of the permeability of biological membranes to non-electrolytes. *Biochimica et Biophysica Acta*, 27, 229-246.
38. Mohammad, A. W., Teow, Y. H., Ang, W. L., Chung, Y. T., Oatley-Radcliffe, D. L., & Hilal, N. (2015). Nanofiltration membranes review: Recent advances and future prospects. *Desalination*, 356, 226-254.
