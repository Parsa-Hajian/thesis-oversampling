# Circular Oversampling for Imbalanced Classification

**Master's Thesis, Università degli Studi di Milano, 2025**

*Circular Oversampling, Seed Selection, and a Controlled Degradation Study*

**Author:** Parsa Hajiannejad

---

## Overview

This thesis proposes three circle-based oversampling algorithms for imbalanced binary classification, a geometry-preserving seed selection pipeline, and a controlled causal study of the imbalance–performance relationship.

**Key contributions:**
- **Geometry-preserving seed selection** using the Bhattacharyya Coefficient, AGTP, JSD, and smoothness regularization
- **GVM-CO** : Gravity-biased Von Mises Circular Oversampling
- **LRE-CO** : Local Region Estimation Circular Oversampling (Voronoi-constrained)
- **LS-CO** : Layered Segmental Circular Oversampling
- **Controlled degradation-recovery protocol** for causal imbalance analysis

## Repository Structure

```
thesis-oversampling/
├── thesis/                          # LaTeX thesis document
│   ├── main.tex                     # Main thesis file
│   ├── main.pdf                     # Compiled thesis
│   ├── references.bib               # Bibliography
│   ├── chapters/                    # Thesis chapters (5 parts, 15 chapters)
│   ├── presentations/               # Beamer slides (per-part .tex + compiled .pdf/.pptx)
│   └── standalone/                  # Standalone compilable chapters
│
├── src/                             # Source code
│   ├── circular_oversampling/       # Oversampling algorithms
│   │   ├── core/                    # GVM-CO, LRE-CO, LS-CO implementations
│   │   ├── seed_selection/          # Geometry-preserving seed selector
│   │   ├── evaluation/              # Classifiers, cross-validation, metrics, statistical tests
│   │   ├── datasets/                # Dataset loader and registry
│   │   ├── preprocessing/           # Clustering, denoising, PCA projection
│   │   ├── comparison/              # Baseline method wrappers
│   │   ├── visualization/           # Plotting utilities
│   │   └── utils/                   # Geometry and helper functions
│   └── causality/                   # Degradation-recovery study
│       ├── data_utils/              # Balanced loader, incremental unbalancer
│       ├── experiments/             # Degradation and recovery runners
│       ├── oversamplers/            # Oversampler wrappers for the study
│       └── visualization/           # Sweep plot generation
│
├── experiments/                     # Experiment scripts
│   ├── circular_oversampling/       # Main benchmark, ablation, figure/table generation
│   └── causality/                   # Causal study figure generation
│
├── configs/                         # Experiment configuration (YAML)
├── tests/                           # Unit tests
├── results/                         # Experiment output (CSV tables)
├── figures/                         # Generated figures (PDF)
│   ├── circular_oversampling/       # Algorithm step-by-step, CD diagrams, heatmaps
│   └── causality/                   # Degradation sweep curves
│
└── requirements.txt                 # Python dependencies
```

## Setup

```bash
pip install -r requirements.txt
```

Dependencies: `numpy`, `scipy`, `scikit-learn`, `imbalanced-learn`, `matplotlib`, `pandas`, `pyyaml`

## Running Experiments

**Full benchmark (14 datasets, 13 methods, 5 classifiers):**
```bash
python experiments/circular_oversampling/run_all.py
```

**Single dataset experiment:**
```bash
python experiments/circular_oversampling/run_single.py --dataset ecoli1
```

**Generate thesis figures and tables:**
```bash
python experiments/circular_oversampling/generate_figures.py
python experiments/circular_oversampling/generate_thesis_tables.py
```

**Causality study:**
```bash
python -m src.causality.experiments.run_degradation
python -m src.causality.experiments.run_recovery
python experiments/causality/generate_figures.py
```

## Python Package

The algorithms are also available as a standalone pip-installable package:

```bash
pip install circover
```

See [circover on PyPI](https://pypi.org/project/circover/) or [GitHub](https://github.com/Parsa-Hajian/circover).

## Thesis Structure

| Part | Chapters | Topic |
|------|----------|-------|
| I    | 1–3      | Foundations, background, literature review |
| II   | 4        | Geometry-preserving seed selection (BC + AGTP + JSD + Z) |
| III  | 5–9      | GVM-CO, LRE-CO, LS-CO — algorithms, experiments, results, ablation |
| IV   | 10–14    | Controlled degradation-recovery causal study |
| V    | 15       | Conclusion and future work |

## License

This work is submitted as a Master's thesis. Code is available for academic use.
