# Research on Multi-objective Optimization of Low-Carbon Container Transportation Routing Using an Enhanced NSGA2 Algorithm

**Python** • **License** • **Paper** • **Data**

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)

An Enhanced NSGA-II Framework for Low-Carbon Container Routing on the Mombasa-Bujumbura Corridor.

---

## Overview

This repository contains the complete code, data, and analysis for the paper:

> **"Research on Multi-objective Optimization of Low Carbon Container Transportation Routing Using an Enhanced NSGA2 Algorithm"**
>
> *Yves Ndikuriyo, Yinggui Zhang, Dung Davou Fom*
>
> *School of Traffic and Transportation Engineering, Central South University, Changsha, China*
>
> *Journal of Cleaner Production (Under Review)*

### What is LC-CTRP?

The **Low-Carbon Container Transportation Routing Problem (LC-CTRP)** optimizes multimodal freight routes, balancing three conflicting objectives:

| Objective | Description | Unit |
|-----------|-------------|------|
| **Cost** | Total transportation + transshipment + carbon costs | USD |
| **Emissions** | CO₂ emissions from road, rail, and waterway transport | kg CO₂ |
| **Time** | Total transit time including border delays and transshipment | hours |

### What is En-NSGA-II?

The **Enhanced NSGA-II (En-NSGA-II)** is a hybrid multi-objective optimization framework that integrates:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           En-NSGA-II Framework                               │
├─────────────────────────────────────────────────────────────────────────────┤
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│   │     PPO      │───▶│     GNN      │───▶│   NSGA-II    │                  │
│   │  Adaptive    │    │   Surrogate  │    │   Pareto     │                  │
│   │ Initialization│    │  Evaluation  │    │    Search    │                  │
│   └──────────────┘    └──────────────┘    └──────────────┘                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

| Component | Description | Key Metric |
|-----------|-------------|------------|
| **PPO** | Learns adaptive initialization policies | 86.2% cost reduction vs random |
| **GNN** | Fast surrogate evaluation of emissions/cost | R² = 0.935/0.954, 5-6× speedup |
| **NSGA-II** | Pareto frontier search with enhancements | 68% lower emissions vs standard |

---

## Key Findings

| Finding | Value | Implication |
|---------|-------|-------------|
| Total road CO₂ emissions | 3.66 million tonnes | Baseline for corridor |
| En-NSGA-II baseline emissions | 474.9 kg CO₂ | 68% lower than NSGA-II |
| En-NSGA-II baseline cost | $346.15 | Only 10% premium |
| Optimal carbon price | $55/ton CO₂ | Policy target |
| Minimum emissions achieved | 336.0 kg CO₂ | 29% reduction |
| Total decarbonization potential | 1.79 million tonnes | 48.8% of baseline |
| Most cost-effective measure | Driver training | $10/tonne |

---

## Algorithm Performance Summary (11 Benchmark Problems)

| Algorithm | HV Wins | IGD Wins | Spread Wins | Runtime Wins | Win Rate |
|-----------|---------|----------|-------------|--------------|----------|
| **En-NSGA-II** | 10 (90.9%) | 8 (72.7%) | 7 (63.6%) | 0 | **56.8%** |
| NSGA-II | 1 (9.1%) | 3 (27.3%) | 3 (27.3%) | 0 | 15.9% |
| SPEA2 | 0 | 0 | 1 (9.1%) | 0 | 2.3% |
| MOEA/D | 0 | 0 | 0 | 11 (100%) | 25.0% |

### HV Performance (Higher is better)

| Problem | En-NSGA-II | NSGA-II | SPEA2 | MOEA/D |
|---------|------------|---------|-------|--------|
| ZDT1 | **99.33** | 79.98 | 65.58 | 63.52 |
| ZDT2 | **99.66** | 82.57 | 79.49 | 94.52 |
| ZDT3 | **100.69** | 81.36 | 75.62 | 65.21 |
| ZDT4 | 9,706.11 | **9,769.00** | 9,640.21 | 8,745.60 |
| ZDT6 | **100.51** | 36.55 | 36.41 | 35.21 |
| DTLZ2 | **2.68** | 2.43 | 1.04 | 2.54 |
| WFG1 | **27.00** | 25.52 | 17.98 | 25.15 |
| WFG2 | **27.00** | 26.96 | 25.19 | 26.94 |

### IGD Performance (Lower is better)

| Problem | En-NSGA-II | NSGA-II | SPEA2 | MOEA/D |
|---------|------------|---------|-------|--------|
| ZDT1 | **0.011** | 0.934 | 2.566 | 1.551 |
| ZDT2 | **0.011** | 1.415 | 2.361 | 1.486 |
| ZDT3 | **0.218** | 0.409 | 1.573 | 0.618 |
| ZDT6 | **0.008** | 5.107 | 6.341 | 5.396 |
| DTLZ2 | **0.096** | 0.190 | 0.467 | 0.182 |

---

## Repository Structure

```
En-NSGA-II-4-LC-CTRP/
│
├── README.md                                    # This file
├── LICENSE                                      # MIT License
├── requirements.txt                             # Python dependencies
│
├── data/
│   ├── raw/                                     # Original datasets
│   └── processed/                               # Cleaned data
│       ├── algorithm_results.csv
│       ├── carbon_price_sensitivity.csv
│       ├── corridor_emissions.csv
│       ├── border_crossing_summary.csv
│       └── lake_ports_cargo.csv
│
├── scripts/                                     # Execution scripts
│   ├── Statistical Analysis.py
│   ├── GNN and PPO validation.py
│   ├── En-NSGA-II vs Exact Method.py
│   ├── COMPREHENSIVE BENCHMARK ANALYSIS 22 Instances.py
│   └── Case study LC-CTRP ANALYSIS.py
│
├── results/
│   ├── figures/                                 # Generated figures
│   │   ├── hv_comparison_boxplot.png
│   │   ├── igd_comparison_boxplot.png
│   │   ├── hv_statistical_significance.png
│   │   ├── igd_statistical_significance.png
│   │   ├── per_objective_accuracy.png
│   │   ├── accuracy_tradeoff_surface.png
│   │   ├── error_distribution_analysis.png
│   │   ├── computational_breakdown.png
│   │   ├── systematic_bias_analysis.png
│   │   ├── pareto_comparison_3d.png
│   │   ├── convergence_*.png
│   │   ├── instance_comparison_summary.png
│   │   └── analysis_plots/pareto_fronts/
│   │       ├── ZDT1_pareto_front.png
│   │       ├── ZDT2_pareto_front.png
│   │       ├── ZDT3_pareto_front.png
│   │       ├── ZDT4_pareto_front.png
│   │       ├── ZDT6_pareto_front.png
│   │       ├── DTLZ1_pareto_front.png
│   │       ├── DTLZ2_pareto_front.png
│   │       ├── DTLZ3_pareto_front.png
│   │       ├── DTLZ4_pareto_front.png
│   │       ├── WFG1_pareto_front.png
│   │       ├── WFG2_pareto_front.png
│   │       └── ZDT_suite_comparison.png
│   │
│   └── tables/                                  # Summary tables
│       ├── complete_benchmark_results.csv
│       ├── complete_summary_results.csv
│       ├── statistical_significance.csv
│       ├── per_objective_metrics.csv
│       ├── accuracy_tradeoff_analysis.csv
│       ├── computational_components.csv
│       ├── error_distribution_analysis.csv
│       ├── detailed_validation_results.csv
│       ├── validation_summary_table.csv
│       └── lc_ctrp_complete_results.json
│
└── docs/                                        # Documentation
    ├── supplementary_material.pdf
    └── data_documentation.md
```

---

## Figures

All figures are generated in `results/figures/` and organized by analysis type.

### 1. Benchmark Analysis Figures (COMPREHENSIVE BENCHMARK ANALYSIS)

| Figure | Description | Location |
|--------|-------------|----------|
| `hv_comparison_boxplot.png` | Hypervolume distribution across 5 runs for 11 benchmark problems | `results/figures/` |
| `igd_comparison_boxplot.png` | IGD distribution across 5 runs for 11 benchmark problems | `results/figures/` |
| `hv_statistical_significance.png` | Statistical significance of HV differences (p-values) | `results/figures/` |
| `igd_statistical_significance.png` | Statistical significance of IGD differences (p-values) | `results/figures/` |

### 2. ZDT Pareto Front Comparisons

| Figure | Description | Problem Characteristics |
|--------|-------------|------------------------|
| `ZDT1_pareto_front.png` | Convex Pareto front comparison | Convex |
| `ZDT2_pareto_front.png` | Concave Pareto front comparison | Concave |
| `ZDT3_pareto_front.png` | Disconnected Pareto front | Disconnected |
| `ZDT4_pareto_front.png` | Many local optima | Multi-modal |
| `ZDT6_pareto_front.png` | Non-uniform density | Non-uniform |
| `ZDT_suite_comparison.png` | Combined 2×3 subplot of all ZDT problems | All ZDT |

### 3. DTLZ Pareto Front Comparisons (3D)

| Figure | Description | Location |
|--------|-------------|----------|
| `DTLZ1_pareto_front.png` | Linear Pareto front (3D) | `results/figures/analysis_plots/pareto_fronts/` |
| `DTLZ2_pareto_front.png` | Concave Pareto front (3D) | `results/figures/analysis_plots/pareto_fronts/` |
| `DTLZ3_pareto_front.png` | Multi-modal with many local optima | `results/figures/analysis_plots/pareto_fronts/` |
| `DTLZ4_pareto_front.png` | Biased density distribution | `results/figures/analysis_plots/pareto_fronts/` |
| `*_2D_projections.png` | 2D projections (f1-f2, f1-f3, f2-f3) | `results/figures/analysis_plots/pareto_fronts/` |

### 4. WFG Pareto Front Comparisons (3D)

| Figure | Description | Location |
|--------|-------------|----------|
| `WFG1_pareto_front.png` | Mixed convex/concave, biased | `results/figures/analysis_plots/pareto_fronts/` |
| `WFG2_pareto_front.png` | Disconnected, constrained | `results/figures/analysis_plots/pareto_fronts/` |

### 5. GNN & PPO Validation Figures

| Figure | Description | Source |
|--------|-------------|--------|
| `per_objective_accuracy.png` | R² scatter plots for cost, emissions, time predictions | GNN validation |
| `accuracy_tradeoff_surface.png` | 3D trade-off surface (accuracy vs evaluation reduction vs HV improvement) | GNN validation |
| `error_distribution_analysis.png` | Error distributions, biases, skewness, over/under prediction | GNN validation |
| `computational_breakdown.png` | Time per evaluation: exact vs GNN components | GNN validation |
| `systematic_bias_analysis.png` | Systematic biases across objective magnitudes | GNN validation |
| `pareto_comparison_3d.png` | Pareto front comparison: exact vs surrogate | GNN validation |

### 6. En-NSGA-II vs Exact Method Figures

| Figure | Description | Location |
|--------|-------------|----------|
| `convergence_S1_8nodes.png` | Convergence plots for S1 (8 nodes) | `results/figures/` |
| `convergence_S2_12nodes.png` | Convergence plots for S2 (12 nodes) | `results/figures/` |
| `convergence_S3_15nodes.png` | Convergence plots for S3 (15 nodes) | `results/figures/` |
| `convergence_S4_18nodes.png` | Convergence plots for S4 (18 nodes) | `results/figures/` |
| `instance_comparison_summary.png` | Cross-instance comparison: optimality gaps, success rates, runtime, scalability | `results/figures/` |

Each convergence plot includes:
- **Top-left**: Individual run convergence (10 runs)
- **Top-right**: Mean convergence with ±1σ confidence interval
- **Bottom-left**: Success rate evolution (≤1% optimality gap)
- **Bottom-right**: Final cost distribution histogram with normal fit

### 7. LC-CTRP Case Study Figures

| Figure | Description | Source |
|--------|-------------|--------|
| `Emissions Performance vs Carbon Price` | Algorithm emissions across 10 price levels | Carbon sensitivity |
| `Cost Performance vs Carbon Price` | Algorithm costs across 10 price levels | Carbon sensitivity |
| `Modal Shift Response to Carbon Pricing` | Rail mode share vs carbon price with zone shading | Carbon sensitivity |
| `Algorithm Performance Summary Table` | Emissions/cost table for all algorithms at all prices | Summary |
| `Algorithm Convergence Curves` | Cost convergence over generations | Case study |
| `Cost-Emissions Trade-off` | Pareto frontier with ideal point | Case study |
| `Border Crossing Distributions` | VaR/CVaR at 95% for Busia, Malaba, Katuna | Data analysis |
| `Algorithm Performance Radar` | Multi-dimensional comparison (6 metrics) | Summary |
| `Route Emissions Bar Chart` | Top 8 route segments by CO₂ emissions | Data analysis |
| `Marginal Abatement Cost Curve` | MAC curve with 5 abatement measures | Policy analysis |

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/YvesNDIKURIYO-2022/En-NSGA-II-4-LC-CTRP.git
cd En-NSGA-II-4-LC-CTRP
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run analyses

```bash
# Comprehensive benchmark (11 problems, 5 runs each)
python "scripts/COMPREHENSIVE BENCHMARK ANALYSIS 22 Instances.py"

# GNN and PPO validation
python "scripts/GNN and PPO validation.py"

# En-NSGA-II vs Exact Method
python "scripts/En-NSGA-II vs Exact Method.py"

# LC-CTRP case study with carbon price sensitivity
python "scripts/Case study LC-CTRP ANALYSIS - ALL 5 ALGORITHMS + CARBON PRICE SENSITIVITY.py"
```

---

## Requirements

```txt
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
```

---

## Data Sources

| Source | Data Type | Link |
|--------|-----------|------|
| NCTTCA Transport Observatory | Road emissions, pollutants, border times | https://top.ttcanc.org |
| EDGAR Database | Country-level GHG emissions | https://edgar.jrc.ec.europa.eu |
| Kenya Ministry of Transport | Rail emissions | https://www.transport.go.ke |
| Tanzania Ports Authority | Lake ports cargo | https://www.ports.go.tz |
| Kenya Ports Authority | Cargo volumes | https://www.kpa.co.ke |

---

## Citation

```bibtex
@article{Ndikuriyo2026,
  title = {Research on Multi-objective Optimization of Low Carbon Container Transportation Routing Using an Enhanced NSGA2 Algorithm},
  author = {Ndikuriyo, Yves and Zhang, Yinggui and Fom, Dung Davou},
  journal = {Journal of Cleaner Production},
  year = {2026},
  note = {Under review}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) file.

---

## Contact

**Yves Ndikuriyo**  
PhD Candidate, Central South University, Changsha, China  
📧 yvesndikuriyo@csu.edu.cn  
🔗 [ORCID: 0009-0006-9324-7265](https://orcid.org/0009-0006-9324-7265)

**Professor Yinggui Zhang**  
📧 ygzhang@csu.edu.cn  
🔗 [ORCID: 0000-0002-5790-0638](https://orcid.org/0000-0002-5790-0638)

---

## Acknowledgments

- NCTTCA for Transport Observatory data
- Tanzania Ports Authority for lake ports data
- Kenya Ministry of Transport for rail emissions data
- EDGAR, IEA, Worldometer, Climate Change Tracker for validation data

---

<div align="center">
  <sub>Built with Python • Analysis for Journal of Cleaner Production • Last updated: May 2026</sub>
</div>
