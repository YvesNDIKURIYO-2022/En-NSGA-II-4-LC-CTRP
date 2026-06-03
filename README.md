# Multi-Objective Optimization of Low-Carbon Container Routing Using PA-NSGA-II

**Python** • **License** • **Paper** • **Data**

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)

A Parameter-Adaptive NSGA-II Framework for Low-Carbon Container Routing on the Mombasa-Bujumbura Corridor.

---

## Overview

This repository contains the complete code, data, and analysis for the paper:

> **"Multi-Objective Optimization of Low-Carbon Container Routing Using Parameter-Adaptive NSGA-II"**
>
> *Yves Ndikuriyo, Yinggui Zhang, Dung Davou Fom*
>
> *School of Traffic and Transportation Engineering, Central South University, Changsha, China*
>
> *Transportation Research Part D: Transport and Environment (Under Review)*

### What is LC-CTRP?

The **Low-Carbon Container Transportation Routing Problem (LC-CTRP)** optimizes multimodal freight routes, balancing three conflicting objectives:

| Objective | Description | Unit |
|-----------|-------------|------|
| **Cost** | Total transportation + transshipment + carbon costs | USD |
| **Emissions** | CO₂ emissions from road, rail, and waterway transport | kg CO₂ |
| **Time** | Total transit time including border delays and transshipment | hours |

### What is PA-NSGA-II?

The **Parameter-Adaptive NSGA-II (PA-NSGA-II)** is a hybrid multi-objective optimization framework that integrates:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PA-NSGA-II Framework                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│   │  Adaptive    │───▶│   Size-      │───▶│   NSGA-II    │                  │
│   │     PPO      │    │   Adaptive   │    │   Pareto     │                  │
│   │  Initialization│   │   Config.   │    │    Search    │                  │
│   └──────────────┘    └──────────────┘    └──────────────┘                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

| Component | Description | Key Metric |
|-----------|-------------|------------|
| **Adaptive-PPO** | Learns adaptive initialization policies | 53.2% lower emissions vs random |
| **Size-Adaptive Config.** | Population size, generations, bias factors scale with n | 37%→57% advantage (20→200 customers) |
| **PA-NSGA-II** | Pareto frontier search with adaptive operators | 45.4% lower emissions vs NSGA-II/SPEA2 |

---

## Key Findings

| Finding | Value | Implication |
|---------|-------|-------------|
| PA-NSGA-II win rate (24 Prins instances) | 83.3% (20/24) | Algorithmic superiority |
| Mean emissions reduction vs NSGA-II/SPEA2 | 45.4% | Significant decarbonization |
| Adaptive-PPO vs random initialization | 53.2% lower (72.7 vs 155.3 kg) | Initialization matters |
| Zero variance across 3 runs | 24/24 instances | Deterministic reliability |
| Baseline corridor emissions (PA-NSGA-II/SPEA2) | 392.9 kg CO₂ | Global optimum identified |
| NSGA-II baseline emissions | 422.3 kg CO₂ | 7.0% higher than optimum |
| Optimal carbon price | $150/ton CO₂ | Policy target |
| Minimum emissions achieved | 346.2 kg CO₂ | 24.7% reduction from $50/ton |
| Wet season performance (PA-NSGA-II) | -12.5% emissions | Adaptive modal shift |
| Wet season performance (SPEA2) | +30.1% emissions | Contrast in robustness |

---

## Algorithm Performance Summary (24 Prins Instances)

| Metric | PA-NSGA-II | NSGA-II | SPEA2 |
|--------|------------|---------|-------|
| Mean emissions (kg) | **67.7** | 123.9 | 121.2 |
| Mean cost (USD) | 302,560 | 164,744 | **157,691** |
| Win rate (% instances) | **83.3%** | — | — |
| Mean Pareto size (large instances) | **3–5** | 1 | 1 |
| Computational time (s) | 6.60 | 5.99 | **5.67** |

### Performance by Instance Size

| Instance Size | PA-NSGA-II Advantage |
|---------------|---------------------|
| 20 customers | 28% |
| 50 customers | 35% |
| 100 customers | 48% |
| 200 customers | 55% |

---

## Carbon Price Sensitivity (Mombasa–Bujumbura Corridor)

| Carbon Price | PA-NSGA-II Emissions | NSGA-II Emissions | SPEA2 Emissions |
|--------------|---------------------|-------------------|-----------------|
| $50/ton | 459.7 kg | 660.9 kg | 1,211.9 kg |
| $100/ton | 401.7 kg | 874.4 kg | 1,026.0 kg |
| $150/ton | **346.2 kg** | 653.9 kg | 1,081.3 kg |

**Key insight:** Carbon pricing alone cannot induce modal shift. The rail share remains constant across $0–150/ton CO₂ due to transshipment penalties (12 hours, $37.50) and border delays (12–24 hours, $250).

---

## Repository Structure

```
PA-NSGA-II-4-LC-CTRP/
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
│   ├── Initialization Validation.py
│   ├── PA-NSGA-II vs Baseline.py
│   ├── COMPREHENSIVE BENCHMARK ANALYSIS.py
│   └── Case study LC-CTRP ANALYSIS.py
│
├── results/
│   ├── figures/                                 # Generated figures
│   │   ├── init_size_performance.png
│   │   ├── init_boxplot.png
│   │   ├── init_radar.png
│   │   ├── carbon_sensitivity.png
│   │   ├── seasonal_sensitivity.png
│   │   ├── corridor_map.png
│   │   └── convergence_plots/
│   │
│   └── tables/                                  # Summary tables
│       ├── complete_benchmark_results.csv
│       ├── complete_summary_results.csv
│       ├── statistical_significance.csv
│       ├── path_selection_results.csv
│       └── lc_ctrp_complete_results.json
│
└── docs/                                        # Documentation
    ├── supplementary_material.pdf
    └── data_documentation.md
```

---

## Figures

All figures are generated in `results/figures/` and organized by analysis type.

### 1. Initialization Validation Figures

| Figure | Description |
|--------|-------------|
| `init_size_performance.png` | Performance stratified by instance size (37%→57% advantage) |
| `init_boxplot.png` | Distribution of best emissions across five methods |
| `init_radar.png` | Multi-criteria radar chart (4 metrics) |

### 2. Benchmark Analysis Figures

| Figure | Description |
|--------|-------------|
| `hv_comparison_boxplot.png` | Hypervolume distribution across 24 Prins instances |
| `igd_comparison_boxplot.png` | IGD distribution across 24 Prins instances |
| `hv_statistical_significance.png` | Statistical significance of HV differences (p-values) |
| `igd_statistical_significance.png` | Statistical significance of IGD differences (p-values) |

### 3. Case Study Figures

| Figure | Description |
|--------|-------------|
| `corridor_map.png` | East African transport network map (19 nodes, 34 arcs) |
| `carbon_sensitivity.png` | Emissions performance across carbon prices ($0–150/ton) |
| `seasonal_sensitivity.png` | Performance across dry, wet, peak seasons |
| `border_var.png` | Border crossing delay distributions (VaR/CVaR at 95%) |
| `path_selection.png` | Algorithm path consistency across 7 price levels |

### 4. Convergence Analysis Figures

| Figure | Description |
|--------|-------------|
| `convergence_S1_8nodes.png` | Convergence for 8-node network |
| `convergence_S2_12nodes.png` | Convergence for 12-node network |
| `convergence_S3_15nodes.png` | Convergence for 15-node network |
| `convergence_S4_18nodes.png` | Convergence for 18-node network |

Each convergence plot includes:
- Individual run convergence (10 runs)
- Mean convergence with ±1σ confidence interval
- Success rate evolution (≤1% optimality gap)
- Final cost distribution histogram

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/YvesNDIKURIYO-2022/PA-NSGA-II-4-LC-CTRP.git
cd PA-NSGA-II-4-LC-CTRP
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run analyses

```bash
# Comprehensive benchmark (24 Prins instances)
python "scripts/COMPREHENSIVE BENCHMARK ANALYSIS.py"

# Initialization validation (Adaptive-PPO vs 4 baselines)
python "scripts/Initialization Validation.py"

# PA-NSGA-II vs NSGA-II/SPEA2
python "scripts/PA-NSGA-II vs Baseline.py"

# LC-CTRP case study with carbon price sensitivity (7 levels)
python "scripts/Case study LC-CTRP ANALYSIS.py"
```

---

## Requirements

```txt
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
pymoo>=0.6.0
stable-baselines3>=2.0.0
networkx>=2.6.0
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
| Prins Benchmarks | VRP instances (24 instances, 20–200 customers) | Augerat et al. (1995), Christofides et al. (1969) |

---

## Citation

```bibtex
@article{Ndikuriyo2026,
  title = {Multi-Objective Optimization of Low-Carbon Container Routing Using Parameter-Adaptive NSGA-II},
  author = {Ndikuriyo, Yves and Zhang, Yinggui and Fom, Dung Davou},
  journal = {Transportation Research Part D: Transport and Environment},
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
- Prins, Augerat, and Christofides for benchmark instances
- School of Traffic and Transportation Engineering, Central South University

---

<div align="center">
  <sub>Built with Python • Analysis for Transportation Research Part D • Last updated: June 2026</sub>
</div>
l | Transportation Research Part D |
| Benchmark count | 24 Prins instances (explicitly stated) |
