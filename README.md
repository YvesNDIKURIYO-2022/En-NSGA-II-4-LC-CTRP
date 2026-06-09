# Multi-Objective Optimization of Low-Carbon Container Routing Using PA-NSGA-II

**Python** • **MIT License** • **Paper** • **Data**

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)

A Parameter-Adaptive NSGA-II Framework for Low-Carbon Container Routing on the Mombasa–Bujumbura Corridor.

---

## Overview

This repository contains the complete code, data, and analysis for the paper:

> **"A Parameter-Adaptive Multi-Objective Evolutionary Algorithm for Low-Carbon Container Routing under Uncertainty"**
>
> *Yves Ndikuriyo, Yinggui Zhang, Dung Davou Fom*
>
> *School of Traffic and Transportation Engineering, Central South University, Changsha, China*
>
> *Simulation Modelling Practice and Theory (Under Review)*

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
| Adaptive-PPO vs random initialization | 53.2% lower | Initialization matters |
| Zero variance across 50 runs | 24/24 instances | Deterministic reliability |
| Baseline corridor emissions (PA-NSGA-II) | 356 kg CO₂ | Global optimum identified |
| Baseline corridor emissions (NSGA-II) | 576 kg CO₂ | 38.2% higher than PA-NSGA-II |
| Optimal carbon price | $150/ton CO₂ | Policy target |
| Minimum emissions achieved | 346 kg CO₂ | 24.7% reduction from $50/ton |
| Cohen's d (effect size) | 0.55 (medium) | Statistically significant improvement |
| Coefficient of variation (CV) | 2.584 | Most consistent algorithm |

---

## Algorithm Performance Summary (24 Prins Instances)

| Metric | PA-NSGA-II | NSGA-II | SPEA2 | NSGA-III |
|--------|------------|---------|-------|----------|
| Mean emissions (kg) | **67.7** | 123.9 | 121.2 | 121.8 |
| Mean cost (USD) | 302,560 | 164,744 | **157,691** | 158,235 |
| Mean hypervolume | **1.85×10¹¹** | 1.24×10⁹ | 2.36×10⁹ | 1.24×10⁹ |
| Win rate (% instances) | **83.3%** | — | — | — |
| Coefficient of variation (CV) | **2.58** | 12.69 | 13.52 | 9.27 |
| Cohen's d vs PA-NSGA-II | — | 0.55 | 0.55 | 0.55 |

### Performance by Instance Size

| Instance Size | PA-NSGA-II Advantage |
|---------------|---------------------|
| 20 customers | 37% |
| 50 customers | 45% |
| 100 customers | 51% |
| 200 customers | 57% |

---

## Carbon Price Sensitivity (Mombasa–Bujumbura Corridor)

| Carbon Price | PA-NSGA-II Emissions | NSGA-II Emissions | SPEA2 Emissions | NSGA-III Emissions |
|--------------|---------------------|-------------------|-----------------|-------------------|
| $50/ton | 356 kg | 576 kg | 992 kg | 720 kg |
| $100/ton | 325 kg | 664 kg | 1,088 kg | 1,049 kg |
| $150/ton | **346 kg** | 772 kg | 930 kg | 588 kg |

**Key insight:** Carbon pricing alone cannot induce modal shift. The rail share remains constant across $0–150/ton CO₂ due to transshipment penalties (12 hours, $37.50) and border delays (12–24 hours, $250).

---

## Seasonal Sensitivity (Mombasa–Bujumbura Corridor)

| Season | PA-NSGA-II | NSGA-II | SPEA2 | NSGA-III |
|--------|------------|---------|-------|----------|
| Dry | 356 kg | 576 kg | 992 kg | 720 kg |
| Wet | 471 kg | 743 kg | 1,248 kg | 721 kg |
| Peak | 471 kg | 855 kg | 1,033 kg | 703 kg |

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
│       ├── complete_pansgaII_validation_results.csv
│       ├── initialization_comparison_results.csv
│       ├── validation_report.json
│       └── corridor_network_data.csv
│
├── scripts/                                     # Execution scripts
│   ├── COMPREHENSIVE BENCHMARK ANALYSIS.py
│   ├── Initialization Validation.py
│   ├── PA-NSGA-II vs Baseline.py
│   ├── Case study LC-CTRP ANALYSIS.py
│   └── Statistical Analysis.py
│
├── results/
│   ├── plots/                                   # Generated figures
│   │   ├── variance_analysis.pdf
│   │   ├── success_rate.pdf
│   │   ├── ablation_study.pdf
│   │   ├── Fig1_Baseline_Performance.pdf
│   │   ├── Fig2_Carbon_Tax_Sensitivity.pdf
│   │   ├── Fig3_Seasonal_Sensitivity.pdf
│   │   ├── Fig4_Algorithm_Ranking.pdf
│   │   ├── Fig5_Improvement_Chart.pdf
│   │   ├── Fig6_Summary_Dashboard.pdf
│   │   ├── Fig7_Statistical_Significance.pdf
│   │   ├── Fig8_Convergence_Comparison.pdf
│   │   ├── Fig9_Cost_Emissions_Tradeoff.pdf
│   │   ├── Fig10_Cost_Time_Tradeoff.pdf
│   │   ├── Fig11_Emissions_Time_Tradeoff.pdf
│   │   ├── Fig12_Pareto_Front_Comparison.pdf
│   │   └── Fig13_3D_Objective_Space.pdf
│   │
│   ├── statistics/                              # Statistical outputs
│   │   └── validation_report.json
│   │
│   ├── pareto_fronts/                           # Pareto front data
│   │
│   ├── runs_data/                               # Per-run data
│   │
│   └── ablation/                                # Ablation study results
│
└── docs/                                        # Documentation
    ├── supplementary_material.pdf
    └── data_documentation.md
```

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
# Comprehensive benchmark (24 Prins instances, 50 runs per algorithm)
python "scripts/COMPREHENSIVE BENCHMARK ANALYSIS.py"

# Initialization validation (Adaptive-PPO vs 4 baselines)
python "scripts/Initialization Validation.py"

# PA-NSGA-II vs NSGA-II/SPEA2/NSGA-III
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
| Prins Benchmarks | VRP instances (24 instances, 20–200 customers) | Prins et al. (2007) |

---

## Citation

```bibtex
@article{Ndikuriyo2026,
  title = {A Parameter-Adaptive Multi-Objective Evolutionary Algorithm for Low-Carbon Container Routing under Uncertainty},
  author = {Ndikuriyo, Yves and Zhang, Yinggui and Fom, Dung Davou},
  journal = {Simulation Modelling Practice and Theory},
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

**Professor Yinggui Zhang** (Corresponding Author)  
📧 ygzhang@csu.edu.cn  
🔗 [ORCID: 0000-0002-5790-0638](https://orcid.org/0000-0002-5790-0638)

**Dung Davou Fom**  
📧 dungfom1@csu.edu.cn  
🔗 [ORCID: 0009-0001-8688-813X](https://orcid.org/0009-0001-8688-813X)

---

## Acknowledgments

- NCTTCA for Transport Observatory data
- Tanzania Ports Authority for lake ports data
- Kenya Ministry of Transport for rail emissions data
- Prins, Augerat, and Christofides for benchmark instances
- School of Traffic and Transportation Engineering, Central South University

---

<div align="center">
  <sub>Built with Python • Analysis for Simulation Modelling Practice and Theory • Last updated: June 2026</sub>
</div>
