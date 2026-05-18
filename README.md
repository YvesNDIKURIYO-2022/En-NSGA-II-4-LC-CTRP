# Research on Multi-objective Optimization of Low-Carbon Container Transportation Routing Using an Enhanced NSGA2 Algorithm

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Paper](https://img.shields.io/badge/Paper-Journal%20of%20Cleaner%20Production-red.svg)
![Data](https://img.shields.io/badge/Data-NCTTCA%20Observatory-orange.svg)

**An Enhanced NSGA-II Framework for Low-Carbon Container Routing**

[Overview](#overview) • [Key Findings](#key-findings) • [Repository Structure](#repository-structure) • [Quick Start](#quick-start) • [Data Sources](#data-sources) • [Results](#results) • [Citation](#citation)

</div>

---

## Overview

This repository contains the complete code, data, and analysis for the paper:

> **"Research on Multi-objective Optimization of Low Carbon Container Transportation Routing Using an Enhanced NSGA2 Algorithm"**  
> *Yves Ndikuriyo, Yinggui Zhang, Dung Davou Fom*  
> *School of Traffic and Transportation Engineering, Central South University, Changsha, China*  
> *Journal of Cleaner Production (Under Review)*

### What is LC-CTRP?

The Low-Carbon Container Transportation Routing Problem (LC-CTRP) optimizes multimodal freight routes, balancing three conflicting objectives:

| Objective | Description | Unit |
|-----------|-------------|------|
| **Cost** | Total transportation + transshipment + carbon costs | USD |
| **Emissions** | CO₂ emissions from road, rail, and waterway transport | kg CO₂ |
| **Time** | Total transit time including border delays and transshipment | hours |

### What is En-NSGA-II?

The **Enhanced NSGA-II (En-NSGA-II)** is a hybrid multi-objective optimization framework that integrates:

```
┌─────────────────────────────────────────────────────────────────┐
│                      En-NSGA-II Framework                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │     PPO      │───▶│     GNN      │───▶│   NSGA-II    │       │
│  │  Adaptive    │    │   Surrogate  │    │   Pareto     │       │
│  │ Initialization│    │  Evaluation  │    │    Search    │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│   Policy Learning     Fast Approximation    Multi-objective     │
│   for Promising       of Emissions &       Evolutionary         │
│   Search Regions      Cost Functions       Optimization        │
└─────────────────────────────────────────────────────────────────┘
```

- **PPO (Proximal Policy Optimization):** Learns adaptive initialization policies to focus search on promising regions of the trade-off space, achieving 86.2% cost reduction vs random initialization.

- **GNN (Graph Neural Network):** Provides fast surrogate evaluation of route emissions (R² = 0.935) and cost (R² = 0.954), achieving 5-6× speedup.

- **NSGA-II:** Performs robust Pareto frontier search with enhanced environmental weighting for decarbonization priorities.

---

## Key Findings

| Finding | Value | Implication |
|---------|-------|-------------|
| **Total road CO₂ emissions** | 3.66 million tonnes | Baseline established for corridor |
| **En-NSGA-II baseline emissions** | 474.9 kg CO₂ | 68% lower than NSGA-II |
| **En-NSGA-II baseline cost** | $346.15 | Only 10% premium over NSGA-II |
| **Optimal carbon price** | **$55/ton CO₂** | Policy target for corridor |
| **Minimum emissions achieved** | **336.0 kg CO₂** | 29% reduction from baseline |
| **Total decarbonization potential** | **1.79 million tonnes** | 48.8% of baseline |
| **Most cost-effective measure** | Driver training | $10/tonne |

### Algorithm Performance Summary

| Algorithm | Cost (USD) | Emissions (kg) | Time (s) | vs NSGA-II |
|-----------|------------|----------------|----------|------------|
| **En-NSGA-II** | **346.15** | **474.9** | 0.71 | **68% lower emissions** |
| NSGA-II | 314.34 | 1,500.9 | 0.73 | Baseline |
| SPEA2 | 552.62 | 563.2 | 0.66 | 15.7% lower emissions |
| MOEA/D | 394.04 | 921.3 | 0.11 | 38.6% lower emissions |
| Branch & Cut | 398.17 | 1,763.2 | 0.05 | 17.5% higher emissions |

### Carbon Price Sensitivity (En-NSGA-II)

| Price ($/ton) | Emissions (kg) | Cost (USD) | Reduction |
|--------------|----------------|------------|------------|
| $0 | 474.9 | 346.15 | Baseline |
| $55 | **336.0** | 492.53 | **29%** |
| $150 | 498.6 | 792.71 | -5% |

---

## Repository Structure

```
NewRepo/
│
├── README.md                       # This file
├── LICENSE                         # MIT License
├── requirements.txt                # Python dependencies
│
├── data/
│   ├── raw/                        # Original datasets
│   │   ├── CO2 Emission - Road.csv
│   │   ├── Pollutants - Road.csv
│   │   ├── rail-related GHG emissions for Kenya-*.xlsx
│   │   ├── Transit time in Kenya.xls
│   │   ├── Transit time in Uganda.xls
│   │   ├── Transit time in Rwanda.xls
│   │   ├── Weighbridge Traffic.xls
│   │   ├── The Vessels Waiting Time before Berth.xls
│   │   ├── Ship Turnaround Time.xls
│   │   └── Volume per country of destination (TC).xls
│   │
│   └── processed/                  # Cleaned and aggregated data
│       ├── corridor_emissions.csv
│       ├── algorithm_results.csv
│       └── carbon_price_sensitivity.csv
│
├── notebooks/                      # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_emissions_analysis.ipynb
│   ├── 03_algorithm_comparison.ipynb
│   ├── 04_carbon_price_analysis.ipynb
│   └── 05_visualization_dashboard.ipynb
│
├── src/                            # Source code
│   ├── __init__.py
│   ├── data_loader.py              # Load all datasets
│   ├── emissions_analyzer.py       # Emissions computation
│   ├── algorithm_comparator.py     # Algorithm comparison
│   ├── carbon_price_sensitivity.py # Carbon price analysis
│   ├── var_cvar_analyzer.py        # VaR/CVaR for borders
│   └── visualization.py            # Plot generation
│
├── results/
│   ├── figures/                    # All generated figures
│   │   ├── figure_top_routes_co2.png
│   │   ├── figure_emissions_by_vehicle_class.png
│   │   ├── figure_mac_curve.png
│   │   ├── figure_border_crossing_times.png
│   │   ├── figure_rail_emissions_trend.png
│   │   ├── figure_national_validation.png
│   │   ├── figure_lake_ports_cargo.png
│   │   ├── figure_algorithm_comparison_bars.png
│   │   ├── figure_carbon_price_sensitivity.png
│   │   ├── figure_carbon_price_emissions_comparison.png
│   │   ├── figure_carbon_price_cost_comparison.png
│   │   └── figure_carbon_price_tradeoff_comparison.png
│   │
│   └── tables/                     # Summary tables
│       ├── algorithm_performance_summary.csv
│       ├── carbon_price_sensitivity_summary.csv
│       └── route_emissions_summary.csv
│
├── docs/                           # Documentation
│   ├── supplementary_material.pdf
│   └── data_documentation.md
│
└── scripts/                        # Execution scripts
    ├── run_full_analysis.py        # Run all analyses
    └── generate_all_figures.py     # Generate all figures
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/YvesNDIKURIYO-2022/NewRepo.git
cd NewRepo
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the full analysis

```bash
python scripts/run_full_analysis.py
```

### 4. Explore Jupyter notebooks

```bash
jupyter notebook notebooks/
```

### 5. Generate all figures

```bash
python scripts/generate_all_figures.py
```

---

## Data Sources

The analysis integrates 17 independent datasets from 9 authoritative sources:

| Source | Data Type | Years | Records |
|--------|-----------|-------|---------|
| **NCTTCA Transport Observatory** | Road CO₂ emissions, pollutants, border times, transit times | 2009-2026 | 132+ |
| **Kenya Ministry of Transport** | Rail GHG emissions, national transport emissions | 2015-2022 | 16 |
| **Tanzania Ports Authority** | Lake ports cargo (Victoria, Tanganyika, Nyasa) | 2022 | 3 lakes |
| **EDGAR Database** | Country-level GHG emissions | 1990-2024 | 7 countries |
| **International Energy Agency** | Uganda CO₂ emissions | 2023 | 1 country |
| **Worldometer** | Uganda historical CO₂ | 1970-2024 | 55 years |
| **Climate Change Tracker** | Kenya GHG breakdown | 2024 | 1 country |
| **Kenya Ports Authority** | Cargo volumes, port performance | 2009-2018 | Monthly |
| **KeNHA / KRA / URA / RRA** | Weighbridge, customs clearance | 2010-2022 | Monthly |

### Data Availability Statement

All data used in this study are publicly available from:

- **NCTTCA Transport Observatory:** https://top.ttcanc.org
- **EDGAR Database:** https://edgar.jrc.ec.europa.eu
- **IEA Data Services:** https://www.iea.org
- **Worldometer:** https://www.worldometers.info
- **Kenya Ministry of Transport:** https://www.transport.go.ke
- **Tanzania Ports Authority:** https://www.ports.go.tz
- **Kenya Ports Authority:** https://www.kpa.co.ke
- **Climate Change Tracker:** https://climate-change-tracker.org

Processed data and analysis scripts are available from the corresponding author upon reasonable request.

---

## Key Results

### Road Freight Emissions by Route

| Rank | Route Section | CO₂ (tonnes) | % of Total |
|------|---------------|--------------|-------------|
| 1 | Mombasa → Nairobi | 1,455,474 | 39.8% |
| 2 | Nairobi → Malaba | 862,799 | 23.6% |
| 3 | Malaba → Kampala | 325,590 | 8.9% |
| 4 | Malaba → Elegu | 239,674 | 6.5% |
| 5 | Kigali → Rusizi | 141,177 | 3.9% |
| 6 | Kobero-Gatumba (Burundi) | 109,636 | 3.0% |
| | **TOTAL** | **3,660,520** | **100%** |

### CO₂ Emissions by Vehicle Class

| Vehicle Class | CO₂ (tonnes) | Percentage |
|---------------|--------------|------------|
| HCV-5&6 Axle | 2,445,939 | 66.8% |
| LDT/2-Axle | 776,262 | 21.2% |
| MCV-3Axle | 318,774 | 8.7% |
| MCV-4Axle | 74,959 | 2.0% |
| HCV-7&8 Axle | 42,522 | 1.2% |
| HCV->8 Axle | 2,062 | 0.1% |

### Pollutant Emissions

| Pollutant | Total (kg) | Percentage |
|-----------|------------|------------|
| NOx | 19,685,911 | 94.3% |
| PM10 | 978,857 | 4.7% |
| BC | 212,003 | 1.0% |

### Border Crossing Risk (VaR/CVaR at 95%)

| Border | Mean (min) | VaR(95%) (min) | CVaR(95%) (min) | EAC Target |
|--------|------------|----------------|-----------------|------------|
| Busia | 98.2 | 122.9 | 129.1 | 30 |
| Malaba | 91.0 | 110.7 | 115.8 | 30 |
| Katuna | 141.0 | 173.9 | 182.9 | 30 |

### Lake Ports Cargo (2022)

| Lake | Cargo (tons) | Route to Bujumbura |
|------|-------------|---------------------|
| Lake Tanganyika | 295,560 | Direct via Kigoma |
| Lake Victoria | 235,042 | Alternative via Uganda |
| Lake Nyasa | 5,410 | Southern corridor |

### National Validation

| Country | Corridor CO₂ (Mt) | National (Mt) | Percentage | Source |
|---------|-------------------|---------------|------------|--------|
| Kenya | 2.59 | 102.84 | 2.5% | EDGAR |
| Uganda | 0.83 | 8.25 | 10.1% | Worldometer |
| Rwanda | 0.34 | 8.35 | 4.1% | EDGAR |

### Marginal Abatement Cost (MAC) Curve

| Measure | Abatement (tonnes) | Cost ($/tonne) |
|---------|-------------------|----------------|
| Driver eco-training | 183,026 | **$10** |
| Empty trip reduction | 378,674 | $25 |
| Modal shift to rail | 549,078 | $50 |
| SGR electrification | 53,000 | $75 |
| EV truck deployment (10% fleet) | 439,262 | $150 |

### Decarbonization Potential

| Scenario | Reduction (tonnes) | % of Baseline |
|----------|-------------------|----------------|
| Fuel Efficiency (10%) | 366,052 | 10.0% |
| Modal Shift to Rail | 549,078 | 15.0% |
| Empty Trip Reduction (45%→30%) | 378,674 | 10.3% |
| SGR Electrification | 53,000 | 1.4% |
| EV Trucks (10% fleet) | 439,262 | 12.0% |
| **COMBINED** | **1,786,067** | **48.8%** |

---

## Figures

All figures are generated in `results/figures/`:

| Figure | Description |
|--------|-------------|
| `figure_top_routes_co2.png` | Top 8 routes by CO₂ emissions |
| `figure_emissions_by_vehicle_class.png` | CO₂ distribution by vehicle class (pie chart) |
| `figure_mac_curve.png` | Marginal Abatement Cost curve |
| `figure_border_crossing_times.png` | Border crossing comparison with EAC target |
| `figure_rail_emissions_trend.png` | Kenya rail emissions (2015-2022) |
| `figure_national_validation.png` | Corridor vs national emissions validation |
| `figure_lake_ports_cargo.png` | Lake ports cargo traffic |
| `figure_algorithm_comparison_bars.png` | Algorithm performance comparison |
| `figure_carbon_price_sensitivity.png` | En-NSGA-II carbon price sensitivity |
| `figure_carbon_price_emissions_comparison.png` | Emissions vs price (all algorithms) |
| `figure_carbon_price_cost_comparison.png` | Cost vs price (all algorithms) |
| `figure_carbon_price_tradeoff_comparison.png` | Cost-emissions trade-off (all algorithms) |

---

## Case Study: Mombasa–Bujumbura Corridor

The case study focuses on the **Mombasa–Bujumbura corridor**, a critical East African trade route connecting the Port of Mombasa (Kenya) to Bujumbura (Burundi), traversing Uganda and Rwanda.

### Corridor Map

```
MOMBASA (Kenya) ──(485 km, 1.46 MMt CO₂)──► NAIROBI
         │
         │ (480 km, 0.86 MMt CO₂)
         ▼
    MALABA (Kenya-Uganda Border)
         │
         │ (171 km, ~0.06 MMt CO₂)
         ▼
    KAMPALA (Uganda)
         │
         │ (508 km, ~0.05 MMt CO₂)
         ▼
    KIGALI (Rwanda)
         │
         │ (298 km, ~0.02 MMt CO₂)
         ▼
    BUJUMBURA (Burundi) ★ DESTINATION

Total Distance: 1,942 km
Total Estimated CO₂: ~2.44 MMt CO₂
```

### Key Corridor Characteristics

| Feature | Description |
|---------|-------------|
| **Nodes** | 19 nodes across 5 countries |
| **Arcs** | 34 bidirectional arcs |
| **Modes** | Road, railway, waterway (Lake Victoria, Lake Tanganyika) |
| **Primary cargo** | Containerized freight |
| **Annual traffic** | ~26,790 trucks/day |
| **Rail share** | 26% (up from 5% in 2017) |

---

## Citation

If you use this code or data in your research, please cite:

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

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

**Yves Ndikuriyo**  
*PhD Candidate*  
School of Traffic and Transportation Engineering  
Central South University, Changsha, China  
Email: yvesndikuriyo@csu.edu.cn  
ORCID: 0009-0006-9324-7265

**Professor Yinggui Zhang**  
Email: ygzhang@csu.edu.cn  
ORCID: 0000-0002-5790-0638

---

## Acknowledgments

This research was supported by:

- **NCTTCA (Northern Corridor Transit and Transport Coordination Authority)** for providing Transport Observatory data
- **Tanzania Ports Authority** for lake ports data
- **Kenya Ministry of Transport** for rail emissions data
- **EDGAR, IEA, Worldometer, Climate Change Tracker** for validation data
- **TradeMark East Africa** for corridor performance reports

---

<div align="center">
  <sub>Built with Python • Analysis for Journal of Cleaner Production • Last updated: May 2026</sub>
</div>
