# Code Survival Intelligence

**A Hybrid Predictive Framework for Cost-Optimal Software Failure Risk Modeling**

---

## Overview

This project transforms static code quality metrics into **financially actionable refactoring recommendations** by combining:

- **Survival Analysis (Cox Proportional Hazards)** — predicts *when* code modules will fail, not just *if*
- **Machine Learning Baselines (Random Forest, Logistic Regression)** — benchmark classifiers for comparison  
- **ROI Scoring Engine** — converts failure probabilities into dollar-value Return on Investment

Built on the [Lenarduzzi et al. Technical Debt Dataset V2](https://github.com/clowee/The-Technical-Debt-Dataset) — 31 Apache Java projects, 154K+ commits, linked SonarQube analysis and SZZ fault-inducing commit history.

## Key Results

| Model | C-index | AUC-ROC | Brier Score |
|---|---|---|---|
| **Cox PH** | **0.80** | **0.660** | **0.083** |
| Logistic Regression | — | 0.643 | 0.122 |
| Random Forest | — | 0.569 | 0.305 |

### Financial Impact (1-year horizon, test set)

| Metric | Value |
|---|---|
| Total files analyzed | 37,102 |
| Files with positive ROI | 17,597 (47.4%) |
| Total Expected Loss | $5,922,733 |
| Total Refactoring Cost | $2,671,502 |
| **Net Savings** | **$3,251,231** |

### Sample Output

The top refactoring priority (`EqualPredicateTest.java`) has:
- 51% probability of failure within 1 year
- $2,064 expected loss if it fails  
- $56 to refactor proactively
- **3,568% ROI**

## Architecture

```
td_V2.db (SQLite)
    │
    ├── Phase 1: Survival Label Construction
    │   └── SZZ fault-inducing commits → time-to-failure targets
    │
    ├── Phase 2: Feature Engineering (16 features)
    │   ├── Code churn (lines added/removed, total churn)
    │   ├── Ownership (contributors, major contributor ratio)
    │   ├── SonarQube file-level (bugs, vulnerabilities, code smells, debt)
    │   └── SonarQube project-level (SQALE index, complexity, coverage)
    │
    ├── Phase 3: Dataset Consolidation
    │   └── 113,922 file observations × 21 columns
    │
    └── Phase 4: Modeling & ROI
        ├── Cox PH → survival curves S(t) at 90/180/365/730 days
        ├── P(failure) = 1 - S(t) → Expected Loss ($)
        └── ROI = (Expected Loss - Refactor Cost) / Refactor Cost
```

## Project Structure

```
├── src/
│   ├── data_processing/
│   │   ├── build_survival_labels.py   # Phase 1: construct survival targets
│   │   ├── feature_extraction.py      # Phase 2: extract 16 features from DB
│   │   └── build_dataset.py           # Phase 3: merge into training set
│   ├── models/
│   │   ├── train_and_compare.py       # Phase 4a: train & compare 3 models
│   │   └── roi_scorer.py              # Phase 4b: dollar-value ROI engine
│   └── db_connector.py                # DB utility class
├── data/                              # Generated CSVs + database (gitignored)
├── results/                           # Plots and result CSVs
├── docs/                              # Project proposal and presentation
└── README.md
```

## Quick Start

### Prerequisites

- Python 3.10+
- [Technical Debt Dataset V2](https://github.com/clowee/The-Technical-Debt-Dataset/releases) (`td_V2.db`, ~1.5GB)

### Setup

```bash
# Clone the repository
git clone https://github.com/Chiron-R/code-survival-intelligence.git
cd code-survival-intelligence

# Create virtual environment
python -m venv .venv
.venv/Scripts/activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Download the database
# Place td_V2.db in the data/ directory
```

### Run the Pipeline

```bash
# Step 1: Build survival labels
python src/data_processing/build_survival_labels.py

# Step 2: Extract features
python src/data_processing/feature_extraction.py

# Step 3: Merge into training set
python src/data_processing/build_dataset.py

# Step 4a: Train models and compare
python src/models/train_and_compare.py

# Step 4b: Compute dollar-value ROI scores
python src/models/roi_scorer.py
```

All results and plots are saved to `results/`.

## Visualizations

The pipeline generates 9 publication-ready plots:

| Plot | Description |
|---|---|
| `roc_comparison.png` | ROC curves for Cox PH vs RF vs LR |
| `feature_importance_rf.png` | Random Forest feature importance ranking |
| `kaplan_meier.png` | Kaplan-Meier survival curves (test set) |
| `metrics_comparison.png` | Model comparison bar chart |
| `roi_priority_heatmap.png` | Top 30 refactoring priorities by ROI |
| `failure_prob_horizons.png` | Multi-horizon failure probability curves |
| `loss_vs_investment.png` | Expected loss vs refactoring cost scatter |
| `risk_tier_distribution.png` | Risk tier distribution (count + dollars) |
| `roi_distribution.png` | ROI score distribution histogram |

## Dataset Attribution

This project uses the [Technical Debt Dataset](https://github.com/clowee/The-Technical-Debt-Dataset) by Lenarduzzi et al.:

> Lenarduzzi, V., Saarimäki, N., & Taibi, D. (2019). *The Technical Debt Dataset*. Proceedings of the 15th International Conference on Predictive Models and Data Analytics in Software Engineering.

## License

MIT
