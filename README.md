# Tax Anomaly Detection System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

An advanced machine learning system for detecting anomalies in Nigeria's fiscal and economic indicators, with particular focus on tax revenue patterns and their relationship to election periods.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Requirements](#data-requirements)
- [Methodology](#methodology)
- [Project Structure](#project-structure)
- [Usage Examples](#usage-examples)
- [Model Performance](#model-performance)
- [Configuration](#configuration)
- [Results Interpretation](#results-interpretation)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## 🎯 Overview

This project applies state-of-the-art statistical and machine learning methods to identify anomalies in Nigeria's fiscal and economic data. The system combines multiple anomaly detection algorithms with consensus voting to provide robust, reliable detection of unusual patterns in tax revenues and economic indicators.

**Key Applications:**
- Tax compliance auditing
- Economic policy analysis
- Fraud detection in fiscal data
- Political economy research
- Revenue forecasting validation

## ✨ Features

- **Multiple Detection Algorithms**: Isolation Forest, Local Outlier Factor (LOF), KNN, HBOS, Elliptic Envelope
- **Consensus Voting**: Combines predictions from multiple models for robust detection
- **Advanced Preprocessing**: Handles skewness, multicollinearity, and missing data
- **Dimensionality Reduction**: PCA-based GDP composite index creation
- **Election Period Analysis**: Evaluates anomaly patterns around political events
- **Comprehensive Evaluation**: Multiple metrics including Average Precision, PR AUC, Precision@k
- **Rich Visualizations**: Time series plots, scatter plots, correlation heatmaps
- **Hyperparameter Tuning**: Grid search across multiple configurations

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Google Colab account (optional, for cloud execution)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/smatralph/Anomaly-Detection.git
   cd Anomaly-Detection
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Required Dependencies

```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
statsmodels>=0.13.0
pyod>=1.0.0
openpyxl>=3.0.0
```

## ⚡ Quick Start

### Basic Usage

```python
from anomaly_detector import TaxAnomalyDetector

# Initialize detector
detector = TaxAnomalyDetector(filepath='data/FIRS_Taxes.xlsx')

# Run full pipeline
detector.load_data() \
        .clean_data() \
        .transform_features() \
        .create_gdp_composite() \
        .fit_models() \
        .consensus_detection() \
        .evaluate_models() \
        .plot_results()

# Get results
results = detector.get_results()
anomalies = detector.get_anomalies()
```

### Google Colab

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Run notebook
!jupyter notebook Tax_Anomaly_Detection.ipynb
```

## 📊 Data Requirements

### Input Data Format

The system expects an Excel file with the following structure:

| Column Name | Description | Type | Required |
|------------|-------------|------|----------|
| Year | Calendar year | Integer | Yes |
| Quarter | Quarter (Q1-Q4) | String | Yes |
| Petroleum Profits Tax | PPT revenue (Billions NGN) | Float | Yes |
| VAT | Value Added Tax (Billions NGN) | Float | Yes |
| Brent Crude Oil Price | Oil price (USD/barrel) | Float | Yes |
| Nominal GDP (Crude Petroleum and Gas) | GDP sector (Billions NGN) | Float | Yes |
| Nominal GDP (Industry) | GDP sector (Billions NGN) | Yes |
| Nominal GDP (Manufacturing) | GDP sector (Billions NGN) | Yes |
| Nominal GDP (Trade) | GDP sector (Billions NGN) | Yes |
| Nominal GDP (Information and Communication) | GDP sector (Billions NGN) | Yes |
| Nominal GDP (Real Estate) | GDP sector (Billions NGN) | Yes |
| Election Flag | Binary indicator (0/1) | Integer | Yes |

### Sample Data

```csv
Year,Quarter,Petroleum Profits Tax,VAT,Brent Crude Oil Price,Election Flag
2015,Q1,234.5,89.3,52.4,1
2015,Q2,198.7,92.1,61.8,0
...
```

## 🔬 Methodology

### 1. Data Preprocessing

**Cleaning Steps:**
- Remove irrelevant columns (Company Income Tax, Gas Income, Government Spending)
- Replace zeros with NaN to handle missing data
- Median imputation by quarter
- Forward/backward fill for remaining gaps

**Transformation:**
- Log transformation for skewed variables (|skew| > 1)
- PCA for GDP composite index creation
- Standardization using Z-score normalization

### 2. Feature Engineering

**GDP Composite Index:**
```python
# Five GDP sectors compressed into single component
PCA(n_components=1).fit([
    "Nominal GDP (Industry)",
    "Nominal GDP (Manufacturing)",
    "Nominal GDP (Trade)",
    "Nominal GDP (Information and Communication)",
    "Nominal GDP (Real Estate)"
])
```

**Final Features:**
- Petroleum Profits Tax (log-transformed)
- VAT (log-transformed)
- Brent Crude Oil Price
- Nominal GDP (Crude Petroleum and Gas)
- GDP Composite Index

### 3. Anomaly Detection Models

| Model | Method | Key Parameters |
|-------|--------|---------------|
| **Isolation Forest** | Tree-based isolation | contamination: [0.05, 0.1, 0.15], n_estimators: 100 |
| **LOF** | Local density deviation | n_neighbors: [5, 10, 15], contamination: [0.05, 0.1, 0.15] |
| **KNN** | k-nearest neighbors distance | n_neighbors: [5, 10, 15], contamination: [0.05, 0.1, 0.15] |
| **HBOS** | Histogram-based outlier score | n_bins: [5, 10, 20], contamination: [0.05, 0.1, 0.15] |
| **Elliptic Envelope** | Gaussian distribution assumption | support_fraction: [0.8, 0.85, 0.9], contamination: [0.05, 0.1, 0.15] |

### 4. Consensus Voting

An observation is flagged as anomalous if **≥3 models** agree:

```
Consensus_Anomaly = (Σ Model_Predictions) ≥ 3
```

This approach balances sensitivity and specificity while reducing false positives.

### 5. Evaluation Metrics

- **Average Precision (AP)**: Quality of anomaly ranking
- **PR AUC**: Area under Precision-Recall curve
- **Precision@k**: Precision at top-k anomalies (k=4, k=8)
- **Election Period Analysis**: Anomalies occurring during elections
- **Chi-squared Test**: Statistical association with election periods

## 📁 Project Structure

```
tax-anomaly-detection/
│
├── README.md
├── requirements.txt
├── LICENSE                         
├── .gitignore
│
├── main/
│   └── Tax_Anomaly_Detection.py
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py       
│   ├── visualization.py            
│   ├── transformation.py         
│   ├── anomaly_models.py          
│   ├── evaluation.py        
│   ├── utils.py                    
│
├── scripts/
│   ├── run_pipeline.py       
│   └── tune_models.py         
│
└── docs/
    ├── project_summary.md
    ├── methods.md
    └── references.md

```

## 💻 Usage Examples

### Example 1: Basic Detection

```python
import pandas as pd
from src.anomaly_detector import TaxAnomalyDetector

# Load and process data
detector = TaxAnomalyDetector('data/FIRS_Taxes.xlsx')
detector.run_full_pipeline()

# View anomalies
anomalies = detector.get_anomalies()
print(f"Detected {len(anomalies)} anomalies")
print(anomalies[['Year', 'Quarter', 'VAT', 'Election Flag']])
```

### Example 2: Custom Configuration

```python
# Custom parameter grid
custom_params = {
    "IsolationForest": {
        "contamination": [0.08, 0.12],
        "n_estimators": [150]
    },
    "LOF": {
        "n_neighbors": [8, 12],
        "contamination": [0.08, 0.12]
    }
}

detector = TaxAnomalyDetector('data/FIRS_Taxes.xlsx')
detector.fit_models(param_grids=custom_params)
```

### Example 3: Visualization Only

```python
from src.visualization import plot_timeline, plot_scatter

# Load results
results = pd.read_csv('results/anomaly_predictions.csv')

# Create plots
plot_timeline(results, anomaly_col='Consensus_Anomaly', 
              election_col='Election Flag')
plot_scatter(results, x='VAT', y='GDP Composite Index',
             anomaly_col='Consensus_Anomaly')
```

## 📈 Model Performance

### Best Model Results (Test Data)

| Model | Anomalies Detected | Average Precision | PR AUC | P@8 | Election Anomalies |
|-------|-------------------|-------------------|--------|-----|-------------------|
| **Isolation Forest** | 3 | 1.00 | 1.00 | 1.00 | 3 |
| **LOF** | 3 | 1.00 | 1.00 | 1.00 | 3 |
| **Elliptic Envelope** | 3 | 1.00 | 1.00 | 1.00 | 3 |
| **KNN** | 0 | - | - | - | 0 |
| **HBOS** | 0 | - | - | - | 0 |
| **Consensus** | 1 | 1.00 | 1.00 | 1.00 | 1 |

### Key Findings

✅ **100% of detected anomalies occurred during election periods**

✅ **Perfect precision scores** for Isolation Forest, LOF, and Elliptic Envelope

✅ **Consensus model** identified the most reliable anomaly

✅ **Strong statistical association** between anomalies and elections (χ² test)

## ⚙️ Configuration

### Configuration File (config.yaml)

```yaml
data:
  filepath: "data/FIRS_Taxes.xlsx"
  irrelevant_columns:
    - "Company Income Tax"
    - "Gas Income"
    - "Government Spending (N' Billion)"
  
preprocessing:
  skew_threshold: 1.0
  imputation_method: "median"
  scaling_method: "standard"
  
feature_engineering:
  pca_components: 1
  gdp_vars:
    - "Nominal GDP (Industry)"
    - "Nominal GDP (Manufacturing)"
    - "Nominal GDP (Trade)"
    - "Nominal GDP (Information and Communication)"
    - "Nominal GDP (Real Estate)"

models:
  isolation_forest:
    contamination: [0.05, 0.1, 0.15]
    n_estimators: [100]
  
  lof:
    n_neighbors: [5, 10, 15]
    contamination: [0.05, 0.1, 0.15]

evaluation:
  consensus_threshold: 3
  k_values: [4, 8]
  metrics:
    - "average_precision"
    - "pr_auc"
    - "precision_at_k"
```

## 📖 Results Interpretation

### Understanding Anomalies

**Consensus Anomaly (Most Reliable):**
- Flagged by ≥3 models simultaneously
- Represents strongest signal of abnormal behavior
- Highest confidence for audit prioritization

**Individual Model Anomalies:**
- May capture different types of irregularities
- Useful for sensitivity analysis
- Consider for secondary review

### Election Period Context

The strong correlation between anomalies and election periods suggests:

1. **Policy shifts** during election campaigns
2. **Reporting adjustments** in politically sensitive periods
3. **Economic volatility** associated with political uncertainty
4. **Strategic fiscal behavior** by government entities

### Actionable Insights

- **Audit Priority**: Focus on consensus anomalies first
- **Temporal Patterns**: Monitor quarters immediately before/after elections
- **Threshold Monitoring**: VAT deviations from GDP trends
- **Comparative Analysis**: Year-over-year changes during election cycles

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit changes** (`git commit -m 'Add AmazingFeature'`)
4. **Push to branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

### Code Standards

- Follow PEP 8 style guide
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation as needed

### Reporting Issues

Please use GitHub Issues to report bugs or request features. Include:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- System information (OS, Python version)

## 📚 Citation

If you use this project in your research, please cite:

```bibtex
@misc{tax_anomaly_detection_2025,
  author = {Smatralph},
  title = {Tax Anomaly Detection System for Nigerian Fiscal Data},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/smatralph/Anomaly-Detection}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Smatralph

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

## 🙏 Acknowledgments

- **PyOD Library**: Comprehensive anomaly detection algorithms
- **Scikit-learn**: Machine learning infrastructure
- **Nigerian FIRS**: Data on tax revenues
- **Research Community**: Methods and best practices

## 📞 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/smatralph/Anomaly-Detection/issues)
- **Discussions**: [GitHub Discussions](https://github.com/smatralph/Anomaly-Detection/discussions)
- **Email**: [r.b.adeagbo@gmail.com]

## 🔄 Version History

- **v1.0.0** (2025-01-XX): Initial release
  - Five anomaly detection models
  - Consensus voting mechanism
  - Election period analysis
  - Comprehensive evaluation metrics

---

**⭐ If you find this project useful, please consider giving it a star!**

For detailed methodology and mathematical foundations, see our [Wiki](https://github.com/smatralph/Anomaly-Detection/wiki).