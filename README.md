# Multi-Layer Perceptron for U.S. Census Poverty Classification

This project implements a supervised **multi-layer perceptron (MLP)** to classify U.S. Census tracts into four ordinal poverty levels based on socioeconomic features. The goal is to investigate how neural networks perform on real-world demographic data and evaluate the relationship between feature preprocessing, class balancing, and model accuracy.

---

## Dataset

- Source: U.S. Census (American Community Survey)
- Unit of analysis: Census **tract** (≈ neighborhoods)
- Target variable: Child poverty rate (%)
- Converted from continuous regression → **4 quantized ordinal classes**
- Features include: income, education level, household statistics, race/ethnicity distributions, etc.

---

## Methodology

| Stage | Description |
|--------|-------------|
| **Data Preparation** | Cleaning, feature selection, missing value handling |
| **Quantization** | Continuous poverty % → 4 balanced ordinal classes |
| **Encoding** | One-hot and normalization of continuous attributes |
| **Model** | Custom MLP (NumPy / from-scratch implementation) |
| **Initialization** | Glorot (Xavier) weight initialization |
| **Adaptive Learning Optimizer** | RMSProp and Adam (configurable) |
| **Loss Function** | Cross-entropy over one-hot labels |
| **Evaluation** | Accuracy, macro-F1, per-class metrics |

---

## Key Learning Outcomes

Implemented an MLP from scratch (no PyTorch/TensorFlow)  
Explored effects of data balancing on model performance  
Applied Glorot initialization + adaptive optimization 
Compared gradient activity across epochs and covergence behavior 
Demonstrated model performance on real demographic data  
Interpreted classification behavior via per-class metrics  

---

## Requirements

Python 3.10+
NumPy
Pandas
Matplotlib / Seaborn
