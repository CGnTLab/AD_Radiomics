# Radiomic-Based Machine Learning Models for Alzheimer's Disease Classification and Progression Prediction

This repository supports the study titled _"Radiomic-based investigation of a potential link between precuneus and fusiform gyrus with Alzheimer’s disease"_, which explores structural biomarkers from MRI data to distinguish stages of Alzheimer’s Disease (AD) and predict its progression using machine learning models.

## 📌 Overview

This project applies radiomic feature analysis of the precuneus and fusiform gyrus regions from T1-weighted MRI images using FreeSurfer. It implements:

- **Random Forest Classification** models for binary classification (AD vs CN, AD vs MCI, MCI vs CN) with and without age.
- **Linear Regression** and **ARIMA Time-Series** models to predict disease progression using volumetric and surface-based MRI features.

## 🧠 Dataset

- Data is sourced from the [Alzheimer's Disease Neuroimaging Initiative (ADNI)](http://adni.loni.usc.edu/)
- 382 subjects: CN (134), MCI (149), AD (99)
- Four time points: baseline, 6, 12, and 24 months

## 🧪 Features

- 9 radiomic features extracted bilaterally from precuneus and fusiform gyrus using FreeSurfer:
  - Gray Matter Volume (GMV), Cortical Thickness (CT), Surface Area, Curvatures, etc.

## 🛠️ Implementation

- **Classification scripts** implemented in Python using `scikit-learn`, `imblearn`, and `joblib` for model training and persistence. See `AD_classification_script.py`.
- **Linear Regression & Time-Series modeling** implemented in:
  - **Python**: Linear Regression using `statsmodels` or `scikit-learn`
  - **R**: ARIMA models using `forecast::auto.arima()` (see `AD_TimeSeries_Script.Rmd`)
- Statistical tests conducted using `scipy.stats.mannwhitneyu` in Python and post-hoc Benjamini-Hochberg correction.

## 🗂️ File Structure

```
📁 AD_Radiomics_Classification
├── AD_classification_script.py           # Random forest models for all group pairs
├── AD_TimeSeries_Script.Rmd              # Time-series model (ARIMA) and regression in R
├── AD_LinearRegression_script.html       # HTML output of linear regression analysis

```

## 🚀 Getting Started

### Dependencies

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Install R dependencies:

```r
install.packages(c("forecast", "ggplot2", "dplyr", "tidyverse"))
```

### Running the Models

#### 1. Classification

```bash
python AD_classification_script.py
```

#### 2. Time-Series & Regression

Open and run the R Markdown script:

```r
rmarkdown::render("AD_TimeSeries_Script.Rmd")
```

## 📈 Results Summary

- **Classification** models achieved up to 85.7% test accuracy (AD vs CN).
- **Time-series models** reached up to 0.98 Pearson correlation for predicting radiomic feature evolution (e.g., left fusiform GMV).

## 📄 Citation

If you use this code or dataset, please cite the manuscript once published.

## 🛡️ License

This project is open-sourced for academic use. See [LICENSE](LICENSE) for more details.
