# Seasonal Agriculture Performance & Yield Intelligence

**Author:** Aayush Kumar Singh

## Overview

This repository contains a comprehensive data science analysis of agricultural productivity across different seasons, states, and farming conditions in India. The project investigates environmental and financial seasonal patterns, identifies profitability drivers, and builds a high-accuracy predictive model for yield estimation using precision agriculture metrics.

## Repository Structure

```
Seasonal-Agriculture-Performance-Analysis/
├── Seasonal_Agriculture_Performance_Analysis.ipynb   # Main analysis notebook
├── create_notebook.py                                 # Script to generate the notebook
├── seasonal_agriculture_performance_dataset.csv       # Agricultural performance dataset
└── .gitignore                                         # Git ignore configuration
```

## Dataset

The dataset (`seasonal_agriculture_performance_dataset.csv`) represents agricultural activities across different seasons, geographical districts, and farming conditions in India. It contains parameters related to farming practices, environmental conditions, crop production, resource usage, and economic performance.

### Data Dictionary

| Column | Category | Description |
| :--- | :--- | :--- |
| **Farm_ID** | Categorical | Unique identifier for each farm unit |
| **State / District** | Categorical | Location details |
| **Crop / Season** | Categorical | Crop type and cultivation season (Kharif, Rabi, Zaid) |
| **Rainfall_mm / Avg_Temperature_C / Humidity_pct** | Climate | Environmental metrics |
| **Soil_pH / Soil_Moisture_pct** | Soil | Agronomic soil properties |
| **Nitrogen_kg_ha / Phosphorus_kg_ha / Potassium_kg_ha** | Soil Nutrients | NPK fertilizer inputs |
| **Irrigation_Method** | Practice | Drip, Sprinkler, Flood, or Rainfed |
| **Fertilizer_kg_ha / Pesticide_Litre_ha** | Inputs | Agrochemical usage per hectare |
| **Yield_Tonnes_Ha** | Target Metric | Crop yield produced per hectare |
| **Production_Tonnes** | Output | Total production volume |
| **Total_Cost_INR / Revenue_INR / Profit_INR** | Financial | Economic outcome metrics |
| **Water_Used_m3 / Water_Efficiency_t_per_1000m3** | Resource | Water usage and volumetric efficiency |

## Analysis Workflow

The main notebook (`Seasonal_Agriculture_Performance_Analysis.ipynb`) follows a structured analytical pipeline:

### 1. Data Cleaning & Quality Control
- Missing value imputation using robust median strategy
- Duplicate record detection
- Statistical summary inspection

### 2. Exploratory Data Analysis (EDA) - Deep Seasonal Focus
- **Seasonal Environmental Variations**: Analysis of rainfall, temperature, humidity, and soil moisture across Kharif, Rabi, and Zaid seasons
- **Seasonal Economic & Yield Outcomes**: Comparison of average yield, revenue, costs, and net profit across seasons
- **Crop vs. Season Performance Matrix**: Heatmaps identifying which specific crops perform best in which season
- **Resource Allocation and Irrigation Analysis**: Interaction between irrigation methods, seasons, and yield

### 3. Statistical Hypothesis Testing
- ANOVA tests to confirm whether differences in Yield, Profit, and Water Efficiency across seasons are statistically significant

### 4. Feature Correlation Matrix
- Pairwise correlation analysis between numerical agronomic, climate, and financial features

### 5. Machine Learning Yield Prediction Modeling
- Predictive models for `Yield_Tonnes_Ha` using:
  - Linear Regression (Baseline)
  - Random Forest Regressor
  - Gradient Boosting Regressor
- Proper train/test split before encoding/scaling
- Model evaluation using MSE, MAE, and R² Score
- Residual analysis and feature importance extraction

## Key Findings

- **Profitability**: Chilli and Sugarcane generate the highest net profit across seasons
- **Irrigation Efficiency**: Drip and Sprinkler systems achieve significantly higher Water Efficiency compared to Flood irrigation
- **Predictive Power**: Random Forest Regressor achieves R² > 0.95, demonstrating reliable yield prediction using agronomic and climate parameters

## Insights & Next Steps

1. **Transition to Precision Irrigation**: Incentivize Drip and Sprinkler systems across Rabi and Zaid seasons
2. **Targeted Soil & Nutrient Management**: Prioritize soil pH correction and balanced NPK fertilizer application
3. **Crop-Season Portfolio Optimization**: Align high-value cash crops with optimal seasonal windows
4. **Pest & Disease Risk Mitigation**: Implement climate-triggered pest advisories during high-humidity Kharif cycles
5. **Financial Risk Management**: Restructure subsidies and price supports for high-variance crops
6. **Regional Knowledge Transfer**: Disseminate best practices from top-yielding districts to lower-yield regions
7. **Operational Model Deployment**: Integrate the Random Forest model into a real-time decision support tool

## Requirements

The analysis requires the following Python packages:
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- scikit-learn

## Usage

1. Clone the repository
2. Ensure the dataset `seasonal_agriculture_performance_dataset.csv` is in the working directory
3. Open and run the notebook `Seasonal_Agriculture_Performance_Analysis.ipynb`

Alternatively, regenerate the notebook using:
```bash
python create_notebook.py
```

## License

This project is open source and available for educational and research purposes.
