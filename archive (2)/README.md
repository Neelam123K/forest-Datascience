# Forest Fire Risk and Burned Area Estimator

A Machine Learning project that predicts the estimated burned area of a forest fire using weather conditions and fire weather indices. The project also classifies the predicted burned area into risk levels and provides actionable recommendations for forest management.

---

## Project Overview

Forest fires can cause significant environmental and economic damage. Early estimation of the potential burned area helps authorities allocate resources and take preventive measures.

This project develops a regression-based machine learning model that:

- Predicts the expected burned area (in hectares)
- Classifies fire severity into risk categories
- Recommends suitable actions based on the predicted risk

---

## Problem Statement

Develop a machine learning model capable of estimating the burned area caused by forest fires using weather conditions and fire weather indices.

---

## Dataset

**Dataset:** Forest Fires Dataset (UCI Machine Learning Repository)

### Features

| Feature | Description |
|---------|-------------|
| X, Y | Spatial coordinates |
| month | Month of observation |
| day | Day of week |
| FFMC | Fine Fuel Moisture Code |
| DMC | Duff Moisture Code |
| DC | Drought Code |
| ISI | Initial Spread Index |
| temp | Temperature (°C) |
| RH | Relative Humidity (%) |
| wind | Wind Speed |
| rain | Rainfall |
| area | Burned Area (Target Variable) |

---

## Project Workflow

### 1. Data Cleaning

- Verified data types
- Checked categorical values
- Confirmed there were no missing values or inconsistencies

### 2. Exploratory Data Analysis (EDA)

The following analyses were performed:

- Distribution analysis
- Correlation analysis
- Histograms
- Boxplots
- Scatter plots
- Feature relationship analysis

#### Key Findings

- The burned area is highly right-skewed.
- Most forest fires affect very small areas.
- Individual features have weak linear relationships with the target variable.
- Rainfall contributes very little to prediction because most observations recorded no rainfall.
- The problem is better suited to nonlinear machine learning models.

### 3. Target Variable Transformation

Since the target variable was highly skewed, a logarithmic transformation was applied.

```python
df["log_area"] = np.log1p(df["area"])
```

This transformation:

- Reduces skewness
- Compresses extreme values
- Improves regression model performance

### 4. Feature Engineering

Categorical variables (`month` and `day`) were converted into numerical features using One-Hot Encoding.

```python
pd.get_dummies(df, columns=["month", "day"], drop_first=True)
```

### 5. Train-Test Split

- Training Set: 80%
- Testing Set: 20%
- Random State: 42

---

## Machine Learning Models

### Linear Regression

Linear Regression was used as the baseline model.

#### Performance

| Metric | Value |
|---------|------:|
| MAE | 1.20 |
| RMSE | 1.51 |
| R² Score | -0.04 |

### Random Forest Regressor

Hyperparameters used:

- Number of Trees: 300
- Maximum Depth: 10
- Minimum Samples Split: 5
- Minimum Samples Leaf: 2

#### Performance

| Metric | Value |
|---------|------:|
| MAE | 1.198 |
| RMSE | 1.503 |
| R² Score | -0.028 |

The Random Forest Regressor achieved slightly better performance and was selected as the final model.

---

## Model Comparison

| Model | MAE | RMSE | R² Score |
|---------|------:|------:|------:|
| Linear Regression | 1.200 | 1.510 | -0.040 |
| Random Forest Regressor | 1.198 | 1.503 | -0.028 |

---

## Risk Classification

The predicted burned area is categorized into different risk levels.

| Predicted Burned Area | Risk Level |
|----------------------:|------------|
| Less than 1 hectare | Low |
| 1–10 hectares | Moderate |
| 10–50 hectares | High |
| More than 50 hectares | Extreme |

---

## Action Recommendation System

Each risk category is associated with a recommended action.

| Risk Level | Recommended Action |
|------------|--------------------|
| Low | Continue routine monitoring of the forest area. |
| Moderate | Increase surveillance and keep firefighting resources on standby. |
| High | Deploy firefighting teams and restrict access to vulnerable areas. |
| Extreme | Initiate emergency response procedures, deploy all available firefighting resources, and consider evacuation if necessary. |

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## Results

- Performed data preprocessing and feature engineering.
- Conducted comprehensive exploratory data analysis.
- Built and evaluated Linear Regression and Random Forest regression models.
- Developed a risk classification system based on predicted burned area.
- Created an action recommendation system to support forest fire management.
- Selected Random Forest Regressor as the final model based on evaluation metrics.

---

## Future Improvements

- Integrate real-time weather data.
- Include satellite imagery and vegetation information.
- Experiment with advanced models such as XGBoost and Gradient Boosting.
- Deploy the model using Flask or Streamlit.
- Incorporate additional environmental variables to improve prediction accuracy.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Forest-Fire-Risk-Estimator.git
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

## Required Libraries

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

---

## How to Run

```bash
git clone https://github.com/your-username/Forest-Fire-Risk-Estimator.git

cd Forest-Fire-Risk-Estimator

pip install -r requirements.txt

jupyter notebook
```

Open the notebook and run all cells sequentially.

---

## Author

**Udit Kaushik**

B.Tech Computer Science Engineering

Data Science and Machine Learning Enthusiast
```