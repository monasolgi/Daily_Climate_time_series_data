# Weather Forecasting using Feature Engineering and XGBoost

## Project Overview

This project predicts daily mean temperature using historical weather observations. The goal is to compare machine learning approaches for multivariate time-series forecasting and investigate the impact of lag features, rolling statistics, and weather-related predictors.

Dataset features include:

- Mean Temperature
- Humidity
- Wind Speed
- Mean Pressure

**Target Variable:** Mean Temperature (`meantemp`)

---

## Dataset

**Dataset:** Daily Climate Time Series Data

**Time Period:** 2013–2017

### Variables

| Feature | Description |
|----------|----------|
| meantemp | Daily mean temperature |
| humidity | Daily humidity |
| wind_speed | Daily wind speed |
| meanpressure | Daily atmospheric pressure |

---

## Data Cleaning

### Datetime Index

The date column was converted to a datetime index to enable time-series operations.

### Outlier Detection

Exploratory analysis revealed unrealistic pressure values such as:

- -3
- 12
- 310
- 7679

These values are physically impossible for atmospheric pressure and were treated as sensor errors.

### Outlier Handling

Outliers were identified using the Interquartile Range (IQR) method:

- Lower Bound = Q1 − 1.5 × IQR
- Upper Bound = Q3 + 1.5 × IQR

Detected outliers were replaced with NaN and interpolated using time-based interpolation.

---

## Exploratory Data Analysis

### Key Observations

- Mean temperature exhibits strong yearly seasonality.
- Humidity shows cyclical behavior.
- Mean pressure has a strong inverse relationship with temperature.
- Wind speed is substantially noisier than the other variables.

### Correlation Analysis

| Variable | Correlation with Temperature |
|----------|----------|
| Humidity | -0.57 |
| Wind Speed | 0.31 |
| Mean Pressure | -0.88 |

---

## Feature Engineering

### Temperature Lag Features

Historical temperature information was incorporated using:

- lag_1
- lag_7
- lag_14
- lag_30

These features provide short-term and seasonal memory.

### Rolling Statistics

Rolling windows were used to capture local trends and variability:

- rolling_mean_7
- rolling_mean_30
- rolling_std_7

### Weather Lag Features

To prevent data leakage, only past weather observations were used:

- humidity_lag_1
- wind_speed_lag_1
- meanpressure_lag_1

Same-day weather variables were excluded.

### Calendar Features

- day_of_week
- day_of_year
- month

---

## Data Leakage Prevention

An initial model achieved unrealistically high performance (R² ≈ 0.999).

Investigation revealed that same-day weather measurements were being used to predict same-day temperature.

To create a genuine forecasting task:

- Same-day weather variables were removed.
- Lagged weather features were introduced.
- Only historical information was allowed.

---

## Model

### XGBoost Regressor

```python
XGBRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    random_state=42
)
```

---

## Validation Strategy

A standard random train/test split is inappropriate for time-series data.

Instead, TimeSeriesSplit with 5 folds was used to ensure that future observations never appear in the training set.

---

## Cross-Validation Results

| Fold | R² |
|------|------|
| 1 | 0.779 |
| 2 | 0.904 |
| 3 | 0.951 |
| 4 | 0.960 |
| 5 | 0.839 |

### Average Performance

- MAE = 1.46
- RMSE = 1.87
- R² = 0.887

---

## Feature Importance

XGBoost identified the following features as most useful:

- Temperature lag features
- Rolling temperature averages
- Atmospheric pressure
- Seasonal calendar variables

These results indicate that historical temperature patterns and atmospheric pressure provide strong predictive information.

---

## Key Concepts Learned

This project demonstrates:

- Time-series preprocessing
- Datetime indexing
- Outlier detection and treatment
- Time-based interpolation
- Multivariate forecasting
- Lag feature engineering
- Rolling-window statistics
- Data leakage prevention
- TimeSeriesSplit cross-validation
- XGBoost forecasting
- Feature importance analysis

---

## Future Improvements

Potential extensions include:

- Prophet forecasting
- ARIMA/SARIMA models
- LSTM-based forecasting
- Hyperparameter tuning
- Additional weather lag features
- Multi-step forecasting
