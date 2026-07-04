# Pakistan House Price Prediction

End-to-end machine learning project that predicts real estate sale prices in Pakistan using property features such as area, location, bedrooms, and property type. Built as part of an ML/AI internship assignment.

## Overview

This project covers the full pipeline from raw data to a deployed prediction tool:

1. **Exploratory Data Analysis** — understanding price distribution, location effects, and data quality issues.
2. **Preprocessing & Feature Engineering** — cleaning, unit standardization, encoding, and outlier handling.
3. **Modeling** — training and comparing Linear Regression, Random Forest, and XGBoost.
4. **Evaluation** — visual and statistical comparison of model performance.
5. **Web App** — an interactive Streamlit app for live price prediction and model insight visualization.

## Dataset

[Pakistan House Price Dataset](https://www.kaggle.com/datasets/jillanisofttech/pakistan-house-price-dataset) (Zameen.com listings), via Kaggle. ~168K property listings across 5 Pakistani cities. Not included in this repo due to size — download `zameen-updated.csv` from the link above and place it in `data/`.

## Key Findings

- "For Rent" and "For Sale" listings are on completely different price scales and must be separated before modeling.
- Property prices are heavily right-skewed — modeling `log(price)` significantly improves results.
- Property area is recorded in two different units (Marla and Kanal) and must be unified.
- `location` has very high cardinality (1,500+ unique neighborhoods) — handled with frequency encoding rather than one-hot encoding.

## Results

| Model | MAE (PKR) | RMSE (PKR) | R² |
|---|---|---|---|
| **Random Forest** | 3,253,167 | 7,330,911 | **0.911** |
| XGBoost | 3,562,537 | 7,632,087 | 0.904 |
| Linear Regression | 9,349,664 | 25,670,662 | -0.090 |

Random Forest performed best. Linear Regression underperforms because house prices depend on non-linear interactions between location, area, and property type that a straight-line model can't capture.

## Project Structure

```
pakistan-house-price-prediction/
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory data analysis
│   ├── 02_preprocessing.ipynb    # Cleaning & feature engineering
│   ├── 03_modeling.ipynb         # Model training & comparison
│   └── 04_predictions.ipynb      # Visual evaluation of predictions
├── webapp/
│   ├── app.py                    # Streamlit application
│   ├── requirements.txt
│   ├── README.md                 # Web app specific setup instructions
│   └── data/                     # Trained model + supporting artifacts
└── data/                         # Place zameen-updated.csv here (not included)
```

## Running the Notebooks

```bash
pip install -r notebooks_requirements.txt   # or install pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, jupyter
jupyter notebook notebooks/01_eda.ipynb
```
Run notebooks in order (01 → 04); each one saves outputs used by the next.

## Running the Web App

```bash
cd webapp
pip install -r requirements.txt
streamlit run app.py
```
Then open the printed `localhost:8501` link. The app has two pages:
- **Predict a Price** — enter property details and get a live price estimate.
- **Model Insights** — model comparison table and evaluation visualizations.

## Skills Demonstrated

- Regression modeling (Linear Regression, Random Forest, XGBoost)
- Feature engineering & high-cardinality categorical encoding
- Outlier handling and target transformation (log scale)
- Model evaluation (MAE, RMSE, R²)
- Building and deploying an interactive ML web app with Streamlit

## Author

Saqib