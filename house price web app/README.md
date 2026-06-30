# Pakistan House Price Predictor — Streamlit App

## Setup
```bash
pip install -r requirements.txt
```

## Run
```bash
streamlit run app.py
```
Then open the URL shown in the terminal (usually http://localhost:8501).

## Folder structure
```
webapp/
├── app.py                  # Streamlit app
├── requirements.txt
└── data/
    ├── best_model.pkl       # Trained Random Forest model
    ├── feature_columns.pkl  # Exact feature order expected by the model
    ├── location_lookup.csv  # City -> location -> lat/long/frequency mapping
    ├── predictions.csv      # Saved test-set predictions (for the Insights tab)
    └── model_comparison.csv # MAE/RMSE/R2 for all 3 trained models
```

## Pages
- **Predict a Price**: pick city, neighborhood, property type, area, bedrooms, bathrooms,
  and listing date to get a live price estimate with an approximate error margin.
- **Model Insights**: model comparison table, predicted-vs-actual scatter plot,
  residual distribution, and error-by-price-band chart — same analysis as notebook 4.

## Note
This app's model was trained only on `purpose == 'For Sale'` listings from the
Zameen.com Pakistan House Price dataset. It does not predict rental prices.
