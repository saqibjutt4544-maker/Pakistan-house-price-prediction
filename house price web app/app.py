import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Pakistan House Price Predictor", layout="wide")

# ---------------------------------------------------------------------------
# Load artifacts (cached so they only load once per session)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("data/best_model.pkl")
    feature_cols = joblib.load("data/feature_columns.pkl")
    location_lookup = pd.read_csv("data/location_lookup.csv")
    predictions = pd.read_csv("data/predictions.csv")
    comparison = pd.read_csv("data/model_comparison.csv", index_col=0)
    return model, feature_cols, location_lookup, predictions, comparison

model, feature_cols, location_lookup, predictions, comparison = load_artifacts()

PROPERTY_TYPES = ["Farm House", "Flat", "House", "Lower Portion", "Penthouse", "Room", "Upper Portion"]
CITY_PROVINCE = {
    "Faisalabad": "Punjab",
    "Islamabad": "Islamabad Capital",
    "Karachi": "Sindh",
    "Lahore": "Punjab",
    "Rawalpindi": "Punjab",
}
CITIES = list(CITY_PROVINCE.keys())

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("🏠 House Price Predictor")
page = st.sidebar.radio("Go to", ["Predict a Price", "Model Insights"])

best_model_name = comparison["RMSE"].astype(float).idxmin()

# ---------------------------------------------------------------------------
# Helper: build a feature row matching training-time encoding
# ---------------------------------------------------------------------------
def build_feature_row(city, location_row, property_type, baths, bedrooms,
                       area_marla, listing_year, listing_month):
    row = {col: 0 for col in feature_cols}

    row["latitude"] = location_row["latitude"]
    row["longitude"] = location_row["longitude"]
    row["baths"] = baths
    row["bedrooms"] = bedrooms
    row["area_marla"] = area_marla
    row["listing_year"] = listing_year
    row["listing_month"] = listing_month
    row["location_freq"] = location_row["location_freq"]

    pt_col = f"property_type_{property_type}"
    if pt_col in row:
        row[pt_col] = 1

    city_col = f"city_{city}"
    if city_col in row:
        row[city_col] = 1

    province = CITY_PROVINCE[city]
    prov_col = f"province_name_{province}"
    if prov_col in row:
        row[prov_col] = 1

    return pd.DataFrame([row])[feature_cols]


# ---------------------------------------------------------------------------
# PAGE 1: Predict a Price
# ---------------------------------------------------------------------------
if page == "Predict a Price":
    st.title("Predict a House Price")
    st.caption("Estimate a Pakistani property's sale price based on its features. "
               "Model trained on Zameen.com listings (For Sale only).")

    col1, col2 = st.columns(2)

    with col1:
        city = st.selectbox("City", CITIES, index=CITIES.index("Lahore"))

        city_locations = location_lookup[location_lookup["city"] == city].sort_values("count", ascending=False)
        location = st.selectbox("Location / Neighborhood", city_locations["location"].tolist())

        property_type = st.selectbox("Property Type", PROPERTY_TYPES, index=PROPERTY_TYPES.index("House"))

    with col2:
        area_marla = st.number_input("Area (Marla)", min_value=1.0, max_value=2000.0, value=10.0, step=0.5,
                                      help="1 Kanal = 20 Marla, so for large plots just multiply.")
        bedrooms = st.number_input("Bedrooms", min_value=0, max_value=15, value=3, step=1)
        baths = st.number_input("Bathrooms", min_value=0, max_value=15, value=3, step=1)

    listing_year = st.slider("Listing Year", min_value=2018, max_value=2026, value=2026)
    listing_month = st.slider("Listing Month", min_value=1, max_value=12, value=6)

    if st.button("Predict Price", type="primary"):
        loc_row = city_locations[city_locations["location"] == location].iloc[0]

        X_input = build_feature_row(
            city=city, location_row=loc_row, property_type=property_type,
            baths=baths, bedrooms=bedrooms, area_marla=area_marla,
            listing_year=listing_year, listing_month=listing_month
        )

        pred_log = model.predict(X_input)[0]
        pred_price = max(0, np.expm1(pred_log))

        st.success(f"### Estimated Price: PKR {pred_price:,.0f}")

        # Show a rough confidence range using the test-set MAE for the model used
        mae = comparison.loc[best_model_name, "MAE"]
        st.caption(f"Typical model error (MAE) is about PKR {mae:,.0f}, "
                   f"so the true price likely falls roughly between "
                   f"PKR {max(0, pred_price - mae):,.0f} and PKR {pred_price + mae:,.0f}.")

        with st.expander("See the exact feature values sent to the model"):
            st.dataframe(X_input.T.rename(columns={0: "value"}))

# ---------------------------------------------------------------------------
# PAGE 2: Model Insights (visualizations from notebook 4)
# ---------------------------------------------------------------------------
else:
    st.title("Model Insights & Performance")

    st.subheader("Model Comparison")
    st.dataframe(
        comparison.style.format({"MAE": "{:,.0f}", "RMSE": "{:,.0f}", "R2": "{:.4f}"})
        .highlight_min(subset=["MAE", "RMSE"], color="lightgreen")
        .highlight_max(subset=["R2"], color="lightgreen")
    )
    st.caption(f"Best model by RMSE: **{best_model_name}**")

    col_map = {
        "Linear Regression": "linear_regression_pred",
        "Random Forest": "random_forest_pred",
        "XGBoost": "xgboost_pred",
    }
    best_col = col_map.get(best_model_name, "random_forest_pred")

    st.subheader("Predicted vs Actual Price")
    fig, ax = plt.subplots(figsize=(7, 7))
    sample = predictions.sample(min(5000, len(predictions)), random_state=1)
    ax.scatter(sample["actual_price"], sample[best_col], alpha=0.3, s=10)
    max_val = max(sample["actual_price"].max(), sample[best_col].max())
    ax.plot([0, max_val], [0, max_val], "r--", label="Perfect prediction")
    ax.set_xlabel("Actual Price (PKR)")
    ax.set_ylabel("Predicted Price (PKR)")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Error Distribution")
    predictions["residual"] = predictions["actual_price"] - predictions[best_col]
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    sns.histplot(predictions["residual"], bins=80, ax=ax2)
    ax2.axvline(0, color="red", linestyle="--")
    ax2.set_xlabel("Residual (Actual - Predicted, PKR)")
    st.pyplot(fig2)

    st.subheader("Error by Price Band")
    predictions["pct_error"] = 100 * predictions["residual"] / predictions["actual_price"]
    predictions["price_band"] = pd.qcut(
        predictions["actual_price"], q=5,
        labels=["Lowest 20%", "Low-Mid 20%", "Mid 20%", "Mid-High 20%", "Highest 20%"]
    )
    band_error = predictions.groupby("price_band", observed=True)["pct_error"].apply(
        lambda x: np.mean(np.abs(x))
    )
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    band_error.plot(kind="bar", ax=ax3, color="coral")
    ax3.set_ylabel("Mean Absolute % Error")
    plt.xticks(rotation=20)
    st.pyplot(fig3)

    st.caption("Lower-priced properties tend to show higher percentage error; "
               "higher-priced properties show larger absolute error but lower percentage error.")
