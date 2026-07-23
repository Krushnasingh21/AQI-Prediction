"""
Air Quality Index (AQI) Prediction Dashboard
---------------------------------------------
An interactive Streamlit app that explores India's city-level air quality
dataset and predicts AQI from pollutant concentrations using Linear
Regression, Random Forest, and XGBoost models.

Run with:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="AQI Prediction Dashboard",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = "city_day.csv"  # rename your uploaded file to this, or update the path
DROP_COLUMNS = ["City", "Date", "AQI_Bucket", "NH3"]

AQI_BUCKETS = [
    (0, 50, "Good", "#00e400"),
    (51, 100, "Satisfactory", "#a3ff00"),
    (101, 200, "Moderate", "#ffff00"),
    (201, 300, "Poor", "#ff7e00"),
    (301, 400, "Very Poor", "#ff0000"),
    (401, 10_000, "Severe", "#8f3f97"),
]


def classify_aqi(value: float):
    for low, high, label, color in AQI_BUCKETS:
        if low <= value <= high:
            return label, color
    return "Unknown", "#808080"


# --------------------------------------------------------------------------
# Data loading & preprocessing
# --------------------------------------------------------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


@st.cache_data
def preprocess(df: pd.DataFrame):
    data = df.dropna(subset=["AQI"]).copy()
    numeric_cols = data.select_dtypes(include=["number"]).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
    model_df = data.drop(columns=DROP_COLUMNS, errors="ignore")
    return data, model_df


@st.cache_resource
def train_models(model_df: pd.DataFrame):
    target = "AQI"
    features = [c for c in model_df.columns if c != target]

    X = model_df[features]
    y = model_df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(
            objective="reg:squarederror", n_estimators=100, random_state=42
        ),
    }

    results = {}
    predictions = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        results[name] = {
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "R2 Score": r2_score(y_test, y_pred),
        }
        predictions[name] = y_pred

    results_df = pd.DataFrame(results).T.sort_values("R2 Score", ascending=False)

    return {
        "models": models,
        "scaler": scaler,
        "features": features,
        "results_df": results_df,
        "X_test": X_test,
        "y_test": y_test,
        "predictions": predictions,
    }


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------
def main():
    st.title("🌫️ Air Quality Index (AQI) Prediction Dashboard")
    st.caption(
        "Explore historical pollution trends across Indian cities and predict "
        "AQI from pollutant concentrations using machine learning."
    )

    try:
        raw_df = load_data(DATA_PATH)
    except FileNotFoundError:
        st.error(
            f"Could not find `{DATA_PATH}`. Place the dataset CSV in the same "
            "folder as this app (or update `DATA_PATH` at the top of app.py)."
        )
        st.stop()

    clean_df, model_df = preprocess(raw_df)
    bundle = train_models(model_df)

    features = bundle["features"]
    best_model_name = bundle["results_df"].index[0]

    # ----------------------------------------------------------------
    # Sidebar - inputs
    # ----------------------------------------------------------------
    st.sidebar.header("🔧 Predict AQI")
    st.sidebar.caption(f"Best performing model: **{best_model_name}**")

    model_choice = st.sidebar.selectbox(
        "Model", list(bundle["models"].keys()), index=list(bundle["models"].keys()).index(best_model_name)
    )

    st.sidebar.subheader("Pollutant levels")
    input_values = {}
    for col in features:
        col_min = float(clean_df[col].min())
        col_max = float(clean_df[col].max())
        col_mean = float(clean_df[col].mean())
        input_values[col] = st.sidebar.slider(
            col, min_value=round(col_min, 2), max_value=round(col_max, 2),
            value=round(col_mean, 2),
        )

    predict_btn = st.sidebar.button("Predict AQI", use_container_width=True)

    st.sidebar.divider()
    st.sidebar.subheader("Explore by city")
    city_options = sorted(raw_df["City"].dropna().unique().tolist())
    selected_city = st.sidebar.selectbox("City", ["All Cities"] + city_options)

    # ----------------------------------------------------------------
    # Tabs
    # ----------------------------------------------------------------
    tab_predict, tab_explore, tab_models = st.tabs(
        ["🔮 Prediction", "📊 Data Exploration", "🤖 Model Performance"]
    )

    # ---------------- Prediction tab ----------------
    with tab_predict:
        col1, col2 = st.columns([1, 1.3])

        with col1:
            st.subheader("Input summary")
            st.dataframe(
                pd.DataFrame([input_values]).T.rename(columns={0: "Value"}),
                use_container_width=True,
            )

        with col2:
            st.subheader("Prediction result")
            if predict_btn:
                model = bundle["models"][model_choice]
                scaler = bundle["scaler"]
                x_input = pd.DataFrame([input_values])[features]
                x_scaled = scaler.transform(x_input)
                pred_aqi = float(model.predict(x_scaled)[0])
                label, color = classify_aqi(pred_aqi)

                st.markdown(
                    f"""
                    <div style="padding: 1.5rem; border-radius: 0.75rem;
                                background-color:{color}22; border: 1px solid {color};">
                        <h2 style="margin:0;">Predicted AQI: {pred_aqi:.1f}</h2>
                        <h4 style="margin:0; color:{color};">Category: {label}</h4>
                        <p style="margin-top:0.5rem;">Model used: {model_choice}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.info("Set pollutant levels in the sidebar and click **Predict AQI**.")

        st.divider()
        st.subheader("AQI category reference")
        legend_cols = st.columns(len(AQI_BUCKETS))
        for c, (low, high, label, color) in zip(legend_cols, AQI_BUCKETS):
            c.markdown(
                f"<div style='background-color:{color}; padding:0.5rem; "
                f"border-radius:0.5rem; text-align:center; color:black;'>"
                f"<b>{label}</b><br>{low}-{high if high < 10000 else '400+'}</div>",
                unsafe_allow_html=True,
            )

    # ---------------- Data exploration tab ----------------
    with tab_explore:
        df_view = raw_df if selected_city == "All Cities" else raw_df[raw_df["City"] == selected_city]

        st.subheader(f"Dataset overview — {selected_city}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Records", f"{len(df_view):,}")
        c2.metric("Cities", df_view["City"].nunique())
        c3.metric("Avg AQI", f"{df_view['AQI'].mean():.1f}" if df_view["AQI"].notna().any() else "N/A")
        c4.metric("Date range", f"{df_view['Date'].min():%Y} - {df_view['Date'].max():%Y}" if df_view["Date"].notna().any() else "N/A")

        st.markdown("#### AQI trend over time")
        trend = df_view.dropna(subset=["AQI"]).groupby(pd.Grouper(key="Date", freq="ME"))["AQI"].mean()
        if not trend.empty:
            st.line_chart(trend)
        else:
            st.info("No AQI data available for this selection.")

        st.markdown("#### Pollutant correlation heatmap")
        num_df = model_df.select_dtypes(include=["number"])
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, ax=ax)
        st.pyplot(fig)

        st.markdown("#### Raw data sample")
        st.dataframe(df_view.head(200), use_container_width=True)

    # ---------------- Model performance tab ----------------
    with tab_models:
        st.subheader("Model comparison")
        st.dataframe(
            bundle["results_df"].style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "R2 Score": "{:.3f}"}),
            use_container_width=True,
        )

        fig, ax = plt.subplots(figsize=(9, 4.5))
        bundle["results_df"].plot(kind="bar", ax=ax)
        ax.set_ylabel("Score")
        ax.set_title("Model Comparison")
        plt.xticks(rotation=0)
        st.pyplot(fig)

        st.markdown(f"#### Feature importance ({model_choice})")
        model = bundle["models"][model_choice]
        if hasattr(model, "feature_importances_"):
            importance = pd.Series(model.feature_importances_, index=features).sort_values()
            fig2, ax2 = plt.subplots(figsize=(9, 5))
            importance.plot(kind="barh", ax=ax2, color="#4c72b0")
            ax2.set_title(f"Feature Importance — {model_choice}")
            st.pyplot(fig2)
        else:
            coef = pd.Series(model.coef_, index=features).sort_values()
            fig2, ax2 = plt.subplots(figsize=(9, 5))
            coef.plot(kind="barh", ax=ax2, color="#4c72b0")
            ax2.set_title(f"Coefficients — {model_choice}")
            st.pyplot(fig2)

        st.markdown("#### Actual vs. Predicted AQI (test sample)")
        n = st.slider("Number of samples to compare", 5, 50, 15)
        y_test = bundle["y_test"].reset_index(drop=True)
        y_pred = bundle["predictions"][model_choice]
        comp_df = pd.DataFrame({"Actual AQI": y_test[:n], "Predicted AQI": y_pred[:n]})
        st.line_chart(comp_df)


if __name__ == "__main__":
    main()
