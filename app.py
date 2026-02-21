import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from datetime import datetime, timedelta
import os

# ── PAGE CONFIG 
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS 
st.markdown(
    """
    <style>
    h1 { color: #1f77b4; padding-bottom: 0.5rem; }
    h2 { color: #2c3e50; padding-top: 0.8rem; }
    .metric-card {
        background: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── HELPER FUNCTIONS 
@st.cache_data
def load_data():
    """
    Load and feature-engineer the historical sales data.
    Reads from sales_data_clean.csv and applies the same per-product-region
    lag/rolling feature engineering as Notebook 4, so column names always
    match what the trained model expects.
    """
    try:
        df = pd.read_csv("data/processed/sales_data_clean.csv")
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values(["Product", "Region", "Date"]).reset_index(drop=True)

        # Time features
        df["Year"]           = df["Date"].dt.year
        df["Month"]          = df["Date"].dt.month
        df["Day"]            = df["Date"].dt.day
        df["DayOfWeek"]      = df["Date"].dt.dayofweek
        df["Quarter"]        = df["Date"].dt.quarter
        df["DayOfYear"]      = df["Date"].dt.dayofyear
        df["WeekOfYear"]     = df["Date"].dt.isocalendar().week.astype(int)

        # Binary flags
        df["Is_Weekend"]     = (df["DayOfWeek"] >= 5).astype(int)
        df["Is_Monday"]      = (df["DayOfWeek"] == 0).astype(int)
        df["Is_Friday"]      = (df["DayOfWeek"] == 4).astype(int)
        df["Is_Month_Start"] = (df["Day"] <= 5).astype(int)
        df["Is_Month_End"]   = (df["Day"] >= 25).astype(int)
        df["Is_Q4"]          = (df["Quarter"] == 4).astype(int)
        df["Is_December"]    = (df["Month"] == 12).astype(int)

        # Trend
        df["Days_Since_Start"] = (df["Date"] - df["Date"].min()).dt.days

        # Per-product-region lag & rolling features (matches NB04)
        grp = df.groupby(["Product", "Region"])["Sales"]
        df["Lag_1"]  = grp.shift(1)
        df["Lag_7"]  = grp.shift(7)
        df["Lag_14"] = grp.shift(14)
        df["Lag_30"] = grp.shift(30)
        df["MA_7"]   = grp.transform(lambda x: x.shift(1).rolling(7,  min_periods=1).mean())
        df["MA_14"]  = grp.transform(lambda x: x.shift(1).rolling(14, min_periods=1).mean())
        df["MA_30"]  = grp.transform(lambda x: x.shift(1).rolling(30, min_periods=1).mean())
        df["Std_7"]  = grp.transform(lambda x: x.shift(1).rolling(7,  min_periods=2).std())

        return df
    except FileNotFoundError:
        st.error(
            "⚠️ Data file not found!  "
            "Make sure `data/processed/sales_data_clean.csv` exists "
            "(run Notebook 2 first)."
        )
        return None


@st.cache_resource
def load_model():
    """Load trained model package."""
    try:
        with open("data/models/sales_forecast_model.pkl", "rb") as f:
            pkg = pickle.load(f)
        return pkg
    except FileNotFoundError:
        st.error(
            "⚠️ Model file not found!  "
            "Please train the model first (Notebook 4)."
        )
        return None


def create_future_features(future_dates, df_historical, feature_names=None):
    """
    Build a feature DataFrame for future dates using the same
    engineering steps as Notebook 4 (per-product-region lags are
    approximated with the historical column mean).
    """
    fdf = pd.DataFrame({"Date": future_dates})

    # Calendar features
    fdf["Year"]             = fdf["Date"].dt.year
    fdf["Month"]            = fdf["Date"].dt.month
    fdf["Day"]              = fdf["Date"].dt.day
    fdf["DayOfWeek"]        = fdf["Date"].dt.dayofweek
    fdf["Quarter"]          = fdf["Date"].dt.quarter
    fdf["DayOfYear"]        = fdf["Date"].dt.dayofyear
    fdf["WeekOfYear"]       = fdf["Date"].dt.isocalendar().week.astype(int)

    # Binary flags
    fdf["Is_Weekend"]       = (fdf["DayOfWeek"] >= 5).astype(int)
    fdf["Is_Monday"]        = (fdf["DayOfWeek"] == 0).astype(int)
    fdf["Is_Friday"]        = (fdf["DayOfWeek"] == 4).astype(int)
    fdf["Is_Month_Start"]   = (fdf["Day"] <= 5).astype(int)
    fdf["Is_Month_End"]     = (fdf["Day"] >= 25).astype(int)
    fdf["Is_Q4"]            = (fdf["Quarter"] == 4).astype(int)
    fdf["Is_December"]      = (fdf["Month"] == 12).astype(int)

    # Trend
    fdf["Days_Since_Start"] = (fdf["Date"] - df_historical["Date"].min()).dt.days

    # Lag / rolling features — filled with historical column mean as approximation
    lag_rolling_cols = ["Lag_1", "Lag_7", "Lag_14", "Lag_30",
                        "MA_7",  "MA_14", "MA_30",  "Std_7"]
    for col in lag_rolling_cols:
        if col in df_historical.columns:
            fdf[col] = df_historical[col].mean()

    # Any remaining columns the model expects (e.g. one-hot category dummies)
    if feature_names:
        for feat in feature_names:
            if feat not in fdf.columns:
                fdf[feat] = df_historical[feat].mean() if feat in df_historical.columns else 0.0

    return fdf

# ── LOAD DATA & MODEL 
df            = load_data()
model_package = load_model()

if df is None or model_package is None:
    st.warning("Please fix the missing files above, then refresh the page.")
    st.stop()

model         = model_package["model"]
scaler        = model_package.get("scaler", None)
feature_names = model_package["feature_names"]
test_r2       = model_package["test_r2"]
test_mae      = model_package["test_mae"]

test_rmse = model_package.get("test_rmse", np.sqrt(test_mae ** 2))

# ── SIDEBAR 
st.sidebar.title("📊 Navigation")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Select Page:",
    [
        "🏠 Home",
        "📈 Data Explorer",
        "🎯 Model Performance",
        "🔮 Forecast Generator",
        "🔍 Feature Analysis",
        "📥 Download Reports",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "**Sales Forecasting Dashboard**\n\n"
    "Linear Regression · Python · Streamlit"
)

# ── PAGE 1 — HOME 
if page == "🏠 Home":
    st.title("📊 Sales Forecasting Dashboard")
    st.markdown("### Welcome!  Predict future sales with machine learning.")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📅 Historical Data",  f"{len(df):,} days",
              f"{df['Date'].min().date()} → {df['Date'].max().date()}")
    c2.metric("💰 Avg Daily Sales",   f"${df['Sales'].mean():,.2f}",
              f"±${df['Sales'].std():,.2f}")
    c3.metric("🎯 Model Accuracy",    f"{test_r2*100:.1f}%",
              f"MAE: ${test_mae:,.2f}")
    c4.metric("🔧 Features Used",     str(len(feature_names)), "Engineered")

    st.markdown("---")
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### 🎯 What This Dashboard Does")
        st.markdown("""
- **Explore** historical sales data
- **Analyse** model performance and accuracy
- **Generate** forecasts for future dates
- **Understand** which factors drive sales
- **Download** reports and predictions
""")

    with col_right:
        st.markdown("### 🚀 Quick Start Guide")
        st.markdown("""
1. **Data Explorer** — view your historical data
2. **Model Performance** — check accuracy metrics
3. **Forecast Generator** — create predictions
4. **Feature Analysis** — see what matters most
5. **Download Reports** — export your results
""")

    st.markdown("---")
    st.markdown("### 📈 Sales Trend Overview (Last 6 Months)")

    last6 = df[df["Date"] >= df["Date"].max() - timedelta(days=180)].copy()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(last6["Date"], last6["Sales"],
            color="steelblue", linewidth=1.8, alpha=0.7, label="Daily Sales")
    ax.plot(last6["Date"], last6["Sales"].rolling(30).mean(),
            color="red", linewidth=2, linestyle="--", label="30-day MA")
    ax.set_xlabel("Date"); ax.set_ylabel("Sales ($)")
    ax.set_title("Sales Trend — Last 6 Months", fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")
    st.markdown("### ℹ️ Model Information")
    mi1, mi2 = st.columns(2)
    with mi1:
        st.markdown(f"""
**Model:** Linear Regression  
**Trained:** {model_package['train_date_range'][0].date()} → {model_package['train_date_range'][1].date()}  
**R² Score:** {test_r2:.4f}  
**MAE:** ${test_mae:,.2f}  
""")
    with mi2:
        st.markdown(f"""
**Features:** {len(feature_names)} engineered  
- Time-based (Year, Month, DayOfWeek …)  
- Binary flags (Is_Weekend, Is_Q4 …)  
- Trend (Days_Since_Start)  
- Lags (Lag 1 / 7 / 14 / 30 days)  
- Rolling averages (MA 7 / 14 / 30-day, Std 7-day)  
""")


# ── PAGE 2 — DATA EXPLORER 
elif page == "📈 Data Explorer":
    st.title("📈 Data Explorer")
    st.markdown("### Explore your historical sales data")
    st.markdown("---")

    d1, d2 = st.columns(2)
    with d1:
        start_date = st.date_input(
            "Start Date",
            value=df["Date"].min().date(),
            min_value=df["Date"].min().date(),
            max_value=df["Date"].max().date(),
            key="explorer_start",     
        )
    with d2:
        end_date = st.date_input(
            "End Date",
            value=df["Date"].max().date(),
            min_value=df["Date"].min().date(),
            max_value=df["Date"].max().date(),
            key="explorer_end",       
        )

    mask = (df["Date"].dt.date >= start_date) & (df["Date"].dt.date <= end_date)
    filt = df[mask].copy()
    st.markdown(f"**Showing {len(filt):,} days of data**")

    st.markdown("### 📊 Summary Statistics")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Average",  f"${filt['Sales'].mean():,.2f}")
    s2.metric("Minimum",  f"${filt['Sales'].min():,.2f}")
    s3.metric("Maximum",  f"${filt['Sales'].max():,.2f}")
    s4.metric("Total",    f"${filt['Sales'].sum():,.2f}")

    st.markdown("---")
    st.markdown("### 📈 Sales Over Time")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(filt["Date"], filt["Sales"], color="steelblue", linewidth=1.5, alpha=0.7)
    ax.set_xlabel("Date"); ax.set_ylabel("Sales ($)")
    ax.set_title("Daily Sales", fontweight="bold")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig); plt.close(fig) 

    st.markdown("---")
    pc1, pc2 = st.columns(2)

    with pc1:
        st.markdown("### 📊 Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(filt["Sales"], bins=50, color="steelblue", edgecolor="black", alpha=0.7)
        ax.set_xlabel("Sales ($)"); ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Daily Sales")
        ax.grid(True, alpha=0.3, axis="y")
        st.pyplot(fig); plt.close(fig)  

    with pc2:
        st.markdown("### 📅 Day-of-Week Analysis")
        day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        filt["Day_Name"] = filt["Date"].dt.day_name()
        day_avg = filt.groupby("Day_Name")["Sales"].mean().reindex(day_order)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(range(7), day_avg.values, color="coral", edgecolor="black", alpha=0.7)
        ax.set_xticks(range(7))
        ax.set_xticklabels([d[:3] for d in day_order], rotation=45)
        ax.set_ylabel("Average Sales ($)")
        ax.set_title("Avg Sales by Day of Week")
        ax.grid(True, alpha=0.3, axis="y")
        st.pyplot(fig); plt.close(fig) 

    st.markdown("---")
    st.markdown("### 📋 Raw Data")
    show_all = st.checkbox("Show all columns", value=False)
    display_df = filt if show_all else filt[["Date","Sales","Month","DayOfWeek","Is_Weekend"]]
    st.dataframe(
        display_df.sort_values("Date", ascending=False).head(100),
        use_container_width=True,
    )

    csv_bytes = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Data as CSV",
        data=csv_bytes,
        file_name=f"sales_data_{start_date}_to_{end_date}.csv",
        mime="text/csv",
        key="download_explorer_csv",   
    )

# ── PAGE 3 — MODEL PERFORMANCE 
elif page == "🎯 Model Performance":
    st.title("🎯 Model Performance")
    st.markdown("### Evaluate model accuracy and reliability")
    st.markdown("---")

    st.markdown("### 📊 Performance Metrics")
    m1, m2, m3 = st.columns(3)
    m1.metric(
        "R² Score", f"{test_r2:.4f}",
        f"{test_r2*100:.1f}% variance explained",
        help="1.0 = perfect. 0.0 = same as guessing the mean.",
    )
    m2.metric(
        "MAE", f"${test_mae:,.2f}",
        f"{(test_mae/df['Sales'].mean())*100:.1f}% of avg sales",
        help="Average absolute prediction error in dollars.",
    )
    m3.metric(
        "RMSE", f"${test_rmse:,.2f}",
        "Penalises large errors more",
        help="Root Mean Squared Error.",
    )

    st.markdown("---")
    st.markdown("### 📝 Quality Assessment")

    if test_r2 > 0.90:
        quality, colour = "🌟 EXCELLENT",    "green"
    elif test_r2 > 0.80:
        quality, colour = "✨ VERY GOOD",    "blue"
    elif test_r2 > 0.70:
        quality, colour = "👍 GOOD",         "orange"
    else:
        quality, colour = "⚠️ NEEDS WORK",   "red"

    st.markdown(f"**Overall Quality:** :{colour}[{quality}]")
    st.markdown(f"""
- Model explains **{test_r2*100:.1f}%** of daily sales variation  
- Average prediction error: **${test_mae:,.2f}**  
- That is **{(test_mae/df['Sales'].mean())*100:.1f}%** of the average daily sale  
""")

    st.markdown("---")
    st.markdown("### 🔍 Top 10 Most Important Features")

    fi = (
        pd.DataFrame({"Feature": feature_names, "Coefficient": model.coef_})
        .assign(Abs=lambda d: d["Coefficient"].abs())
        .sort_values("Abs", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    colours = ["green" if c > 0 else "red" for c in fi["Coefficient"]]
    ax.barh(range(len(fi)), fi["Abs"], color=colours, alpha=0.7, edgecolor="black")
    ax.set_yticks(range(len(fi)))
    ax.set_yticklabels(fi["Feature"])
    ax.set_xlabel("Absolute Coefficient Value")
    ax.set_title("Top 10 Most Important Features", fontweight="bold")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")
    ax.legend(
        handles=[
            Patch(facecolor="green", alpha=0.7, label="Increases Sales"),
            Patch(facecolor="red",   alpha=0.7, label="Decreases Sales"),
        ],
        loc="lower right",
    )
    st.pyplot(fig); plt.close(fig)

# ── PAGE 4 — FORECAST GENERATOR 
elif page == "🔮 Forecast Generator":
    st.title("🔮 Forecast Generator")
    st.markdown("### Generate future sales predictions")
    st.markdown("---")

    st.markdown("### ⚙️ Forecast Settings")
    fc1, fc2 = st.columns(2)

    with fc1:
        forecast_days = st.slider(
            "Number of days to forecast",
            min_value=7, max_value=365, value=180, step=7,
            key="forecast_slider",     
        )

    with fc2:
        last_date = df["Date"].max()
        st.markdown(f"""
**Forecast Period:**  
- {forecast_days} days (~{forecast_days // 30} months)  
- From **{(last_date + timedelta(days=1)).date()}**  
- To   **{(last_date + timedelta(days=forecast_days)).date()}**  
""")

    if "last_forecast_days" not in st.session_state:
        st.session_state["last_forecast_days"] = forecast_days

    if st.session_state["last_forecast_days"] != forecast_days:
        st.session_state.pop("forecast_df", None)
        st.session_state.pop("predictions", None)
        st.session_state["last_forecast_days"] = forecast_days

    if st.button("🚀 Generate Forecast", type="primary", key="gen_forecast_btn"):
        with st.spinner("Generating forecast …"):
            future_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=forecast_days, freq="D",
            )
            fdf = create_future_features(future_dates, df, feature_names=feature_names)
            X_future = fdf[feature_names]
            if scaler is not None:
                X_future = scaler.transform(X_future)
            preds = model.predict(X_future)
            fdf["Predicted_Sales"] = preds
            st.session_state["forecast_df"]  = fdf
            st.session_state["predictions"]  = preds
        st.success("✅ Forecast generated successfully!")

    if "forecast_df" in st.session_state:
        fdf   = st.session_state["forecast_df"]
        preds = st.session_state["predictions"]

        st.markdown("---")
        st.markdown("### 📊 Forecast Summary")
        fs1, fs2, fs3, fs4 = st.columns(4)
        fs1.metric("Average Daily",  f"${preds.mean():,.2f}")
        fs2.metric("Minimum",        f"${preds.min():,.2f}")
        fs3.metric("Maximum",        f"${preds.max():,.2f}")
        fs4.metric("Total Forecast", f"${preds.sum():,.2f}")

        hist_avg     = df["Sales"].mean()
        pct_change   = ((preds.mean() - hist_avg) / hist_avg) * 100
        if pct_change > 0:
            st.info(f"📈 Forecast avg is **{pct_change:.1f}% higher** than historical avg (${hist_avg:,.2f})")
        else:
            st.warning(f"📉 Forecast avg is **{abs(pct_change):.1f}% lower** than historical avg (${hist_avg:,.2f})")

        st.markdown("---")
        st.markdown("### 📈 Forecast Visualisation")
        last6 = df[df["Date"] >= df["Date"].max() - timedelta(days=180)]
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(last6["Date"], last6["Sales"],
                color="blue",  linewidth=2, alpha=0.7, label="Historical")
        ax.plot(fdf["Date"],   fdf["Predicted_Sales"],
                color="red",   linewidth=2, linestyle="--", alpha=0.8, label="Forecast")
        ax.axvline(df["Date"].max(), color="gray", linestyle=":", linewidth=2, label="Today")
        ax.axvspan(fdf["Date"].min(), fdf["Date"].max(), alpha=0.08, color="red")
        ax.set_xlabel("Date"); ax.set_ylabel("Sales ($)")
        ax.set_title("Sales Forecast: Historical + Future", fontweight="bold")
        ax.legend(); ax.grid(True, alpha=0.3)
        st.pyplot(fig); plt.close(fig)   

        st.markdown("---")
        st.markdown("### 📅 Monthly Breakdown")

        fdf_copy = fdf.copy()
        fdf_copy["Year_Month"] = fdf_copy["Date"].dt.to_period("M")
        monthly = (
            fdf_copy.groupby("Year_Month")["Predicted_Sales"]
            .agg(["sum", "mean", "count"])
            .rename(columns={"sum": "Total Sales", "mean": "Avg Daily", "count": "Days"})
        )
        monthly.index = monthly.index.astype(str)
        monthly = monthly.reset_index().rename(columns={"Year_Month": "Month"})

        st.dataframe(
            monthly.style.format(
                {"Total Sales": "${:,.2f}", "Avg Daily": "${:,.2f}", "Days": "{:.0f}"}
            ),
            use_container_width=True,
        )

        st.markdown("---")
        st.markdown("### 📋 Daily Predictions (first 30 rows)")
        show_df = fdf[["Date", "Predicted_Sales", "Is_Weekend", "Month", "Quarter"]].copy()
        show_df["Day_Name"]  = show_df["Date"].dt.day_name()
        show_df["Is_Weekend"] = show_df["Is_Weekend"].map({0: "No", 1: "Yes"})
        st.dataframe(
            show_df[["Date","Day_Name","Predicted_Sales","Is_Weekend","Month"]],
            use_container_width=True,
        )

        csv_bytes = fdf[["Date","Predicted_Sales"]].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Forecast as CSV",
            data=csv_bytes,
            file_name=f"sales_forecast_{forecast_days}days.csv",
            mime="text/csv",
            key="download_forecast_csv",  
        )

# ── PAGE 5 — FEATURE ANALYSIS

elif page == "🔍 Feature Analysis":
    st.title("🔍 Feature Analysis")
    st.markdown("### Understand what drives sales")
    st.markdown("---")

    st.markdown("### 📊 Complete Feature Importance Ranking")

    feature_importance = (
        pd.DataFrame({"Feature": feature_names, "Coefficient": model.coef_})
        .assign(Abs_Coefficient=lambda d: d["Coefficient"].abs())
        .sort_values("Abs_Coefficient", ascending=False)
        .reset_index(drop=True)
    )
    feature_importance["Impact"] = feature_importance["Coefficient"].apply(
        lambda x: "📈 Increases Sales" if x > 0 else "📉 Decreases Sales"
    )
    feature_importance.index = feature_importance.index + 1  # rank starts at 1

    st.dataframe(
        feature_importance[["Feature", "Coefficient", "Impact"]].style.format(
            {"Coefficient": "${:,.2f}"}
        ),
        use_container_width=True,
    )

    st.markdown("---")
    st.markdown("### 🔬 Top 5 Features — Detailed Analysis")

    for rank, row in feature_importance.head(5).iterrows():
        feat  = row["Feature"]
        coef  = row["Coefficient"]
        dirn  = "increases" if coef > 0 else "decreases"
        icon  = "📈" if coef > 0 else "📉"

        with st.expander(f"**#{rank}: {feat}** — Coefficient: ${coef:,.2f}"):
            st.markdown(f"{icon} **Impact:** This feature {dirn} sales by **${abs(coef):,.2f}**")

            if "Weekend" in feat:
                st.markdown(f"""
**Interpretation:**  
Weekends add **${abs(coef):,.2f}** to daily sales vs weekdays.  
""")
            elif "Days_Since_Start" in feat:
                st.markdown(f"""
**Interpretation:**  
Sales grow **${abs(coef):,.2f}** per day (≈ **${abs(coef)*365:,.2f}** per year).  
""")
            elif feat == "Month":
                st.markdown(f"""
**Interpretation:**  
Each month later in the year shifts sales by **${abs(coef):,.2f}**.  
December vs January impact: **${abs(coef)*11:,.2f}**.  
""")
            elif "Q4" in feat:
                st.markdown(f"""
**Interpretation:**  
Q4 (Oct–Dec) changes sales by **${abs(coef):,.2f}** per day.  
""")
            elif "MA" in feat or "Lag" in feat:
                st.markdown(f"""
**Interpretation:**  
Recent sales momentum influences today's prediction.  
""")
            else:
                st.markdown(f"""
**Interpretation:**  
This feature {dirn} sales by **${abs(coef):,.2f}**.
""")

    st.markdown("---")
    st.markdown("### 🎯 Compare Specific Features")

    selected_features = st.multiselect(
        "Select features to compare",
        options=feature_names,
        default=feature_importance.head(5)["Feature"].tolist(),
        key="feature_compare_multiselect",   
    )

    if selected_features:
        sel_imp = feature_importance[
            feature_importance["Feature"].isin(selected_features)
        ]

        fig, ax = plt.subplots(figsize=(10, max(4, len(sel_imp) * 0.6)))
        colours = ["green" if c > 0 else "red" for c in sel_imp["Coefficient"]]
        ax.barh(
            range(len(sel_imp)), sel_imp["Coefficient"],
            color=colours, alpha=0.7, edgecolor="black",
        )
        ax.set_yticks(range(len(sel_imp)))
        ax.set_yticklabels(sel_imp["Feature"])
        ax.set_xlabel("Coefficient Value")
        ax.set_title("Feature Coefficients Comparison", fontweight="bold")
        ax.axvline(x=0, color="black", linestyle="--", linewidth=1)
        ax.grid(True, alpha=0.3, axis="x")
        st.pyplot(fig); plt.close(fig) 

# ── PAGE 6 — DOWNLOAD REPORTS 
elif page == "📥 Download Reports":
    st.title("📥 Download Reports")
    st.markdown("### Export your analysis and forecasts")
    st.markdown("---")

    forecast_dir = "data/forecasts"
    st.markdown("### 📄 Pre-generated Report Files")

    if os.path.exists(forecast_dir):
        files = [
            f for f in os.listdir(forecast_dir)
            if os.path.isfile(os.path.join(forecast_dir, f))
        ]
        if files:
            for fname in files:
                fpath     = os.path.join(forecast_dir, fname)
                file_size = os.path.getsize(fpath) / 1024

                rc1, rc2, rc3 = st.columns([3, 1, 1])
                rc1.markdown(f"**{fname}**")
                rc2.markdown(f"{file_size:.1f} KB")

                with open(fpath, "rb") as fh:
                    rc3.download_button(
                        label="⬇️ Download",
                        data=fh,
                        file_name=fname,
                        key=f"dl_btn_{fname}",   # ← unique per file
                    )
                st.markdown("---")
        else:
            st.info("No pre-generated reports found. Generate a forecast first!")
    else:
        st.warning("⚠️ `data/forecasts/` directory not found. Run Notebook 5 first.")

    st.markdown("### 📝 Generate a Quick Summary Report On-Demand")

    report_type = st.selectbox(
        "Report type",
        ["Summary Report", "Model Performance Report"],
        key="report_type_select",   
    )

    if st.button("Generate Report", key="generate_report_btn"):  
        with st.spinner("Building report …"):

            fi_report = (
                pd.DataFrame({"Feature": feature_names, "Coefficient": model.coef_})
                .assign(Abs=lambda d: d["Coefficient"].abs())
                .sort_values("Abs", ascending=False)
                .reset_index(drop=True)
            )

            lines = [
                "=" * 70,
                f"SALES FORECASTING — {report_type.upper()}",
                "=" * 70,
                f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "MODEL PERFORMANCE",
                "-" * 40,
                f"Model Type : Linear Regression",
                f"R² Score   : {test_r2:.4f}  ({test_r2*100:.1f}%)",
                f"MAE        : ${test_mae:,.2f}",
                f"RMSE       : ${test_rmse:,.2f}",
                "",
                "HISTORICAL DATA",
                "-" * 40,
                f"Date Range  : {df['Date'].min().date()} → {df['Date'].max().date()}",
                f"Total Days  : {len(df):,}",
                f"Avg Daily   : ${df['Sales'].mean():,.2f}",
                f"Total Sales : ${df['Sales'].sum():,.2f}",
                "",
                "TOP 5 FEATURES",
                "-" * 40,
            ]
            for i, row in fi_report.head(5).iterrows():
                lines.append(f"{i+1}. {row['Feature']:25s}  ${row['Coefficient']:,.2f}")

            lines += ["", "=" * 70, "END OF REPORT", "=" * 70]
            report_text = "\n".join(lines)

        st.success("✅ Report ready!")
        st.download_button(
            label="📥 Download Report",
            data=report_text.encode("utf-8"),
            file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            key="download_generated_report",   
        )

# ── FOOTER
st.markdown("---")
st.markdown(
    f"<div style='text-align:center;color:gray;padding:1rem 0'>"
    f"📊 Sales Forecasting Dashboard &nbsp;|&nbsp; "
    f"Built by PAARTH GANESH"
    f"</div>",
    unsafe_allow_html=True,
)