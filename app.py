# app.py – Crime EDA Dashboard (Final Version)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

st.set_page_config(
    page_title="Crime EDA - Indian Cities",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

# -----------------------------
# Preprocess (Handles all datasets)
# -----------------------------
def preprocess(df):

    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    col_map = {}

    for name in df.columns:
        low = name.lower()
        if "year" in low:
            col_map["Year"] = name
        elif "city" in low or "district" in low or "place" in low:
            col_map["City"] = name
        elif "crime head" in low or "crime description" in low:
            col_map["Crime_Type"] = name
        elif any(k in low for k in ["crime", "offence", "offense", "description"]):
            if "Crime_Type" not in col_map:
                col_map["Crime_Type"] = name
        elif any(k in low for k in ["cases", "case", "count", "total", "reported"]):
            col_map["Cases_Reported"] = name

    # Year
    if "Year" in col_map:
        year_series = pd.to_numeric(df[col_map["Year"]], errors="ignore")
    else:
        year_series = pd.Series(pd.NA, index=df.index)

    # City
    if "City" in col_map:
        city_series = df[col_map["City"]].astype(str)
    else:
        city_series = pd.Series("Unknown", index=df.index)

    # Crime Type
    if "Crime_Type" in col_map:
        crime_series = df[col_map["Crime_Type"]].astype(str)
    else:
        crime_series = pd.Series("Misc", index=df.index)

    # Cases
    if "Cases_Reported" in col_map:
        cases_series = pd.to_numeric(df[col_map["Cases_Reported"]], errors="coerce").fillna(0)
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            cases_series = pd.to_numeric(df[numeric_cols[0]], errors="coerce").fillna(0)
        else:
            cases_series = pd.Series(1, index=df.index)

    # Build Clean DF
    norm = pd.DataFrame({
        "Year": year_series,
        "City": city_series,
        "Crime_Type": crime_series,
        "Cases_Reported": cases_series
    })

    # Remove "nan", None, Null values for dropdowns
    for col in ["City", "Crime_Type"]:
        norm[col] = norm[col].astype(str).replace(
            ["nan", "NaN", "NONE", "null", "None", ""],
            np.nan
        ).fillna("Unknown")

    return norm


# -----------------------------
# Load Dataset
# -----------------------------
DEFAULT_FILE = "updated_crime_dataset_2020_2024_final.csv"

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df_raw = load_data(uploaded_file)
elif os.path.exists(DEFAULT_FILE):
    df_raw = load_data(DEFAULT_FILE)
    st.success(f"Loaded dataset: {DEFAULT_FILE}")
else:
    st.error("No dataset found. Upload a CSV file!")
    st.stop()

df = preprocess(df_raw)

# Auto-fix: If entire column is zero → treat each row as 1 case
if df["Cases_Reported"].sum() == 0:
    df["Cases_Reported"] = 1.0

# -----------------------------
# FILTER SECTION
# -----------------------------
years = sorted(df["Year"].dropna().unique().tolist())
cities = sorted(df["City"].unique().tolist())
crime_types = sorted(df["Crime_Type"].unique().tolist())

year_options = ["All"] + years
city_options = ["All"] + cities
crime_options = ["All"] + crime_types

col1, col2, col3, col4 = st.columns([3, 3, 3, 1])

selected_year = col1.multiselect("Select Year", year_options, default=["All"])
selected_city = col2.multiselect("Select City", city_options, default=["All"])
selected_crime = col3.multiselect("Select Crime Type", crime_options, default=["All"])

apply_btn = col4.button("Apply")

df_filtered = df.copy()

if "All" not in selected_year:
    df_filtered = df_filtered[df_filtered["Year"].isin(selected_year)]

if "All" not in selected_city:
    df_filtered = df_filtered[df_filtered["City"].isin(selected_city)]

if "All" not in selected_crime:
    df_filtered = df_filtered[df_filtered["Crime_Type"].isin(selected_crime)]

if df_filtered.empty:
    st.error("No data found. Try different filters.")
    st.stop()

# -----------------------------
# KPIs
# -----------------------------
total_records = len(df_filtered)
total_cases = int(df_filtered["Cases_Reported"].sum())
unique_cities = df_filtered["City"].nunique()
unique_crimes = df_filtered["Crime_Type"].nunique()

st.info(
    f"For selected filters → Records: **{total_records}**, Total Cases: **{total_cases}**, "
    f"Cities: **{unique_cities}**, Crime Types: **{unique_crimes}**"
)

# -----------------------------
# BAR CHART (22 COLOR SUPPORT)
# -----------------------------
st.subheader("📊 Crime Type Distribution (Bar Chart)")

crime_sum = (
    df_filtered.groupby("Crime_Type")["Cases_Reported"]
    .sum()
    .sort_values(ascending=True)
    .reset_index()
)

# 22 unique colors
colors = px.colors.qualitative.Alphabet + px.colors.qualitative.Dark24

color_map = {
    crime_sum["Crime_Type"][i]: colors[i % len(colors)]
    for i in range(len(crime_sum))
}

fig_bar = px.bar(
    crime_sum,
    y="Crime_Type",
    x="Cases_Reported",
    color="Crime_Type",
    color_discrete_map=color_map,
    orientation="h",
    title="Total Cases by Crime Type",
)

fig_bar.update_layout(height=700)
st.plotly_chart(fig_bar, use_container_width=True)

# -----------------------------
# PIE CHART
# -----------------------------
st.subheader("🥧 City-wise Crime Share (Pie Chart)")

city_sum = (
    df_filtered.groupby("City")["Cases_Reported"]
    .sum()
    .reset_index()
)

fig_pie = px.pie(
    city_sum,
    names="City",
    values="Cases_Reported",
    title="Crime Share Across Cities",
)

st.plotly_chart(fig_pie, use_container_width=True)

# -----------------------------
# HISTOGRAM
# -----------------------------
st.subheader("📈 Distribution of Cases (Histogram)")

fig_hist = px.histogram(
    df_filtered,
    x="Cases_Reported",
    nbins=30,
)
st.plotly_chart(fig_hist, use_container_width=True)

# -----------------------------
# SHOW TABLE
# -----------------------------
with st.expander("📄 View Filtered Dataset"):
    st.dataframe(df_filtered)

st.caption("Built using Streamlit + Pandas + Plotly — Complete Crime EDA Dashboard.")
