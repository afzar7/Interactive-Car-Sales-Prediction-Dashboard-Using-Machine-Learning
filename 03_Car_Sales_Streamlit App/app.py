
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score

# ---------------------------------------------------------
# PAGE SETTINGS
# ---------------------------------------------------------
st.set_page_config(
    page_title="Car Sales ML Dashboard",
    page_icon="🚗",
    layout="wide"
)

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Cleaned_Car_sales.csv")

    # Remove accidental index column if it exists
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    return df

df = load_data()

# ---------------------------------------------------------
# TRAIN MODELS
# ---------------------------------------------------------
numeric_features = [
    "Price_in_thousands",
    "Engine_size",
    "Horsepower",
    "Wheelbase",
    "Width",
    "Length",
    "Curb_weight",
    "Fuel_capacity",
    "Fuel_efficiency",
    "Power_perf_factor",
    "year_resale_value"
]

# Regression target
X = df[numeric_features]
y_sales = df["Sales_in_thousands"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y_sales, test_size=0.2, random_state=42
)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

rf_regressor = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)
rf_regressor.fit(X_train, y_train)

# Classification target
# Category: 0 = Low Sales, 1 = High Sales
y_category = df["Category"]

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X, y_category, test_size=0.2, random_state=42, stratify=y_category
)

rf_classifier = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
rf_classifier.fit(Xc_train, yc_train)

# K-Means
cluster_features = ["Price_in_thousands", "Horsepower"]
cluster_data = df[cluster_features]

scaler = StandardScaler()
cluster_scaled = scaler.fit_transform(cluster_data)

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)
df["Cluster"] = kmeans.fit_predict(cluster_scaled)

# ---------------------------------------------------------
# MODEL METRICS
# ---------------------------------------------------------
linear_pred = linear_model.predict(X_test)
rf_pred = rf_regressor.predict(X_test)
class_pred = rf_classifier.predict(Xc_test)

linear_r2 = r2_score(y_test, linear_pred)
linear_rmse = np.sqrt(mean_squared_error(y_test, linear_pred))

rf_r2 = r2_score(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

classification_accuracy = accuracy_score(yc_test, class_pred)

# ---------------------------------------------------------
# SIDEBAR - CAR SELECTION
# ---------------------------------------------------------
st.sidebar.title("🚗 Car Selection")

# Unique manufacturers only
manufacturers = sorted(df["Manufacturer"].dropna().unique())

selected_manufacturer = st.sidebar.radio(
    "Select Manufacturer",
    manufacturers
)

manufacturer_df = df[
    df["Manufacturer"] == selected_manufacturer
].copy()

# Unique model names
models = sorted(manufacturer_df["Model"].dropna().unique())

if len(models) >= 2:
    selected_models = st.sidebar.multiselect(
        "Select 2 Car Models",
        models,
        default=models[:2],
        max_selections=2
    )
else:
    selected_models = models

if len(selected_models) != 2:
    st.warning("Please select exactly 2 car models from the selected manufacturer.")
    st.stop()

car1 = manufacturer_df[
    manufacturer_df["Model"] == selected_models[0]
].iloc[0]

car2 = manufacturer_df[
    manufacturer_df["Model"] == selected_models[1]
].iloc[0]

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.title("🚗 Car Sales Machine Learning Dashboard")
st.markdown(
    "Compare two cars and explore predictions from multiple machine-learning algorithms."
)

st.divider()

# ---------------------------------------------------------
# CAR COMPARISON
# ---------------------------------------------------------
st.header("🚘 Car Comparison")

col1, col2 = st.columns(2)

with col1:
    st.subheader(f"{car1['Manufacturer']} {car1['Model']}")
    st.metric(
        "Price",
        f"${car1['Price_in_thousands']:.2f}K"
    )
    st.metric(
        "Sales",
        f"{car1['Sales_in_thousands']:.2f}K"
    )
    st.metric(
        "Horsepower",
        f"{car1['Horsepower']:.0f} HP"
    )

with col2:
    st.subheader(f"{car2['Manufacturer']} {car2['Model']}")
    st.metric(
        "Price",
        f"${car2['Price_in_thousands']:.2f}K"
    )
    st.metric(
        "Sales",
        f"{car2['Sales_in_thousands']:.2f}K"
    )
    st.metric(
        "Horsepower",
        f"{car2['Horsepower']:.0f} HP"
    )

# ---------------------------------------------------------
# PRICE AND SALES CHART
# ---------------------------------------------------------
st.header("💰 Price and Sales Information")

comparison_df = pd.DataFrame({
    "Car": [
        f"{car1['Manufacturer']} {car1['Model']}",
        f"{car2['Manufacturer']} {car2['Model']}"
    ],
    "Price": [
        car1["Price_in_thousands"],
        car2["Price_in_thousands"]
    ],
    "Sales": [
        car1["Sales_in_thousands"],
        car2["Sales_in_thousands"]
    ]
})

chart_type = st.selectbox(
    "Chart type",
    ["Bar Chart", "Scatter Chart"]
)

if chart_type == "Bar Chart":
    chart_long = comparison_df.melt(
        id_vars="Car",
        value_vars=["Price", "Sales"],
        var_name="Metric",
        value_name="Value"
    )

    fig = px.bar(
        chart_long,
        x="Car",
        y="Value",
        color="Metric",
        barmode="group",
        title="Price vs Sales"
    )
else:
    fig = px.scatter(
        comparison_df,
        x="Price",
        y="Sales",
        text="Car",
        size="Sales",
        title="Price vs Sales"
    )
    fig.update_traces(textposition="top center")

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# PREDICTIONS FOR SELECTED CARS
# ---------------------------------------------------------
selected_data = pd.DataFrame([car1, car2])
selected_X = selected_data[numeric_features]

linear_selected = linear_model.predict(selected_X)
rf_selected = rf_regressor.predict(selected_X)
class_selected = rf_classifier.predict(selected_X)

prediction_df = pd.DataFrame({
    "Car": [
        f"{car1['Manufacturer']} {car1['Model']}",
        f"{car2['Manufacturer']} {car2['Model']}"
    ],
    "Actual Sales (K)": selected_data["Sales_in_thousands"].values,
    "Linear Regression (K)": linear_selected,
    "Random Forest (K)": rf_selected,
    "Sales Category": [
        "High Sales" if x == 1 else "Low Sales"
        for x in class_selected
    ],
    "Cluster": df[
        df["Model"].isin(selected_models)
    ]["Cluster"].values
})

# ---------------------------------------------------------
# LINEAR REGRESSION
# ---------------------------------------------------------
st.header("📊 Linear Regression Prediction")

lr_col1, lr_col2 = st.columns(2)

for i, col in enumerate([lr_col1, lr_col2]):
    with col:
        car_name = prediction_df.iloc[i]["Car"]
        st.metric(
            car_name,
            f"{prediction_df.iloc[i]['Linear Regression (K)']:.2f}K"
        )

st.write(
    f"Model R² score: **{linear_r2:.3f}**  |  "
    f"RMSE: **{linear_rmse:.3f}**"
)

# ---------------------------------------------------------
# RANDOM FOREST REGRESSION
# ---------------------------------------------------------
st.header("🌲 Random Forest Regression")

rf_col1, rf_col2 = st.columns(2)

for i, col in enumerate([rf_col1, rf_col2]):
    with col:
        car_name = prediction_df.iloc[i]["Car"]
        st.metric(
            car_name,
            f"{prediction_df.iloc[i]['Random Forest (K)']:.2f}K"
        )

st.write(
    f"Model R² score: **{rf_r2:.3f}**  |  "
    f"RMSE: **{rf_rmse:.3f}**"
)

# ---------------------------------------------------------
# RANDOM FOREST CLASSIFICATION
# ---------------------------------------------------------
st.header("🎯 Random Forest Classification")

class_col1, class_col2 = st.columns(2)

for i, col in enumerate([class_col1, class_col2]):
    with col:
        car_name = prediction_df.iloc[i]["Car"]
        category = prediction_df.iloc[i]["Sales Category"]

        st.metric(
            car_name,
            category
        )

st.write(
    f"Classification accuracy: **{classification_accuracy:.2%}**"
)

# ---------------------------------------------------------
# K-MEANS
# ---------------------------------------------------------
st.header("🔵 K-Means Clustering")

cluster_col1, cluster_col2 = st.columns(2)

for i, col in enumerate([cluster_col1, cluster_col2]):
    with col:
        car_name = prediction_df.iloc[i]["Car"]
        cluster = int(prediction_df.iloc[i]["Cluster"])

        st.metric(
            car_name,
            f"Cluster {cluster}"
        )

st.info(
    "Cars in the same cluster have similar Price and Horsepower patterns. "
    "The cluster numbers themselves do not represent a ranking."
)

# ---------------------------------------------------------
# INTERACTIVE CLUSTER CHART
# ---------------------------------------------------------
cluster_plot = px.scatter(
    df,
    x="Price_in_thousands",
    y="Horsepower",
    color=df["Cluster"].astype(str),
    hover_data=["Manufacturer", "Model", "Sales_in_thousands"],
    text=None,
    title="Cars Grouped by Price and Horsepower",
    labels={"color": "Cluster"}
)

st.plotly_chart(cluster_plot, use_container_width=True)

# ---------------------------------------------------------
# MODEL COMPARISON CHART
# ---------------------------------------------------------
st.header("📈 Interactive Model Comparison")

model_comparison = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest Regression"
    ],
    "R2 Score": [
        linear_r2,
        rf_r2
    ]
})

fig_model = px.bar(
    model_comparison,
    x="Model",
    y="R2 Score",
    text="R2 Score",
    title="Regression Model Performance"
)

fig_model.update_traces(
    texttemplate="%{text:.3f}",
    textposition="outside"
)

st.plotly_chart(fig_model, use_container_width=True)

# ---------------------------------------------------------
# DETAILED COMPARISON TABLE
# ---------------------------------------------------------
st.header("📋 Detailed Car Comparison")

display_columns = [
    "Manufacturer",
    "Model",
    "Vehicle_type",
    "Price_in_thousands",
    "Sales_in_thousands",
    "year_resale_value",
    "Engine_size",
    "Horsepower",
    "Fuel_capacity",
    "Fuel_efficiency",
    "Power_perf_factor"
]

st.dataframe(
    selected_data[display_columns].reset_index(drop=True),
    use_container_width=True,
    hide_index=True
)

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.divider()
