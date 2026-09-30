import streamlit as st
import pandas as pd
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

st.title("Airline Passengers Time-Series Explorer")

# Load the dataset
@st.cache_data
def load_data():
    return sns.load_dataset("flights")

df = load_data()
# Create a datetime column
df["date"] = pd.to_datetime(
    df["year"].astype(str) + "-" + df["month"].astype(str),
    format="%Y-%b"
)

# Set date as the index
df = df.set_index("date")

tab1, tab2 = st.tabs(["Charts", "Data"])

with tab2:
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
# Sidebar controls
st.sidebar.header("Controls")

resolution = st.sidebar.selectbox(
    "Resolution",
    ["Monthly", "Quarterly", "Yearly"]
)
rolling_window = st.sidebar.slider(
    "Rolling Window (months)",
    min_value=3,
    max_value=24,
    value=12
)
if resolution == "Monthly":
    series = df["passengers"].resample("MS").mean()
elif resolution == "Quarterly":
    series = df["passengers"].resample("QS").mean()
else:
    series = df["passengers"].resample("YS").mean()

with tab1:
    st.subheader(f"{resolution} Passenger Data")
    st.line_chart(series)


# Trend using a rolling mean
monthly = df["passengers"].resample("MS").mean()
trend = monthly.rolling(rolling_window).mean()

trend_data = pd.DataFrame({
    "Monthly Passengers": monthly,
    f"{rolling_window}-Month Rolling Mean": trend
})

with tab1:
    st.subheader("Passenger Trend")
    st.line_chart(trend_data)
# Seasonal decomposition
decomposition = seasonal_decompose(
    monthly,
    model="multiplicative",
    period=12
)
# Show uncertainty using a rolling mean +/- 2 standard deviations
rolling_mean = monthly.rolling(rolling_window).mean()
rolling_std = monthly.rolling(rolling_window).std()

uncertainty_data = pd.DataFrame({
    "Rolling Mean": rolling_mean,
    "Upper Bound": rolling_mean + 2 * rolling_std,
    "Lower Bound": rolling_mean - 2 * rolling_std
})

with tab1:
    st.subheader("Trend with Uncertainty")
    st.line_chart(uncertainty_data)

with tab1:
    st.subheader("Seasonality")
    st.line_chart(decomposition.seasonal)
st.subheader("Temporal Honesty Note")
st.write(
    "The monthly resolution is shown to preserve the seasonal pattern in the data. "
    f"A {rolling_window}-month rolling mean is used to show the long-term trend without hiding "
    "the original monthly variation."
)