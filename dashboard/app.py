import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Fraud Dashboard", layout="wide")

# =========================
# FINAL UI CSS (FIXED FONTS + ICONS)
# =========================
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(to right, #FFF8E1, #FFE0B2);
}

/* Header */
.header {
    background: linear-gradient(to right, #FB8C00, #FFA726);
    padding: 25px;
    border-radius: 15px;
    color: white;
    text-align: center;
    font-size: 30px;
    font-weight: bold;
}

/* KPI Card */
.card {
    background: linear-gradient(to right, #FFF3E0, #FFE0B2);
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0px 5px 12px rgba(0,0,0,0.1);
}

/* Title inside card */
.card-title {
    font-size: 20px;
    font-weight: 600;
    margin-bottom: 5px;
}

/* Value inside card */
.card-value {
    font-size: 34px;
    font-weight: bold;
    color: #E65100;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #FFF8E1;
    font-size: 18px;
}

/* Slider label */
label {
    font-size: 16px !important;
}

</style>
""", unsafe_allow_html=True)

# =========================
# LOGIN
# =========================
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    st.title("🔐 Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        if u == "admin" and p == "1234":
            st.session_state["logged_in"] = True
        else:
            st.error("Invalid credentials")
    st.stop()

# =========================
# HEADER
# =========================
st.markdown("""
<div class="header">
💳 Credit Card Fraud Detection using Machine Learning<br>
<small>Real-Time Fraud Monitoring Dashboard</small>
</div>
""", unsafe_allow_html=True)

st.write("")

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv")
df["timestamp"] = pd.Timestamp("2024-01-01") + pd.to_timedelta(df["Time"], unit="s")
df["prediction"] = df["Class"]
df["label"] = df["prediction"].map({0: "Legitimate", 1: "Fraud"})

# =========================
# FILTER
# =========================
st.sidebar.title("🔍 Filters")

amount = st.sidebar.slider(
    "Transaction Amount",
    float(df["Amount"].min()),
    float(df["Amount"].max()),
    (0.0, 2000.0)
)

fraud_only = st.sidebar.checkbox("Show Only Fraud Transactions")

df = df[
    (df["Amount"] >= amount[0]) &
    (df["Amount"] <= amount[1])
]

if fraud_only:
    df = df[df["prediction"] == 1]
# =========================
# KPI CARDS
# =========================
total = len(df)
fraud = df["prediction"].sum()
rate = fraud / total if total else 0
fraud_amt = df[df["prediction"] == 1]["Amount"].sum()

c1, c2, c3, c4 = st.columns(4)

c1.markdown(f"""
<div class='card'>
<div class='card-title'>💳 Total Transactions</div>
<div class='card-value'>{total:,}</div>
</div>
""", unsafe_allow_html=True)

c2.markdown(f"""
<div class='card'>
<div class='card-title'>🚨 Fraud Cases</div>
<div class='card-value'>{fraud:,}</div>
</div>
""", unsafe_allow_html=True)

c3.markdown(f"""
<div class='card'>
<div class='card-title'>📊 Fraud Rate</div>
<div class='card-value'>{rate:.2%}</div>
</div>
""", unsafe_allow_html=True)

c4.markdown(f"""
<div class='card'>
<div class='card-title'>💰 Fraud Amount</div>
<div class='card-value'>{fraud_amt:,.0f}</div>
</div>
""", unsafe_allow_html=True)

st.write("")

# =========================
# CHARTS
# =========================
col1, col2 = st.columns(2)

# LINE (ORANGE)
with col1:
    trend = df[df["prediction"] == 1].groupby(
        df["timestamp"].dt.hour
    ).size().reset_index(name="count")

    fig = px.line(trend, x="timestamp", y="count", markers=True)
    fig.update_traces(line_color="#FB8C00")

    st.plotly_chart(fig, use_container_width=True)

# PIE (GREEN + RED)
with col2:
    pie = df["label"].value_counts().reset_index()
    pie.columns = ["label", "count"]

    fig = px.pie(
        pie,
        names="label",
        values="count",
        hole=0.5,
        color="label",
        color_discrete_map={
            "Fraud": "#E53935",
            "Legitimate": "#43A047"
        }
    )

    st.plotly_chart(fig, use_container_width=True)

# =========================
# HISTOGRAMS
# =========================
col3, col4 = st.columns(2)

with col3:
    fig = px.histogram(df, x="Amount", nbins=40,
                       color_discrete_sequence=["#1E88E5"])
    st.plotly_chart(fig, use_container_width=True)

with col4:
    df["prob"] = df["Amount"] / df["Amount"].max()
    fig = px.histogram(df, x="prob", nbins=40,
                       color_discrete_sequence=["#26A69A"])
    st.plotly_chart(fig, use_container_width=True)

# =========================
# TABLE
# =========================
st.subheader("📋 Recent High-Value Transactions")

st.dataframe(
    df.sort_values("Amount", ascending=False)[
        ["timestamp", "Amount", "label"]
    ].head(10),
    use_container_width=True
)