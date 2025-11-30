import streamlit as st
import pandas as pd
import os


OUTPUT_FILE = "C:/Users/Admin/Desktop/EcommercePlatform/output/user_recommendations.csv"

st.set_page_config(page_title="E-Commerce Recommender", layout="wide")

st.markdown("<h1 style='text-align:center;'>🛍️ E-Commerce Recommendation System</h1>", unsafe_allow_html=True)
st.write("")


@st.cache_data
def load_data():
    if not os.path.exists(OUTPUT_FILE):
        st.error("❌ Recommendation file not found! Run main.py first.")
        return None

    df = pd.read_csv(OUTPUT_FILE)
    return df


# Load Data
df = load_data()

if df is not None:
    st.success("📂 Data loaded successfully!")

    # Sidebar Filters
    st.sidebar.header("🔍 Filters")

    # Customer Dropdown
    customers = sorted(df["CustomerID"].unique())
    selected_customer = st.sidebar.selectbox("Select a Customer ID", customers)

    # Show recommendations
    st.subheader(f"🎯 Recommendations for Customer **{selected_customer}**")

    filtered = df[df["CustomerID"] == selected_customer]

    if filtered.empty:
        st.warning("⚠ No recommendations available for this customer.")
    else:
        st.dataframe(filtered, use_container_width=True)

    # Show full dataset
    expander = st.expander("📊 View Full Recommendation Dataset")
    expander.dataframe(df)

    # CSV Download Button
    st.download_button(
        label="⬇ Download Full CSV",
        data=df.to_csv(index=False),
        file_name="user_recommendations.csv",
        mime="text/csv",
    )



st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>🚀 Built using <b>PySpark + Streamlit</b></p>", unsafe_allow_html=True)
