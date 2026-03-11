import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import datetime

# --- CONFIGURATION ---
st.set_page_config(page_title="Thomson Reuters - Retail AI Forecaster", page_icon="📈", layout="wide")

# --- PREMIUM UI/UX STYLING (User Custom CSS Maps) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Comic+Neue:wght@400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Comic Neue', system-ui, sans-serif;
    }
    
    /* Main Background Pattern & Gradient */
    .stApp {
        background: linear-gradient(180deg, #ffffff 0%, #e8f4fd 100%) !important;
        color: #1c1c1e !important;
    }
    
    /* Typography Overrides */
    h1, h2, h3, p, span, div {
        color: #1c1c1e;
    }
    
    /* Title Highlight Gradient */
    h1 {
        background: linear-gradient(135deg, #3a7bd5 0%, #00d2ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700 !important;
        padding-bottom: 10px;
        font-size: 3.8rem !important;
        letter-spacing: -2px;
    }
    
    /* Metric Cards (Matches feature-card & strip-item hover) */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        color: #2b5db0 !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.95rem !important;
        color: #64748b !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    [data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.75);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.4);
        border-radius: 22px;
        padding: 30px 35px;
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.08);
        transition: 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    [data-testid="metric-container"]:hover {
        background: white;
        transform: translateY(-8px);
        border-color: #3a7bd5;
        box-shadow: 0 18px 40px rgba(58, 123, 213, 0.25);
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.9) !important;
        border-right: 1px solid #e5e7eb;
    }
    
    /* Styled Button (Matches .btn-primary) */
    .stButton > button {
        background: linear-gradient(135deg, #3a7bd5 0%, #00d2ff 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 16px 34px !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px !important;
        transition: 0.3s !important;
        box-shadow: 0 12px 30px rgba(58, 123, 213, 0.35) !important;
        width: 100%;
        overflow: hidden;
    }
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 18px 40px rgba(58, 123, 213, 0.45) !important;
    }
    
    /* DataFrame/Table Customization (Matches credential-item) */
    .stDataFrame {
        background: #fff;
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid #e8edf2;
        box-shadow: 0 3px 12px rgba(0, 0, 0, 0.04);
        transition: all 0.25s ease;
    }
</style>
""", unsafe_allow_html=True)

st.title("AI-Powered Retail Sales Forecaster")
st.markdown("""
Welcome to the interactive **Customer Lifetime Value (LTV)** Prediction Engine.
Adjust the parameters below to dynamically train the **XGBoost Regressor** on historical RFM metrics 
and instantly predict the future macro pipeline and micro VIP customer behavior.
""")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    file_path = os.path.join(os.path.dirname(__file__), '../data/cleaned_retail.csv')
    df = pd.read_csv(file_path, parse_dates=['InvoiceDate'])
    return df

try:
    df = load_data()
    min_date = df['InvoiceDate'].min().date()
    max_date = df['InvoiceDate'].max().date()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# --- SIDEBAR INPUTS ---
st.sidebar.header("⚙️ Forecast Parameters")
st.sidebar.info(f"Dataset Range:\n{min_date} to {max_date}")

# Default to 2011-08-01 (when data is robust)
target_date = st.sidebar.date_input("Select Target Split Date", datetime.date(2011, 8, 1), min_value=min_date, max_value=max_date)
period_months = st.sidebar.slider("Forecast Horizon (Months)", min_value=1, max_value=12, value=6)

if st.sidebar.button("🚀 Generate Forecast", type="primary"):
    with st.spinner("Calculating RFM metrics, training XGBoost, and simulating future pipeline..."):
        
        target_timestamp = pd.to_datetime(target_date)
        
        # 1. Historical 3-Month Lookback
        past_3m_date = target_timestamp - pd.DateOffset(months=3)
        past_3m_df = df[(df['InvoiceDate'] >= past_3m_date) & (df['InvoiceDate'] < target_timestamp)]
        past_revenue = past_3m_df['Revenue'].sum()
        
        # 2. Build RFM Store
        obs_df = df[df['InvoiceDate'] < target_timestamp]
        if len(obs_df) == 0:
            st.error("❌ Target date is too early. Not enough historical data to generate features.")
            st.stop()
            
        rfm = obs_df.groupby('Customer ID').agg({
            'InvoiceDate': lambda x: (target_timestamp - x.max()).days,
            'Invoice': 'nunique',
            'Revenue': 'sum',
        }).rename(columns={'InvoiceDate': 'Recency', 'Invoice': 'Frequency', 'Revenue': 'Monetary'})
        
        tenure = obs_df.groupby('Customer ID').agg({'InvoiceDate': lambda x: (target_timestamp - x.min()).days}).rename(columns={'InvoiceDate': 'Tenure'})
        X = rfm.join(tenure)
        X = X[X['Monetary'] > 0]
        
        # 3. Dynamic XGBoost Training
        train_split_date = target_timestamp - pd.DateOffset(months=period_months)
        train_obs = df[df['InvoiceDate'] < train_split_date]
        train_pred = df[(df['InvoiceDate'] >= train_split_date) & (df['InvoiceDate'] < target_timestamp)]
        
        if len(train_obs) == 0 or len(train_pred) == 0:
            st.error("❌ Not enough historical data to train the model dynamically. Choose a later date.")
            st.stop()
            
        train_rfm = train_obs.groupby('Customer ID').agg({
            'InvoiceDate': lambda x: (train_split_date - x.max()).days,
            'Invoice': 'nunique',
            'Revenue': 'sum'
        }).rename(columns={'InvoiceDate': 'Recency', 'Invoice': 'Frequency', 'Revenue': 'Monetary'})
        
        train_tenure = train_obs.groupby('Customer ID').agg({'InvoiceDate': lambda x: (train_split_date - x.min()).days}).rename(columns={'InvoiceDate': 'Tenure'})
        X_train = train_rfm.join(train_tenure)
        
        y_train_df = train_pred.groupby('Customer ID').agg({'Revenue': 'sum'}).rename(columns={'Revenue': 'Target'})
        train_data = X_train.join(y_train_df, how='left').fillna({'Target': 0})
        train_data = train_data[train_data['Monetary'] > 0]
        
        model = xgb.XGBRegressor(
            objective='reg:squarederror', n_estimators=100, max_depth=5, 
            tree_method='hist', n_jobs=-1, random_state=42
        )
        model.fit(train_data[['Recency', 'Frequency', 'Monetary', 'Tenure']], train_data['Target'])
        
        # 4. Predict
        future_predictions = np.maximum(model.predict(X[['Recency', 'Frequency', 'Monetary', 'Tenure']]), 0)
        X['Predicted_Spend'] = future_predictions
        total_forecast = X['Predicted_Spend'].sum()
        
        # --- DISPLAY RESULTS ---
        
        # Top-level metrics
        st.subheader("📊 Macro Business Forecast")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(f"Historical Revenue ({past_3m_date.date()} to {target_date})", f"${past_revenue:,.0f}")
        with col2:
            delta_val = total_forecast - past_revenue
            st.metric(f"Predicted Future Revenue (Next {period_months} Months)", f"${total_forecast:,.0f}", f"${delta_val:,.0f}")
        with col3:
            st.metric("Total Active Customers Modeled", f"{len(X):,}")
            
        st.divider()
        
        # VIP Customers Table
        st.subheader("🎯 Top VIP Customers to Target")
        st.markdown("Marketing should prioritize these predicted high-value LTV accounts immediately.")
        
        top_vips = X.nlargest(10, 'Predicted_Spend').reset_index()
        top_vips['Customer ID'] = top_vips['Customer ID'].astype(str).str.replace('.0', '', regex=False)
        
        display_df = top_vips[['Customer ID', 'Predicted_Spend', 'Monetary', 'Frequency', 'Recency', 'Tenure']].copy()
        display_df.rename(columns={
            'Predicted_Spend': f'Predicted {period_months}M Spend ($)',
            'Monetary': 'Lifetime Sunk Revenue ($)',
            'Frequency': 'Total Past Orders',
            'Recency': 'Days Since Last Order',
            'Tenure': 'Days Since First Order'
        }, inplace=True)
        
        st.dataframe(
            display_df.style.format({
                f'Predicted {period_months}M Spend ($)': "{:,.2f}",
                'Lifetime Sunk Revenue ($)': "{:,.2f}"
            }),
            use_container_width=True
        )

        # --- BRAND NEW INTERACTIVE CHARTS ---
        st.subheader("📈 Predicted VIP LTV Distribution & Churn Risk")
        
        # Recreate the "Business Segment" Logic from your backend
        conditions = [
            (X['Predicted_Spend'] > np.percentile(X['Predicted_Spend'], 80)),
            (X['Predicted_Spend'] > np.percentile(X['Predicted_Spend'], 50))
        ]
        choices = ['High Value VIP', 'Medium Value']
        X['Business Segment'] = np.select(conditions, choices, default='Low Value / Churn Risk')
        
        # Two-column layout for the charts
        col_chart_1, col_chart_2 = st.columns(2)
        
        with col_chart_1:
            st.markdown("**Expected Revenue by Segment**")
            # Group the predicted spend by segment to mimic Tableau's first chart
            segment_rev = X.groupby('Business Segment')['Predicted_Spend'].sum()
            st.bar_chart(segment_rev, height=350)
            
        with col_chart_2:
            st.markdown("**The Customer Matrix (Recency vs. Frequency)**")
            # Native Streamlit scatter plot colored by their Business Segment
            st.scatter_chart(X, x="Recency", y="Frequency", color="Business Segment", height=350)
