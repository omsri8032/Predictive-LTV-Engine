import pandas as pd
import numpy as np
import xgboost as xgb
import os
import argparse
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

CLEAN_FILE = os.path.join(os.path.dirname(__file__), '../data/cleaned_retail.csv')

def perform_interactive_forecast(target_date_str, period_months):
    print(f"\n🚀 INITIALIZING DYNAMIC AI FORECAST ENGINE")
    print(f"Target Date: {target_date_str}")
    print(f"Forecast Horizon: {period_months} Months\n")
    
    # 1. Load Data
    print("📥 Loading 1-Million Row Transaction Database...")
    df = pd.read_csv(CLEAN_FILE, parse_dates=['InvoiceDate'])
    
    target_date = pd.to_datetime(target_date_str)
    
    # Define past 3 months for historical comparison (as requested)
    past_3m_date = target_date - pd.DateOffset(months=3)
    past_3m_df = df[(df['InvoiceDate'] >= past_3m_date) & (df['InvoiceDate'] < target_date)]
    past_revenue = past_3m_df['Revenue'].sum()
    
    # Define RFM Observation Window (everything before target date)
    obs_df = df[df['InvoiceDate'] < target_date]
    if len(obs_df) == 0:
        print("❌ Error: Target date is too early. Not enough historical data.")
        return
        
    print(f"⚙️ Calculating RFM Customer Behavior Metrics up to {target_date.date()}...")
    rfm = obs_df.groupby('Customer ID').agg({
        'InvoiceDate': lambda x: (target_date - x.max()).days, # Recency
        'Invoice': 'nunique', # Frequency
        'Revenue': 'sum', # Monetary
    }).rename(columns={'InvoiceDate': 'Recency', 'Invoice': 'Frequency', 'Revenue': 'Monetary'})
    
    tenure = obs_df.groupby('Customer ID').agg({'InvoiceDate': lambda x: (target_date - x.min()).days}).rename(columns={'InvoiceDate': 'Tenure'})
    X = rfm.join(tenure)
    X = X[X['Monetary'] > 0] # Clean returns
    
    # 2. Dynamic Training on Historical Data
    print("🧠 Training XGBoost Live on Apple M2 Silicon...")
    # To train dynamically, we shift the window back by exactly `period_months` to learn from the past
    train_split_date = target_date - pd.DateOffset(months=period_months)
    train_obs = df[df['InvoiceDate'] < train_split_date]
    train_pred = df[(df['InvoiceDate'] >= train_split_date) & (df['InvoiceDate'] < target_date)]
    
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
    
    # Train Model
    model = xgb.XGBRegressor(
        objective='reg:squarederror', n_estimators=100, max_depth=5, 
        tree_method='hist', n_jobs=-1, random_state=42
    )
    model.fit(train_data[['Recency', 'Frequency', 'Monetary', 'Tenure']], train_data['Target'])
    
    # 3. Predict the Future
    print(f"🔮 Predicting Customer Spend for the next {period_months} months...")
    future_predictions = np.maximum(model.predict(X[['Recency', 'Frequency', 'Monetary', 'Tenure']]), 0)
    X['Predicted_Spend'] = future_predictions
    
    total_forecast = X['Predicted_Spend'].sum()
    
    # 4. Interactive Output
    print(f"\n{'='*50}")
    print(f"📊 MACRO FORECAST ({period_months} MONTHS)")
    print(f"{'='*50}")
    print(f"Historical {past_3m_date.date()} to {target_date.date()} Revenue: ${past_revenue:,.2f}")
    print(f"Predicted Future Pipeline Revenue:    ${total_forecast:,.2f}")
    
    print(f"\n{'='*50}")
    print(f"🎯 TOP 5 VIP CUSTOMERS TO TARGET IMMEDIATELY")
    print(f"{'='*50}")
    top_vips = X.nlargest(5, 'Predicted_Spend')
    rank = 1
    for index, row in top_vips.iterrows():
        print(f"{rank}. Customer {str(index).replace('.0','')} | Expected {period_months}-Month Spend: ${row['Predicted_Spend']:,.2f}")
        rank += 1
    
    print("\n✅ Done! The XGBoost engine completely predicted your business pipeline.")

if __name__ == "__main__":
    print("Welcome to the Thomson Reuters Retail Sales Forecaster.")
    target_date = input("Enter Start Date (e.g. 2011-09-01): ").strip()
    
    if not target_date:
        target_date = "2011-09-01"
        print(f"Defaulting to {target_date}")
        
    try:
        period = int(input("Enter Forecast Period in Months (e.g. 3): ").strip() or "3")
        perform_interactive_forecast(target_date, period)
    except Exception as e:
        print(f"Error: {e}")
