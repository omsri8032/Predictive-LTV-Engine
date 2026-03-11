import pandas as pd
import numpy as np
import os

CLEAN_FILE = os.path.join(os.path.dirname(__file__), '../data/cleaned_retail.csv')
FEATURES_FILE = os.path.join(os.path.dirname(__file__), '../data/rfm_features.csv')

def build_features():
    print(f"Loading {CLEAN_FILE}...")
    df = pd.read_csv(CLEAN_FILE, parse_dates=['InvoiceDate'])
    
    min_date = df['InvoiceDate'].min()
    max_date = df['InvoiceDate'].max()
    print(f"Data range: {min_date} to {max_date}")
    
    # Define Time Windows
    # Observation: From start to 2011-05-31
    # Prediction: 2011-06-01 to 2011-11-30 (6 months to predict future spending)
    split_date = pd.to_datetime('2011-06-01')
    end_date = pd.to_datetime('2011-12-01')
    
    # Filter to only the core 2 years (exclude incomplete December 2011)
    df = df[df['InvoiceDate'] < end_date]
    
    obs_df = df[df['InvoiceDate'] < split_date]
    pred_df = df[(df['InvoiceDate'] >= split_date)]
    
    print(f"Observation window: {obs_df['InvoiceDate'].min().date()} to {obs_df['InvoiceDate'].max().date()} ({len(obs_df)} rows)")
    print(f"Prediction window (Target): {pred_df['InvoiceDate'].min().date()} to {pred_df['InvoiceDate'].max().date()} ({len(pred_df)} rows)")
    
    snapshot_date = split_date

    # --- 1. RFM Features (Observation Window) ---
    print("Calculating RFM features...")
    rfm = obs_df.groupby('Customer ID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days, # Recency
        'Invoice': 'nunique',                                    # Frequency
        'Revenue': 'sum'                                         # Monetary
    }).rename(columns={
        'InvoiceDate': 'Recency',
        'Invoice': 'Frequency',
        'Revenue': 'Monetary'
    })
    
    # Customer Tenure/Age
    tenure = obs_df.groupby('Customer ID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.min()).days
    }).rename(columns={'InvoiceDate': 'Tenure'})
    
    rfm = rfm.join(tenure)
    
    # --- 2. Target Variable (Prediction Window) ---
    print("Calculating Target Variable (6-Month Future Spend)...")
    target = pred_df.groupby('Customer ID').agg({
        'Revenue': 'sum'
    }).rename(columns={'Revenue': 'Target_6M_Spend'})
    
    # Left join so we keep all customers from observation window
    # Fill target with 0 for customers who churned (didn't purchase in next 6 months)
    features = rfm.join(target, how='left')
    features['Target_6M_Spend'] = features['Target_6M_Spend'].fillna(0)
    
    # Ensure no negative monetary values (edge cases)
    features = features[features['Monetary'] > 0]
    
    print(f"Final feature store shape: {features.shape}")
    retention_rate = (features['Target_6M_Spend'] > 0).mean() * 100
    print(f"Percentage of customers who returned in prediction window (Baseline Retention): {retention_rate:.1f}%")
    
    features.to_csv(FEATURES_FILE)
    print(f"Saved feature matrix to {FEATURES_FILE}")

if __name__ == "__main__":
    build_features()
