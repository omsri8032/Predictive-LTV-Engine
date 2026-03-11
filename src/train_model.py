import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

FEATURES_FILE = os.path.join(os.path.dirname(__file__), '../data/rfm_features.csv')
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), '../data/ltv_predictions.csv')

def train_xgboost():
    print(f"Loading feature matrix from {FEATURES_FILE}...")
    df = pd.read_csv(FEATURES_FILE)
    
    # 1. Feature and Target Selection
    X = df[['Recency', 'Frequency', 'Monetary', 'Tenure']]
    y = df['Target_6M_Spend']
    
    # 2. Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training Data: {X_train.shape[0]} customers")
    print(f"Testing Data: {X_test.shape[0]} customers")
    
    # 3. Model Configuration (Optimized for Apple Silicon M2)
    # XGBoost tabular analytics uses deep multi-threading. By leveraging the 'hist' 
    # tree_method and all M-series efficiency/performance cores (n_jobs=-1), we 
    # achieve hardware-native acceleration bypassing the need for explicit GPU offloading.
    print("\n⚡ Configuring XGBoost for Apple M2 Silicon performance...")
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=150,
        learning_rate=0.05,
        max_depth=5,
        tree_method='hist', # Histogram-based algorithm (fastest on M2 Architecture)
        n_jobs=-1,          # Utilize all CPU cores heavily
        random_state=42
    )
    
    # 4. Train Model
    print("Training XGBoost Regressor...")
    xgb_model.fit(X_train, y_train)
    
    # 5. Predict & Evaluate
    print("Generating predictions on test set...")
    y_pred = xgb_model.predict(X_test)
    y_pred = np.maximum(y_pred, 0) # Revenue cannot be negative
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print("\n📊 --- Model Performance Metrics ---")
    print(f"Mean Absolute Error (MAE): ${mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
    print(f"R-squared (R2): {r2:.3f}")
    print("----------------------------------\n")
    
    # 6. Apply to entire population for Business Dashboard Insights
    print("Generating final predictive segments for all customers (for Tableau BI)...")
    df['Predicted_6M_Spend'] = np.maximum(xgb_model.predict(X), 0)
    
    # Rank customers into 10 decile buckets based on predicted future spend
    df['LTV_Segment_Rank'] = pd.qcut(df['Predicted_6M_Spend'].rank(method='first'), q=10, labels=False) + 1
    
    # Create descriptive business labels
    conditions = [
        (df['LTV_Segment_Rank'] >= 9), # Top 20%
        (df['LTV_Segment_Rank'] >= 5) & (df['LTV_Segment_Rank'] <= 8), # Mid 40%
        (df['LTV_Segment_Rank'] <= 4)  # Bottom 40%
    ]
    choices = ['High Value VIP', 'Medium Value', 'Low Value / Churn Risk']
    df['Business_Segment'] = np.select(conditions, choices, default='Unknown')
    
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Final LTV predictions and segments saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    train_xgboost()
