import pandas as pd
import mysql.connector
import os
from dotenv import load_dotenv

# Load local environment variables (if any) or fallback to defaults
load_dotenv()

PREDICTIONS_FILE = os.path.join(os.path.dirname(__file__), '../data/ltv_predictions.csv')

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', 'root')
}

def load_to_mysql():
    print("🔌 Connecting to MySQL Server...")
    
    # 1. Connect without database to CREATE DATABASE
    init_conn = mysql.connector.connect(**DB_CONFIG)
    cursor = init_conn.cursor()
    cursor.execute("CREATE DATABASE IF NOT EXISTS retail_ltv")
    init_conn.commit()
    cursor.close()
    init_conn.close()
    
    # 2. Connect to the new retail_ltv database
    conn = mysql.connector.connect(**DB_CONFIG, database='retail_ltv')
    cursor = conn.cursor()
    
    print("🏗️ Creating `customer_predictions` schema...")
    cursor.execute("DROP TABLE IF EXISTS customer_predictions")
    
    # Create strongly typed columns for Tableau ingestion
    schema = """
    CREATE TABLE customer_predictions (
        customer_id VARCHAR(50) PRIMARY KEY,
        recency INT,
        frequency INT,
        monetary FLOAT,
        tenure INT,
        actual_6m_spend FLOAT,
        predicted_6m_spend FLOAT,
        ltv_segment_rank INT,
        business_segment VARCHAR(50),
        INDEX idx_segment (business_segment)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """
    cursor.execute(schema)
    conn.commit()
    
    # 3. Load DataFrame and Insert
    print(f"📥 Loading predictions from {PREDICTIONS_FILE}...")
    df = pd.read_csv(PREDICTIONS_FILE)
    
    insert_sql = """
    INSERT INTO customer_predictions 
        (customer_id, recency, frequency, monetary, tenure, actual_6m_spend, predicted_6m_spend, ltv_segment_rank, business_segment)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    
    rows = []
    for _, row in df.iterrows():
        rows.append((
            str(row['Customer ID']).replace('.0', ''), # Clean ID formatting
            int(row['Recency']) if pd.notna(row['Recency']) else 0,
            int(row['Frequency']) if pd.notna(row['Frequency']) else 0,
            float(row['Monetary']) if pd.notna(row['Monetary']) else 0.0,
            int(row['Tenure']) if pd.notna(row['Tenure']) else 0,
            float(row['Target_6M_Spend']) if pd.notna(row['Target_6M_Spend']) else 0.0,
            float(row['Predicted_6M_Spend']) if pd.notna(row['Predicted_6M_Spend']) else 0.0,
            int(row['LTV_Segment_Rank']) if pd.notna(row['LTV_Segment_Rank']) else 0,
            str(row['Business_Segment'])
        ))
    
    print("🚀 Batch inserting XGBoost predictions into MySQL...")
    cursor.executemany(insert_sql, rows)
    conn.commit()
    
    print(f"✅ Successfully loaded {cursor.rowcount} high-value records into `retail_ltv.customer_predictions`")
    
    cursor.close()
    conn.close()

if __name__ == "__main__":
    load_to_mysql()
