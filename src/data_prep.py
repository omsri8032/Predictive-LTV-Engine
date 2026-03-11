import pandas as pd
import os

DATA_FILE = os.path.join(os.path.dirname(__file__), '../data/online_retail_II.xlsx')
CLEAN_FILE = os.path.join(os.path.dirname(__file__), '../data/cleaned_retail.csv')

def clean_data():
    print(f"Loading raw data from {DATA_FILE}...")
    # The dataset has two sheets (2009-2010 and 2010-2011)
    xls = pd.ExcelFile(DATA_FILE)
    df1 = pd.read_excel(xls, 'Year 2009-2010')
    df2 = pd.read_excel(xls, 'Year 2010-2011')
    
    df = pd.concat([df1, df2], ignore_index=True)
    print(f"Raw shape: {df.shape}")
    
    # 1. Drop null CustomerIDs (we can't predict LTV for guest checkouts)
    df = df.dropna(subset=['Customer ID'])
    
    # 2. Remove cancelled orders (Invoice beginning with 'C')
    df['Invoice'] = df['Invoice'].astype(str)
    df = df[~df['Invoice'].str.startswith('C')]
    
    # 3. Handle anomalies (keep positive quantities and prices)
    df = df[(df['Quantity'] > 0) & (df['Price'] > 0)]
    
    # 4. Calculate Total Revenue per line item
    df['Revenue'] = df['Quantity'] * df['Price']
    
    print(f"Unique Customers: {df['Customer ID'].nunique()}")
    print(f"Cleaned shape: {df.shape}")
    
    # Save to intermediate CSV for fast loading later
    df.to_csv(CLEAN_FILE, index=False)
    print(f"Saved cleaned data to {CLEAN_FILE}")
    
if __name__ == "__main__":
    clean_data()
