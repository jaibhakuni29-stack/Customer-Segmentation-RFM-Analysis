import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

print("=" * 80)
print("GENERATING SAMPLE DATASETS")
print("=" * 80)

# ============================================================================
# PROJECT 1: E-Commerce Sales Data (5000 records)
# ============================================================================

print("\n[STEP 1] Generating E-Commerce Sales Dataset...")

# Create date range (Jan 2023 to Mar 2024)
start_date = datetime(2023, 1, 1)
end_date = datetime(2024, 3, 31)
date_range = pd.date_range(start=start_date, end=end_date, freq='D')

# Number of transactions
n_records = 5000

# Create dictionary with all columns
data = {
    'Order_ID': [f'ORD_{i:05d}' for i in range(1, n_records + 1)],
    'Date': np.random.choice(date_range, n_records),
    'Customer_ID': np.random.randint(1000, 2500, n_records),
    'Product_Category': np.random.choice(['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books'], n_records),
    'Product_Name': np.random.choice(['Laptop', 'Phone', 'Headphones', 'T-Shirt', 'Jeans', 'Plant Pot', 'Yoga Mat', 'Dumbbell', 'Novel', 'Cookbook'], n_records),
    'Quantity': np.random.randint(1, 6, n_records),
    'Unit_Price': np.random.choice([10, 25, 50, 100, 200, 500, 1000, 1500], n_records),
    'Discount_Percent': np.random.choice([0, 5, 10, 15, 20, 25], n_records),
    'Payment_Method': np.random.choice(['Credit Card', 'Debit Card', 'PayPal', 'UPI'], n_records),
    'Shipping_Days': np.random.randint(1, 15, n_records),
    'Customer_Satisfaction': np.random.choice([1, 2, 3, 4, 5], n_records, p=[0.05, 0.1, 0.2, 0.35, 0.3])
}

# Create DataFrame
df_ecommerce = pd.DataFrame(data)

# Calculate Total_Price and Final_Price
df_ecommerce['Total_Price'] = df_ecommerce['Quantity'] * df_ecommerce['Unit_Price']
df_ecommerce['Final_Price'] = df_ecommerce['Total_Price'] * (1 - df_ecommerce['Discount_Percent'] / 100)

# Add some missing values (realistic scenario)
missing_indices = np.random.choice(df_ecommerce.index, 30, replace=False)
df_ecommerce.loc[missing_indices, 'Product_Name'] = np.nan

# Save to CSV
df_ecommerce.to_csv('Project_1_ECommerce_Sales_Analytics/data/raw_sales_data.csv', index=False)
print(f"  ✓ Saved: {len(df_ecommerce)} E-Commerce records")

# ============================================================================
# PROJECT 2: Customer Data (1500 records)
# ============================================================================

print("\n[STEP 2] Generating Customer Dataset...")

customer_data = {
    'Customer_ID': range(1000, 2500),
    'Customer_Name': [f'Customer_{i}' for i in range(1000, 2500)],
    'Email': [f'customer_{i}@email.com' for i in range(1000, 2500)],
    'Age': np.random.randint(18, 70, 1500),
    'City': np.random.choice(['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Hyderabad', 'Pune', 'Kolkata'], 1500),
    'Signup_Date': pd.date_range(start='2022-01-01', end='2024-03-31', periods=1500)
}

df_customers = pd.DataFrame(customer_data)
df_customers.to_csv('Project_2_Customer_Segmentation/data/customers.csv', index=False)
print(f"  ✓ Saved: {len(df_customers)} customer records")


# PROJECT 2: Transaction Data (8000 records)
# ============================================================================

print("\n[STEP 3] Generating Transaction Dataset...")

transaction_data = {
    'Transaction_ID': [f'TXN_{i:06d}' for i in range(1, 8001)],
    'Customer_ID': np.random.choice(df_customers['Customer_ID'], 8000),
    'Transaction_Date': np.random.choice(pd.date_range(start='2022-01-01', end='2024-03-31'), 8000),
    'Amount': np.random.uniform(100, 5000, 8000)
}

df_transactions = pd.DataFrame(transaction_data)
df_transactions.to_csv('Project_2_Customer_Segmentation/data/transactions.csv', index=False)
print(f"  ✓ Saved: {len(df_transactions)} transaction records")

print("\n" + "=" * 80)
print("✅ ALL DATASETS GENERATED SUCCESSFULLY!")
print("=" * 80)
print("\nFiles created:")
print("  1. Project_1_ECommerce_Sales_Analytics/data/raw_sales_data.csv (5000 rows)")
print("  2. Project_2_Customer_Segmentation/data/customers.csv (1500 rows)")
print("  3. Project_2_Customer_Segmentation/data/transactions.csv (8000 rows)")
print("\nReady to analyze! 🚀")