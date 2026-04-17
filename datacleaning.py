import pandas as pd
import numpy as np

# 1. Ingest Data (using chunking or specified dtypes for memory efficiency on large datasets)
# Use the actual filename inplace of accepted_dataset.csv and rejected_dataset.csv
accepted_df = pd.read_csv('accepted_dataset.csv', low_memory=False)
rejected_df = pd.read_csv('rejected_dataset.csv', low_memory=False)

# 2. Clean 'Accepted' Loans

# Select only features relevant to the product analysis
acc_cols = ['id', 'loan_amnt', 'term', 'int_rate', 'installment', 'emp_length', 
            'home_ownership', 'annual_inc', 'purpose', 'loan_status', 'dti']
accepted_clean = accepted_df[acc_cols].copy()

# Handle Missing Values (Impute median for numerical, 'Unknown' for categorical)
accepted_clean['annual_inc'] = accepted_clean['annual_inc'].fillna(accepted_clean['annual_inc'].median())
accepted_clean['emp_length'] = accepted_clean['emp_length'].fillna('Unknown')

# Feature Engineering: Income-to-Installment Ratio (Quantifying affordability)
accepted_clean['monthly_inc'] = accepted_clean['annual_inc'] / 12
accepted_clean['inc_to_installment_ratio'] = np.where(
    accepted_clean['installment'] > 0, 
    accepted_clean['monthly_inc'] / accepted_clean['installment'], 
    0
)

accepted_clean['is_defaulted'] = accepted_clean['loan_status'].apply(
    lambda x: 1 if x in ['Charged Off', 'Default', 'Does not meet the credit policy. Status:Charged Off'] else 0
)

# 3. Clean 'Rejected' Loans
rej_cols = ['Amount Requested', 'Application Date', 'Loan Title', 'Risk_Score', 
            'Debt-To-Income Ratio', 'Employment Length']
rejected_clean = rejected_df[rej_cols].copy()

rejected_clean.rename(columns={
    'Amount Requested': 'requested_amnt',
    'Application Date': 'app_date',
    'Loan Title': 'purpose',
    'Risk_Score': 'risk_score',
    'Debt-To-Income Ratio': 'dti_string',
    'Employment Length': 'emp_length'
}, inplace=True)

# Format DTI (strip '%' and convert to float)
rejected_clean['dti'] = rejected_clean['dti_string'].str.replace('%', '').astype(float) / 100
rejected_clean.drop(columns=['dti_string'], inplace=True)
rejected_clean['emp_length'] = rejected_clean['emp_length'].fillna('Unknown')

accepted_clean.to_csv('clean_accepted.csv', index=False)
rejected_clean.to_csv('clean_rejected.csv', index=False)
