#!/usr/bin/env python3
"""
TabFM / Foundation Model Feature Preparation Service
Prepares customer tabular features from cleaned M-Pesa transactions
ready for TabFM / TabPFN tabular prediction models in the next stage.
"""

from datetime import datetime
import pandas as pd
import numpy as np


def prepare_tabfm_features(cleaned_df):
    """
    Transforms cleaned transaction records into aggregated tabular feature vectors
    per Master Client / Similar ID for TabFM model training and inference.
    Features extracted:
    - Monetary: total_spent, mean_spend, median_spend, max_spend, spend_std
    - Frequency: transaction_count, frequency_per_month
    - Recency: days_since_last_txn, customer_tenure_days
    - Behavioral: certainty_score, phone_recovered_flag, payment_type_diversity
    """
    customer_df = cleaned_df[cleaned_df['Is_Customer_Payment'] == 1].copy()
    if customer_df.empty:
        return pd.DataFrame()

    # Normalize amounts
    def parse_amount(val):
        if pd.isna(val) or val == '':
            return 0.0
        cleaned = str(val).replace(',', '').replace(' ', '').replace('$', '')
        try:
            return float(cleaned)
        except ValueError:
            return 0.0

    customer_df['Effective_Amount'] = customer_df['Paid In'].apply(parse_amount)
    mask_zero = customer_df['Effective_Amount'] == 0
    if mask_zero.any():
        customer_df.loc[mask_zero, 'Effective_Amount'] = customer_df.loc[mask_zero, 'Amount Paid'].apply(parse_amount)

    # Parse transaction dates
    customer_df['Parsed_Date'] = pd.to_datetime(
        customer_df['Completion Time'], format='%d-%m-%Y %H:%M:%S', errors='coerce'
    )
    # Fallback parsing
    unparsed = customer_df['Parsed_Date'].isna()
    if unparsed.any():
        customer_df.loc[unparsed, 'Parsed_Date'] = pd.to_datetime(customer_df.loc[unparsed, 'Completion Time'], errors='coerce')

    max_date = customer_df['Parsed_Date'].max()
    if pd.isna(max_date):
        max_date = datetime.now()

    # Group by Master_Client_ID (or Similar_ID if missing)
    customer_df['Client_Key'] = customer_df['Master_Client_ID'].replace('', np.nan).fillna(customer_df['Similar_ID'])

    feature_rows = []
    for client_key, group in customer_df.groupby('Client_Key'):
        amounts = group['Effective_Amount'].dropna()
        dates = group['Parsed_Date'].dropna()

        total_spent = float(amounts.sum())
        mean_spend = float(amounts.mean()) if not amounts.empty else 0.0
        median_spend = float(amounts.median()) if not amounts.empty else 0.0
        max_spend = float(amounts.max()) if not amounts.empty else 0.0
        std_spend = float(amounts.std()) if len(amounts) > 1 else 0.0

        txn_count = len(group)
        recency_days = float((max_date - dates.max()).days) if not dates.empty else 999.0
        tenure_days = float((dates.max() - dates.min()).days) if not dates.empty else 0.0

        certainty_mode = group['Certainty_Level'].mode().iloc[0] if not group['Certainty_Level'].empty else 'Medium'
        has_phone_filled = int((group['Filled_Phone'] != '').any())

        feature_rows.append({
            'client_id': client_key,
            'client_name': group['From_Name'].iloc[0] if not group['From_Name'].empty else 'Unknown',
            'phone': group['From_Phone'].iloc[0] if not group['From_Phone'].empty else '',
            'filled_phone': group['Filled_Phone'].iloc[0] if not group['Filled_Phone'].empty else '',
            'total_spent': round(total_spent, 2),
            'mean_spend': round(mean_spend, 2),
            'median_spend': round(median_spend, 2),
            'max_spend': round(max_spend, 2),
            'std_spend': round(std_spend, 2),
            'transaction_count': txn_count,
            'recency_days': recency_days,
            'tenure_days': tenure_days,
            'certainty_level': certainty_mode,
            'phone_recovered': has_phone_filled,
        })

    features_df = pd.DataFrame(feature_rows)
    return features_df


def run_tabfm_predictions(features_df, target_col='next_30d_spend', model_type='mock'):
    """
    Placeholder/Hook for TabFM inference stage.
    When ready, plug in tabfm / tabpfn pretrained foundation model weights here.
    """
    if features_df.empty:
        return features_df

    predicted_df = features_df.copy()
    # Baseline analytical prediction heuristic as a placeholder until TabFM weights are linked
    predicted_df['predicted_churn_risk'] = np.where(predicted_df['recency_days'] > 60, 'High', 'Low')
    predicted_df['projected_monthly_spend'] = (predicted_df['total_spent'] / (predicted_df['tenure_days'] / 30.0 + 1.0)).round(2)
    return predicted_df
