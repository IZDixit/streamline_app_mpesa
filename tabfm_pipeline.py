#!/usr/bin/env python3
"""
TabFM / Foundation Model Feature Preparation Service
Prepares customer tabular features from cleaned M-Pesa transactions
ready for TabFM / TabPFN tabular prediction models in the next stage.

Performance: fully vectorised — no Python-level per-group loops.
"""

from datetime import datetime
import pandas as pd
import numpy as np


def prepare_tabfm_features(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms cleaned transaction records into aggregated tabular feature vectors
    per Master Client / Similar ID for TabFM model training and inference.

    Features extracted:
    - Monetary: total_spent, mean_spend, median_spend, max_spend, spend_std
    - Frequency: transaction_count
    - Recency: recency_days, tenure_days
    - Behavioral: certainty_level, phone_recovered_flag

    Implementation: fully vectorised groupby — O(N log N), no Python for-loops.
    """
    customer_df = cleaned_df[cleaned_df["Is_Customer_Payment"] == 1].copy()
    if customer_df.empty:
        return pd.DataFrame()

    # ── 1. Vectorised amount parsing ─────────────────────────────────────────
    def _vec_parse(series: pd.Series) -> pd.Series:
        return pd.to_numeric(
            series.astype(str)
                  .str.replace(r"[,\s\$]", "", regex=True)
                  .replace({"nan": np.nan, "": np.nan}),
            errors="coerce",
        ).fillna(0.0)

    paid_in   = _vec_parse(customer_df.get("Paid In",   pd.Series(0.0, index=customer_df.index)))
    amt_paid  = _vec_parse(customer_df.get("Amount Paid", pd.Series(0.0, index=customer_df.index)))
    customer_df["Effective_Amount"] = np.where(paid_in > 0, paid_in, amt_paid)

    # ── 2. Vectorised date parsing ───────────────────────────────────────────
    customer_df["Parsed_Date"] = pd.to_datetime(
        customer_df.get("Completion Time"), format="%d-%m-%Y %H:%M:%S", errors="coerce"
    )
    # Fallback for non-standard formats
    mask_unparsed = customer_df["Parsed_Date"].isna()
    if mask_unparsed.any():
        customer_df.loc[mask_unparsed, "Parsed_Date"] = pd.to_datetime(
            customer_df.loc[mask_unparsed, "Completion Time"], errors="coerce"
        )

    max_date = customer_df["Parsed_Date"].max()
    if pd.isna(max_date):
        max_date = pd.Timestamp(datetime.now())

    # ── 3. Client key ────────────────────────────────────────────────────────
    customer_df["Client_Key"] = (
        customer_df["Master_Client_ID"]
        .replace("", np.nan)
        .fillna(customer_df.get("Similar_ID", pd.Series("", index=customer_df.index)))
    )

    customer_df["Filled_Phone"] = customer_df.get("Filled_Phone", pd.Series("", index=customer_df.index)).fillna("").astype(str)

    # ── 4. All aggregations in one vectorised groupby pass ───────────────────
    grp = customer_df.groupby("Client_Key", sort=False)

    # Monetary
    agg_money = grp["Effective_Amount"].agg(
        total_spent="sum",
        mean_spend="mean",
        median_spend="median",
        max_spend="max",
        std_spend="std",
    ).fillna(0.0).round(2)

    # Frequency
    agg_count = grp["Is_Customer_Payment"].count().rename("transaction_count")

    # Dates
    agg_dates = grp["Parsed_Date"].agg(last_date="max", first_date="min")
    agg_dates["recency_days"] = (max_date - agg_dates["last_date"]).dt.days.fillna(999).astype(float)
    agg_dates["tenure_days"]  = (agg_dates["last_date"] - agg_dates["first_date"]).dt.days.fillna(0).astype(float)

    # Behavioural — first values
    agg_meta = grp.agg(
        client_name=("From_Name",    "first"),
        phone      =("From_Phone",   "first"),
        filled_phone=("Filled_Phone","first"),
        similar_id =("Similar_ID",   "first") if "Similar_ID" in customer_df.columns else ("Client_Key", "first"),
    )

    # Certainty mode (most common value per client) — fast path via value_counts
    if "Certainty_Level" in customer_df.columns:
        vc = (
            customer_df.groupby(["Client_Key", "Certainty_Level"])
            .size()
            .rename("_cnt")
            .reset_index()
        )
        certainty_mode = (
            vc.loc[vc.groupby("Client_Key")["_cnt"].idxmax()]
            .set_index("Client_Key")["Certainty_Level"]
            .rename("certainty_level")
        )
    else:
        certainty_mode = pd.Series("Low", index=agg_money.index, name="certainty_level")

    # Phone recovered flag (vectorised any())
    phone_recovered = (
        grp["Filled_Phone"]
        .agg(lambda x: int((x != "").any()))
        .rename("phone_recovered")
    )

    # ── 5. Assemble ─────────────────────────────────────────────────────────
    features_df = (
        agg_money
        .join(agg_count)
        .join(agg_dates[["recency_days", "tenure_days"]])
        .join(agg_meta)
        .join(certainty_mode)
        .join(phone_recovered)
    )
    features_df.index.name = "client_id"
    features_df = features_df.reset_index()

    # Normalise string columns
    for col in ("client_name", "phone", "filled_phone"):
        features_df[col] = features_df[col].fillna("").astype(str)

    return features_df


def run_tabfm_predictions(features_df: pd.DataFrame, target_col: str = "next_30d_spend", model_type: str = "mock") -> pd.DataFrame:
    """
    Placeholder/Hook for TabFM inference stage.
    When ready, plug in tabfm / tabpfn pretrained foundation model weights here.
    """
    if features_df.empty:
        return features_df

    predicted_df = features_df.copy()
    # Baseline heuristic until TabFM weights are linked
    predicted_df["predicted_churn_risk"] = np.where(predicted_df["recency_days"] > 60, "High", "Low")
    predicted_df["projected_monthly_spend"] = (
        predicted_df["total_spent"] / (predicted_df["tenure_days"] / 30.0 + 1.0)
    ).round(2)
    return predicted_df
