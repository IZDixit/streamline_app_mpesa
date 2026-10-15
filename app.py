#!/usr/bin/env python3
"""
M-Pesa Predictive Analytics — Streamlit App
Entity resolution, phone de-masking & transactional behaviour extraction.
"""

import time
import streamlit as st
import pandas as pd
import numpy as np

from cleaner import clean_mpesa_pipeline
from tabfm_pipeline import prepare_tabfm_features, run_tabfm_predictions

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="M-Pesa Analytics | Bonje",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0d0f14; color: #e2e8f0; }
section[data-testid="stSidebar"] { background: #111318; border-right: 1px solid #1e2330; }
.kpi-card { background: #141720; border: 1px solid #1e2330; border-radius: 12px; padding: 20px 24px; }
.kpi-card:hover { box-shadow: 0 0 0 1px #38bdf8; }
.kpi-title { font-size: 11px; text-transform: uppercase; letter-spacing: .08em; color: #64748b; font-weight: 600; }
.kpi-value { font-size: 2rem; font-weight: 700; margin: 6px 0 4px; }
.kpi-sub { font-size: 12px; color: #64748b; }
.kpi-blue   .kpi-value { color: #38bdf8; }
.kpi-green  .kpi-value { color: #34d399; }
.kpi-amber  .kpi-value { color: #fbbf24; }
.kpi-purple .kpi-value { color: #a78bfa; }
.styled-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.styled-table th { background: #1a1f2e; color: #94a3b8; text-transform: uppercase; font-size: 10px; letter-spacing: .1em; padding: 10px 14px; border-bottom: 1px solid #1e2330; }
.styled-table td { padding: 11px 14px; border-bottom: 1px solid #1a1f2e; color: #e2e8f0; }
.styled-table tr:hover td { background: #161a25; }
.badge-high   { background: rgba(52,211,153,.15); color: #34d399; border: 1px solid rgba(52,211,153,.25); padding: 2px 8px; border-radius: 20px; font-size: 11px; }
.badge-medium { background: rgba(251,191,36,.12); color: #fbbf24; border: 1px solid rgba(251,191,36,.25); padding: 2px 8px; border-radius: 20px; font-size: 11px; }
.badge-low    { background: rgba(239,68,68,.12);  color: #f87171; border: 1px solid rgba(239,68,68,.25);  padding: 2px 8px; border-radius: 20px; font-size: 11px; }
.section-header { font-size: 18px; font-weight: 700; color: #f1f5f9; margin-bottom: 14px; }
.status-panel { background: #141720; border: 1px solid #1e2330; border-radius: 10px; padding: 20px; margin-top: 16px; }
.metric-highlight { font-size: 13px; color: #94a3b8; margin-bottom: 4px; }
.metric-highlight span { color: #f1f5f9; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ─── Session State Init ──────────────────────────────────────────────────────
for key in ["df", "metrics", "features_df"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ─── Sidebar Navigation ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:8px 0 20px'>
        <div style='font-size:22px;font-weight:700;color:#f1f5f9'>📊 M-Pesa Analytics</div>
        <div style='font-size:11px;color:#475569;margin-top:4px'>Bonje Predictive Pipeline</div>
    </div>
    """, unsafe_allow_html=True)
    page = st.radio("Navigation", ["🏠  Dashboard", "⚙️  Run Pipeline", "🧠  TabFM Features"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("<div style='font-size:11px;color:#334155'>Entity resolution · Phone de-masking · TabFM prep</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════
if page == "🏠  Dashboard":
    st.markdown("<h2 style='color:#f1f5f9;margin-bottom:4px'>Customer Payment Analytics</h2><p style='color:#64748b;font-size:13px'>Entity resolution, phone de-masking, and transactional behaviour extraction.</p>", unsafe_allow_html=True)

    if st.session_state.df is None:
        st.info("👋 No data loaded yet. Go to **⚙️ Run Pipeline** to upload your M-Pesa Excel file.")
        st.stop()

    df = st.session_state.df
    metrics = st.session_state.metrics
    customer_df = df[df["Is_Customer_Payment"] == 1].copy()

    # KPI Cards
    c1, c2, c3, c4 = st.columns(4)
    kpis = [
        ("c1", "kpi-blue",   "Unique Master Clients", f"{metrics['unique_master_clients']:,}", f"🔗 {metrics['unique_similar_ids']:,} Similar Groups"),
        ("c2", "kpi-green",  "Gross Customer Volume",  f"KES {metrics['total_spent']:,.0f}", "📈 Total Fueling & Merchant Payments"),
        ("c3", "kpi-amber",  "Customer Transactions",  f"{metrics['customer_payments_count']:,}", f"Total rows: {metrics['total_rows']:,}"),
        ("c4", "kpi-purple", "Asterisk Phones Resolved", f"{metrics['matched_asterisk_count']:,}", "🛡️ High/Med Pattern Matches"),
    ]
    for col, cls, title, val, sub in zip([c1, c2, c3, c4], ["kpi-blue","kpi-green","kpi-amber","kpi-purple"], *zip(*[(k[2],k[3],k[4]) for k in kpis])):
        with col:
            st.markdown(f'<div class="kpi-card {cls}"><div class="kpi-title">{title}</div><div class="kpi-value">{val}</div><div class="kpi-sub">{sub}</div></div>', unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    col_a, col_b = st.columns([2, 1])
    with col_b:
        st.markdown("<div class='section-header'>🎯 Certainty Breakdown</div>", unsafe_allow_html=True)
        cert_df = pd.DataFrame({"Level": ["High","Medium","Low"], "Count": [metrics["high_certainty_count"], metrics["medium_certainty_count"], metrics["low_certainty_count"]]})
        st.bar_chart(cert_df.set_index("Level"), color=["#34d399"])

    with col_a:
        st.markdown("<div class='section-header'>📋 Client Master Directory</div>", unsafe_allow_html=True)

        agg = customer_df.groupby("Master_Client_ID").agg(
            Client_Name=("From_Name","first"), Phone=("From_Phone","first"),
            Recovered_Phone=("Filled_Phone","first"), Similar_ID=("Similar_ID","first"),
            Certainty=("Certainty_Level","first"), Transactions=("Is_Customer_Payment","count")
        ).reset_index()

        for col_name in ["Paid In","Amount Paid"]:
            if col_name in customer_df.columns:
                customer_df["_amt"] = pd.to_numeric(customer_df[col_name].astype(str).str.replace(r"[,\s]","",regex=True), errors="coerce").fillna(0)
                if customer_df["_amt"].sum() > 0:
                    spent = customer_df.groupby("Master_Client_ID")["_amt"].sum().reset_index()
                    spent.columns = ["Master_Client_ID","Total_Spent"]
                    agg = agg.merge(spent, on="Master_Client_ID", how="left")
                    break
        if "Total_Spent" not in agg.columns:
            agg["Total_Spent"] = 0.0
        agg = agg.sort_values("Total_Spent", ascending=False).reset_index(drop=True)

        search = st.text_input("🔍 Filter by name, phone, or ID", placeholder="e.g. JOHN, 2547…, MC_1")
        cert_filter = st.selectbox("Certainty Level", ["All","High","Medium","Low"])
        filtered = agg.copy()
        if search:
            m = (filtered["Client_Name"].str.contains(search,case=False,na=False) |
                 filtered["Phone"].str.contains(search,case=False,na=False) |
                 filtered["Master_Client_ID"].str.contains(search,case=False,na=False) |
                 filtered["Similar_ID"].str.contains(search,case=False,na=False))
            filtered = filtered[m]
        if cert_filter != "All":
            filtered = filtered[filtered["Certainty"] == cert_filter]
        st.caption(f"Showing {len(filtered):,} clients")

        badge_map = {"High":"badge-high","Medium":"badge-medium","Low":"badge-low"}
        rows = ""
        for i, row in filtered.head(200).iterrows():
            cls = badge_map.get(row["Certainty"],"badge-medium")
            rows += f"""<tr>
              <td>{i+1}</td>
              <td><b>{row['Client_Name']}</b><br><span style='color:#64748b;font-size:11px'>{row['Phone']}</span></td>
              <td style='color:#38bdf8;font-family:monospace'>{row['Recovered_Phone'] or '—'}</td>
              <td style='font-family:monospace;font-size:12px'>{row['Master_Client_ID']}</td>
              <td style='font-family:monospace;font-size:12px;color:#94a3b8'>{row['Similar_ID']}</td>
              <td><span class='{cls}'>{row['Certainty']}</span></td>
              <td style='text-align:right;font-family:monospace'>KES {row['Total_Spent']:,.0f}</td>
              <td style='text-align:center'>{row['Transactions']}</td>
            </tr>"""
        st.markdown(f"""
        <div style='max-height:480px;overflow-y:auto;border-radius:10px;border:1px solid #1e2330'>
        <table class='styled-table'>
          <thead><tr><th>#</th><th>Client Name & Phone</th><th>Recovered Phone</th><th>Master Client ID</th><th>Similar ID</th><th>Certainty</th><th style='text-align:right'>Total Spent</th><th style='text-align:center'>Txns</th></tr></thead>
          <tbody>{rows}</tbody>
        </table></div>""", unsafe_allow_html=True)

    st.markdown("---")
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️  Download Full Cleaned CSV", data=csv_bytes, file_name="mpesa_cleaned.csv", mime="text/csv")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2 — RUN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════
elif page == "⚙️  Run Pipeline":
    st.markdown("<h2 style='color:#f1f5f9;margin-bottom:4px'>⚙️ Pipeline Runner</h2><p style='color:#64748b;font-size:13px'>Upload your M-Pesa Excel file (.xlsx) and run the full cleaning & entity resolution pipeline.</p>", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload M-Pesa Excel (.xlsx)", type=["xlsx","xls","csv"], help="File must have a 'Details' column in M-Pesa statement format.")
    sheet_hint = st.text_input("Sheet name (optional)", placeholder="Leave blank to auto-detect")
    run_btn = st.button("🚀 Run Pipeline", type="primary", disabled=(uploaded is None))

    if run_btn and uploaded is not None:
        file_bytes = uploaded.read()
        suffix = uploaded.name.rsplit(".",1)[-1].lower()
        tmp_path = f"/tmp/mpesa_upload_{int(time.time())}.{suffix}"
        with open(tmp_path, "wb") as f:
            f.write(file_bytes)

        with st.status("Running pipeline…", expanded=True) as status:
            st.write("📂 Reading file…")
            sheet = sheet_hint.strip() or None
            try:
                df, metrics = clean_mpesa_pipeline(tmp_path, output_csv_path=None, sheet_name=sheet)
                st.write("🧩 Building TabFM features…")
                fdf = prepare_tabfm_features(df)
                fdf = run_tabfm_predictions(fdf)
                st.session_state.df = df
                st.session_state.metrics = metrics
                st.session_state.features_df = fdf
                status.update(label="✅ Pipeline complete!", state="complete")
            except Exception as e:
                status.update(label="❌ Pipeline failed", state="error")
                st.error(str(e))
                import traceback; st.code(traceback.format_exc())
                st.stop()

        m = metrics
        st.markdown(f"""
        <div class='status-panel'>
          <div class='section-header'>📊 Pipeline Summary</div>
          <div class='metric-highlight'>Duration: <span>{m['duration_seconds']}s</span></div>
          <div class='metric-highlight'>Total Rows: <span>{m['total_rows']:,}</span></div>
          <div class='metric-highlight'>Customer Payments: <span>{m['customer_payments_count']:,}</span></div>
          <div class='metric-highlight'>Unique Master Clients: <span>{m['unique_master_clients']:,}</span></div>
          <div class='metric-highlight'>Similar Groups: <span>{m['unique_similar_ids']:,}</span></div>
          <div class='metric-highlight'>Asterisks Resolved: <span>{m['matched_asterisk_count']:,}</span></div>
          <div class='metric-highlight'>High Certainty: <span style='color:#34d399'>{m['high_certainty_count']:,}</span></div>
          <div class='metric-highlight'>Medium Certainty: <span style='color:#fbbf24'>{m['medium_certainty_count']:,}</span></div>
          <div class='metric-highlight'>Low Certainty: <span style='color:#f87171'>{m['low_certainty_count']:,}</span></div>
          <div class='metric-highlight'>Total Spent (KES): <span>{m['total_spent']:,.0f}</span></div>
        </div>""", unsafe_allow_html=True)
        st.success("✅ Data ready! Navigate to **🏠 Dashboard** to explore the results.")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3 — TABFM FEATURES
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🧠  TabFM Features":
    st.markdown("<h2 style='color:#f1f5f9;margin-bottom:4px'>🧠 TabFM Feature Preview</h2><p style='color:#64748b;font-size:13px'>Aggregated client feature vectors ready for TabFM / TabPFN inference.</p>", unsafe_allow_html=True)

    if st.session_state.features_df is None:
        st.info("👋 Run the pipeline first to generate TabFM features.")
        st.stop()

    fdf = st.session_state.features_df
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Clients", f"{len(fdf):,}")
    c2.metric("Avg Recency (days)", f"{fdf['recency_days'].mean():.1f}")
    c3.metric("Avg Txns / Client", f"{fdf['transaction_count'].mean():.1f}")
    c4.metric("High Churn Risk", f"{int((fdf['predicted_churn_risk']=='High').sum()):,}")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Churn Risk Distribution")
        st.bar_chart(fdf["predicted_churn_risk"].value_counts())
    with col2:
        st.markdown("#### Projected Monthly Spend (Top 20)")
        top20 = fdf.nlargest(20, "projected_monthly_spend")[["client_name","projected_monthly_spend"]].set_index("client_name")
        st.bar_chart(top20)

    st.markdown("---")
    st.markdown("#### Full Feature Table")
    cols = ["client_id","client_name","phone","filled_phone","total_spent","mean_spend","transaction_count","recency_days","tenure_days","certainty_level","phone_recovered","predicted_churn_risk","projected_monthly_spend"]
    st.dataframe(fdf[[c for c in cols if c in fdf.columns]].head(200), use_container_width=True, height=420)
    st.download_button("⬇️  Download Feature Vectors CSV", data=fdf.to_csv(index=False).encode(), file_name="tabfm_features.csv", mime="text/csv")
