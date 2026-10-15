# M-Pesa Predictive Analytics — Streamlit App

Upload M-Pesa Excel statements and instantly get:
- **Entity resolution** — groups masked/unmasked phone records into unique client identities
- **Phone de-masking** — recovers `254712***280` → `254712030280`
- **Master Client IDs** — Union-Find linking across all payment variants
- **TabFM features** — per-client tabular feature vectors for churn & spend prediction

---

## 🚀 Deploy to Streamlit Cloud

1. Push this folder to a **public GitHub repo**
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in
3. Click **New app** → select your repo
4. Set **Main file path**: `streamlit_app/app.py`  
   *(or `app.py` if you placed it at the repo root)*
5. Click **Deploy** — done!

---

## 🖥️ Run Locally

```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 File Structure

```
streamlit_app/
├── app.py              # Main Streamlit application
├── cleaner.py          # Core pipeline: extraction, phone matching, Union-Find
├── tabfm_pipeline.py   # TabFM feature extraction & baseline predictions
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## 📂 Input Format

Your Excel file must:
- Contain a sheet named `001.` (or any first sheet) with headers on **row 2**
- Have a **`Details`** column with M-Pesa transaction descriptions
- Match M-Pesa merchant statement format (Merchant Payment, Business Buy Goods, etc.)
