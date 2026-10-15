#!/usr/bin/env python3
"""
M-Pesa Data Cleaning & Client Linking Pipeline Service
Ported from final_working_solution_with_master.py for Django application.
Features:
- Regex extraction of customer phone, name, and payment types from M-Pesa details.
- Two-pass asterisk/phone grouping to recover masked phone numbers.
- Master Client linking using Union-Find algorithm.
- High/Medium/Low certainty scoring and match reasoning.
"""

from collections import defaultdict
import os
from pathlib import Path
import re
import time
import pandas as pd
import numpy as np


def extract_customer_payment_info(details_str):
    """Extract phone and name for customer payments."""
    if pd.isna(details_str) or not isinstance(details_str, str) or not details_str.strip():
        return "", "", 0

    details_str = details_str.strip()

    # Pattern 1: Merchant Payment from [phone] - [name]
    pattern1 = r'Merchant\s+Payment\s+from\s+(254[0-9*]+|0[0-9*]+)\s*-\s*(.*)'
    match1 = re.search(pattern1, details_str, re.IGNORECASE)
    if match1:
        return match1.group(1), match1.group(2).strip(), 1

    # Pattern 2: Merchant Payment Online from [phone] - [name]
    pattern2 = r'Merchant\s+Payment\s+Online\s+from\s+(254[0-9*]+|0[0-9*]+)\s*-\s*(.*)'
    match2 = re.search(pattern2, details_str, re.IGNORECASE)
    if match2:
        return match2.group(1), match2.group(2).strip(), 1

    # Pattern 3: Business Buy Goods by [number] - [name] via API REF [ref]
    pattern3 = r'Business\s+Buy\s+Goods\s+by\s+(.+?)\s*-\s*(.+?)\s+via\s+API\s+REF(?:\s+(.*))?'
    match3 = re.search(pattern3, details_str, re.IGNORECASE)
    if match3:
        name = match3.group(2).strip()
        api_ref = match3.group(3) if match3.group(3) else ""
        phone = ""
        if api_ref:
            phone_ref_match = re.search(r'(254[0-9]{9,})', api_ref)
            if phone_ref_match:
                phone = phone_ref_match.group(1)
        if not phone:
            first_part = match3.group(1).strip()
            if re.match(r'^(254[0-9]+|0[0-9]+)$', first_part):
                phone = first_part
        return phone, name, 1

    # Pattern 4: Small Business Pay Merchant from [phone] - [name]
    pattern4 = r'Small\s+Business\s+Pay\s+Merchant\s+from\s+(254[0-9*]+|0[0-9*]+)\s*-\s*(.*)'
    match4 = re.search(pattern4, details_str, re.IGNORECASE)
    if match4:
        return match4.group(1), match4.group(2).strip(), 1

    # Pattern 5: FSI to Merchantby [number] - [name] via API REF [ref]
    pattern5 = r'FSI\s+to\s+Merchantby\s+(.+?)\s*-\s*(.+?)\s+via\s+API\s+REF(?:\s+(.*))?'
    match5 = re.search(pattern5, details_str, re.IGNORECASE)
    if match5:
        name = match5.group(2).strip()
        api_ref = match5.group(3) if match5.group(3) else ""
        phone = ""
        if api_ref:
            phone_ref_match = re.search(r'(254[0-9]{9,})', api_ref)
            if phone_ref_match:
                phone = phone_ref_match.group(1)
        if not phone:
            first_part = match5.group(1).strip()
            if re.match(r'^(254[0-9]+|0[0-9]+)$', first_part):
                phone = first_part
        return phone, name, 1

    # Pattern 6: FSI Merchant Interoperability [number] - [name] via API REF [ref]
    pattern6 = r'FSI\s+Merchant\s+Interoperability\s+(.+?)\s*-\s*(.+?)\s+via\s+API\s+REF(?:\s+(.*))?'
    match6 = re.search(pattern6, details_str, re.IGNORECASE)
    if match6:
        name = match6.group(2).strip()
        api_ref = match6.group(3) if match6.group(3) else ""
        phone = ""
        if api_ref:
            phone_ref_match = re.search(r'(254[0-9]{9,})', api_ref)
            if phone_ref_match:
                phone = phone_ref_match.group(1)
        if not phone:
            first_part = match6.group(1).strip()
            if re.match(r'^(254[0-9]+|0[0-9]+)$', first_part):
                phone = first_part
        return phone, name, 1

    return "", "", 0


def get_phone_visible_digits(phone):
    """Extract only the visible (non-asterisk) digits from a phone number, in order."""
    if pd.isna(phone) or phone == '':
        return ""
    return ''.join(c for c in str(phone) if c != '*')


def get_phone_prefix(phone, length=3):
    """Get the first N digits of a phone (for bucketing)."""
    visible = get_phone_visible_digits(phone)
    if not visible:
        return ""
    if visible.startswith('254'):
        visible = visible[3:]
    elif visible.startswith('0'):
        visible = visible[1:]
    return visible[:length] if len(visible) >= length else visible


def get_phone_suffix(phone, length=3):
    """Get the last N digits of a phone (for bucketing)."""
    visible = get_phone_visible_digits(phone)
    if not visible:
        return ""
    if visible.startswith('254'):
        visible = visible[3:]
    elif visible.startswith('0'):
        visible = visible[1:]
    return visible[-length:] if len(visible) >= length else visible


def get_name_prefix(name):
    """Get the first word of a name (before any asterisks or spaces)."""
    if pd.isna(name) or name == '':
        return ""
    clean = str(name).split('*')[0].strip()
    if not clean:
        clean = str(name).replace('*', '').strip()
    parts = clean.split()
    return parts[0].lower() if parts else ""


def name_matches(asterisk_name, full_name):
    """Check if an asterisk name matches a full name."""
    if not asterisk_name or not full_name:
        return False
    visible = asterisk_name.replace('*', '').strip()
    if not visible:
        return False
    return full_name.upper().startswith(visible.upper())


def phone_matches(asterisk_phone, full_phone):
    """Check if an asterisk phone matches a full phone by comparing non-asterisk positions."""
    if not asterisk_phone or not full_phone:
        return False

    a = str(asterisk_phone)
    f = str(full_phone)

    if len(a) != len(f):
        return False

    for i in range(len(a)):
        if a[i] != '*':
            if a[i] != f[i]:
                return False
    return True


def normalize_phone_for_master_matching(phone_value):
    """Normalize a phone number to a comparable, asterisk-free value."""
    if pd.isna(phone_value):
        return ""
    raw = str(phone_value).strip()
    if raw in ('', 'nan', 'NaN'):
        return ""
    digits = re.sub(r'\D', '', raw)
    if not digits:
        return ""
    if digits.startswith('254'):
        return digits
    if digits.startswith('0'):
        return '254' + digits[1:]
    if len(digits) == 9:
        return '254' + digits
    return digits


def normalize_name_for_master_matching(name_value):
    """Normalize a name so near-identical customer names can be linked."""
    if pd.isna(name_value):
        return ""
    name = str(name_value).upper().strip()
    name = name.replace('*', ' ')
    name = re.sub(r'[^A-Z0-9\s]', ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def create_master_client_links(df):
    """
    Link all customer-payment rows that represent the same client using Union-Find.
    Scales efficiently across large transaction volumes.
    """
    customer_payments = df[df['Is_Customer_Payment'] == 1]
    parent = {}
    key_to_root = {}
    master_counter = 1

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a, b):
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return ra
        if ra < rb:
            parent[rb] = ra
            return ra
        parent[ra] = rb
        return rb

    def get_or_create_root(value):
        if value not in parent:
            parent[value] = value
        return value

    def register_key(key, node_id):
        if not key:
            return
        if key in key_to_root:
            union(node_id, key_to_root[key])
        else:
            key_to_root[key] = node_id

    # First pass: attach each row to a node and merge by exact phone/name/similar-id keys
    for idx, row in customer_payments.iterrows():
        node_id = f"row_{idx}"
        get_or_create_root(node_id)
        phone = normalize_phone_for_master_matching(row.get('From_Phone', ''))
        filled_phone = normalize_phone_for_master_matching(row.get('Filled_Phone', ''))
        name = normalize_name_for_master_matching(row.get('From_Name', ''))
        similar_id = str(row.get('Similar_ID', '')).strip() if pd.notna(row.get('Similar_ID', '')) else ''

        if phone:
            register_key(f'phone:{phone}', node_id)
        if filled_phone:
            register_key(f'phone:{filled_phone}', node_id)
        if name:
            register_key(f'name:{name}', node_id)
        if similar_id and similar_id not in ('nan', 'NaN'):
            register_key(f'similar:{similar_id}', node_id)

    # Second pass: assign a stable master client ID to each connected component
    component_roots = {}
    for idx in customer_payments.index:
        node_id = f"row_{idx}"
        root = find(node_id)
        if root not in component_roots:
            component_roots[root] = f"MC_{master_counter}"
            master_counter += 1
        df.loc[idx, 'Master_Client_ID'] = component_roots[root]

    return df


def clean_mpesa_pipeline(input_file_path, output_csv_path=None, sheet_name=None):
    """
    Complete end-to-end cleaning pipeline.
    Reads Excel or CSV, extracts customer information, resolves masked phone numbers,
    assigns certainty scores, builds master client identifiers, and optionally writes output CSV.
    Returns (cleaned_df, summary_metrics).
    """
    start_time = time.time()
    input_path = Path(input_file_path)

    # Determine reader based on file extension
    ext = input_path.suffix.lower()
    if ext in ['.xlsx', '.xls']:
        excel_kwargs = {'header': 1}
        if sheet_name:
            excel_kwargs['sheet_name'] = sheet_name
        else:
            # Check sheet names, default to '001.' if present or first sheet
            xl = pd.ExcelFile(input_file_path)
            if '001.' in xl.sheet_names:
                excel_kwargs['sheet_name'] = '001.'
            else:
                excel_kwargs['sheet_name'] = 0
        df = pd.read_excel(input_file_path, **excel_kwargs)
    elif ext == '.csv':
        df = pd.read_csv(input_file_path, low_memory=False)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Expected .xlsx, .xls, or .csv")

    # Remove unwanted trailing/empty columns
    unwanted_columns = [
        'INV. AMT.', 'LTRS.', 'nan.1', 'Unnamed: 12', 'Unnamed: 13',
        'Unnamed: 14', 'Unnamed: 15', 'Unnamed: 16', 'Unnamed: 17'
    ]
    df = df.drop(columns=[col for col in unwanted_columns if col in df.columns], errors='ignore')

    # Ensure required columns exist
    if 'Details' not in df.columns:
        raise ValueError("Input file must contain a 'Details' column.")

    # Initialize new columns
    df['From_Name'] = ''
    df['From_Phone'] = ''
    df['Is_Customer_Payment'] = 0
    df['Has_Asterisks_Name'] = 0
    df['Has_Asterisks_Phone'] = 0
    df['Similar_ID'] = ''
    df['Certainty_Level'] = ''
    df['Filled_Phone'] = ''
    df['Match_Reason'] = ''
    df['Master_Client_ID'] = ''

    # PASS 1: Extract customer details
    for idx, row in df.iterrows():
        details = row['Details']
        phone, name, is_customer = extract_customer_payment_info(details)

        df.loc[idx, 'From_Name'] = name
        df.loc[idx, 'From_Phone'] = phone
        df.loc[idx, 'Is_Customer_Payment'] = is_customer

        if is_customer:
            df.loc[idx, 'Has_Asterisks_Name'] = 1 if '*' in str(name) else 0
            df.loc[idx, 'Has_Asterisks_Phone'] = 1 if '*' in str(phone) else 0

    # PASS 2: Two-pass grouping & phone recovery
    customer_idx = df[df['Is_Customer_Payment'] == 1].index.tolist()

    non_asterisk_phones = []
    asterisk_phones = []
    blank_phones = []

    for idx in customer_idx:
        phone = str(df.loc[idx, 'From_Phone'])
        name = str(df.loc[idx, 'From_Name'])

        if phone == '' or phone == 'nan':
            blank_phones.append((idx, name))
        elif '*' in phone:
            asterisk_phones.append((idx, phone, name))
        else:
            non_asterisk_phones.append((idx, phone, name))

    phone_groups = defaultdict(list)
    name_groups = defaultdict(list)

    for idx, phone, name in non_asterisk_phones:
        phone_groups[phone].append(idx)
        name_groups[name.strip().lower()].append(idx)

    for idx, name in blank_phones:
        name_groups[name.strip().lower()].append(idx)

    similar_id_counter = 1
    group_assignments = {}

    for phone, indices in phone_groups.items():
        similar_id = f"ID_{similar_id_counter}"
        certainty = 'High' if len(indices) > 1 else 'Medium'
        for idx in indices:
            group_assignments[idx] = (similar_id, certainty)
        similar_id_counter += 1

    for name_lower, indices in name_groups.items():
        unassigned = [idx for idx in indices if idx not in group_assignments]
        if unassigned:
            similar_id = f"ID_{similar_id_counter}"
            certainty = 'Medium' if len(unassigned) > 1 else 'Low'
            for idx in unassigned:
                group_assignments[idx] = (similar_id, certainty)
            similar_id_counter += 1

    # Phone and name buckets for asterisk matching
    phone_buckets = defaultdict(list)
    for idx, phone, name in non_asterisk_phones:
        prefix = get_phone_prefix(phone, 3)
        suffix = get_phone_suffix(phone, 3)
        if prefix and suffix:
            phone_buckets[(prefix, suffix)].append((phone, idx))

    name_buckets = defaultdict(list)
    for idx, phone, name in non_asterisk_phones:
        name_prefix = get_name_prefix(name)
        if name_prefix:
            name_buckets[name_prefix].append((phone, idx, name))

    matched_asterisk_count = 0
    for idx, phone, name in asterisk_phones:
        matched = False
        prefix = get_phone_prefix(phone, 3)
        suffix = get_phone_suffix(phone, 3)

        if prefix and suffix:
            bucket_key = (prefix, suffix)
            if bucket_key in phone_buckets:
                for full_phone, full_idx in phone_buckets[bucket_key]:
                    if phone_matches(phone, full_phone):
                        full_name = str(df.loc[full_idx, 'From_Name'])
                        if '*' not in name or name_matches(name, full_name) or full_name == name:
                            existing_id, _ = group_assignments[full_idx]
                            group_assignments[idx] = (existing_id, 'Medium')
                            df.loc[idx, 'Filled_Phone'] = full_phone
                            df.loc[idx, 'Match_Reason'] = 'Phone pattern match'
                            matched = True
                            matched_asterisk_count += 1
                            break

        if not matched:
            name_prefix = get_name_prefix(name)
            if name_prefix and name_prefix in name_buckets:
                for full_phone, full_idx, full_name in name_buckets[name_prefix]:
                    if name_matches(name, full_name):
                        if '*' in phone and full_phone:
                            if phone_matches(phone, full_phone):
                                existing_id, _ = group_assignments[full_idx]
                                group_assignments[idx] = (existing_id, 'High')
                                df.loc[idx, 'Filled_Phone'] = full_phone
                                df.loc[idx, 'Match_Reason'] = 'Name + Phone match'
                                matched = True
                                matched_asterisk_count += 1
                                break
                        elif phone == '' or phone == 'nan':
                            existing_id, _ = group_assignments[full_idx]
                            group_assignments[idx] = (existing_id, 'Medium')
                            df.loc[idx, 'Match_Reason'] = 'Name match'
                            matched = True
                            matched_asterisk_count += 1
                            break

        if not matched:
            similar_id = f"ID_{similar_id_counter}"
            group_assignments[idx] = (similar_id, 'Low')
            similar_id_counter += 1

    # Group remaining asterisk-only pairs
    unmatched_asterisk_idx = [
        idx for idx, phone, name in asterisk_phones
        if idx in group_assignments and group_assignments[idx][1] == 'Low'
    ]
    asterisk_pattern_groups = defaultdict(list)
    for idx in unmatched_asterisk_idx:
        phone = str(df.loc[idx, 'From_Phone'])
        name = str(df.loc[idx, 'From_Name'])
        pattern = phone.replace('*', '')
        name_prefix = get_name_prefix(name)
        asterisk_pattern_groups[(pattern, name_prefix)].append(idx)

    for (pattern, name_prefix), indices in asterisk_pattern_groups.items():
        if len(indices) > 1:
            first_idx = indices[0]
            existing_id, _ = group_assignments[first_idx]
            for idx in indices[1:]:
                group_assignments[idx] = (existing_id, 'Medium')

    # Assign group IDs to dataframe
    for idx in customer_idx:
        if idx in group_assignments:
            similar_id, certainty = group_assignments[idx]
            df.loc[idx, 'Similar_ID'] = similar_id
            df.loc[idx, 'Certainty_Level'] = certainty
        else:
            similar_id = f"ID_{similar_id_counter}"
            group_assignments[idx] = (similar_id, 'Low')
            df.loc[idx, 'Similar_ID'] = similar_id
            df.loc[idx, 'Certainty_Level'] = 'Low'
            similar_id_counter += 1

    # Fill phone numbers for asterisk records
    id_to_phone = {}
    for idx in customer_idx:
        if idx in group_assignments:
            sid, _ = group_assignments[idx]
            phone = str(df.loc[idx, 'From_Phone'])
            if '*' not in phone and phone != '' and phone != 'nan':
                if sid not in id_to_phone:
                    id_to_phone[sid] = phone

    for idx in customer_idx:
        if idx in group_assignments and df.loc[idx, 'Has_Asterisks_Phone'] == 1:
            sid, _ = group_assignments[idx]
            if sid in id_to_phone and df.loc[idx, 'Filled_Phone'] == '':
                df.loc[idx, 'Filled_Phone'] = id_to_phone[sid]
                if df.loc[idx, 'Certainty_Level'] == 'Low':
                    df.loc[idx, 'Certainty_Level'] = 'Medium'

    # Master client linking (Union-Find)
    df = create_master_client_links(df)

    # Save output CSV if requested
    if output_csv_path:
        out_path = Path(output_csv_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Ensure phone columns are formatted as string
        for col in ['From_Phone', 'Filled_Phone']:
            if col in df.columns:
                df[col] = df[col].astype(str).replace('nan', '').replace('.0', '')
        df.to_csv(out_path, index=False)

    # Calculate summary metrics
    duration = round(time.time() - start_time, 2)
    customer_payments = df[df['Is_Customer_Payment'] == 1]
    
    # Calculate total spent
    total_spent = 0.0
    for col in ['Paid In', 'Amount Paid']:
        if col in customer_payments.columns:
            cleaned_col = pd.to_numeric(
                customer_payments[col].astype(str).str.replace(r'[,\s]', '', regex=True),
                errors='coerce'
            ).fillna(0)
            total_spent = float(cleaned_col.sum())
            if total_spent > 0:
                break

    metrics = {
        'total_rows': len(df),
        'customer_payments_count': len(customer_payments),
        'unique_similar_ids': int(customer_payments['Similar_ID'].nunique()) if not customer_payments.empty else 0,
        'unique_master_clients': int(customer_payments['Master_Client_ID'].nunique()) if not customer_payments.empty else 0,
        'high_certainty_count': int((customer_payments['Certainty_Level'] == 'High').sum()),
        'medium_certainty_count': int((customer_payments['Certainty_Level'] == 'Medium').sum()),
        'low_certainty_count': int((customer_payments['Certainty_Level'] == 'Low').sum()),
        'matched_asterisk_count': matched_asterisk_count,
        'total_spent': total_spent,
        'duration_seconds': duration,
        'output_csv_path': str(output_csv_path) if output_csv_path else None
    }

    return df, metrics
