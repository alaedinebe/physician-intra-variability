# -*- coding: utf-8 -*-
"""Baseline table utilities for Prism analysis."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# Standardized clinical features for Table 1 (outcome 'recommendation' excluded).
BASELINE_CONTINUOUS = [
    "age",
    "biomarker_hba1c_ngsp_blood",
    "non_hdl_cholesterol",
    "hdl_cholesterol",
    "ldl_cholesterol",
    "systolic_blood_pressure",
    "diastolic_blood_pressure",
    "estimated_glomerular_filtration_rate",
    "score_questionnaire_HAD",
]

# Publication-ready labels with units (order: demographics, then biomarkers).
BASELINE_LABELS = {
    "age": "Age (years)",
    "is_male": "Male sex",
    "biomarker_hba1c_ngsp_blood": "HbA1c (%)",
    "systolic_blood_pressure": "Systolic Blood Pressure (mmHg)",
    "diastolic_blood_pressure": "Diastolic Blood Pressure (mmHg)",
    "non_hdl_cholesterol": "Non-HDL Cholesterol (mmol/L)",
    "hdl_cholesterol": "HDL Cholesterol (mmol/L)",
    "ldl_cholesterol": "LDL Cholesterol (mmol/L)",
    "estimated_glomerular_filtration_rate": "eGFR (mL/min/1.73 m²)",
    "score_questionnaire_HAD": "HAD questionnaire score",
    "is_smoker": "Current Smoker",
    "has_car": "Has a car",
}

# Column name used for stratification by dataset strategy labels.
STRATEGY_COL_CANDIDATES = ("dataset_strategy", "strategy_name")


def generate_baseline_table_one(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    Generate Table 1: Baseline Patient Characteristics in NEJM/BMJ style.

    Uses the standardized clinical features (including lipids in mmol/L); the outcome variable (recommendation)
    is excluded. Continuous variables are reported as Mean (SD); categorical as n (%).
    If a strategy column is present (e.g. dataset_strategy), columns are stratified
    by strategy with a "Total Cohort" column first; otherwise only Total Cohort is shown.

    Arguments:
        df: Patient-level DataFrame with baseline features (and optionally strategy).
        output_dir: Experiment root; files are written under ``output_dir/data/``.

    Returns:
        DataFrame with publication-ready labels as index and statistic columns (Total and/or per strategy).
    """
    output_dir = Path(output_dir)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Detect stratification column (optional).
    strategy_col = None
    for cand in STRATEGY_COL_CANDIDATES:
        if cand in df.columns:
            strategy_col = cand
            break

    # Build list of group labels and corresponding subsets: Total first, then by strategy.
    group_labels = ["Total Cohort"]
    groups = [df]
    if strategy_col is not None:
        strategy_order = df[strategy_col].dropna().unique().tolist()
        for s in strategy_order:
            group_labels.append(str(s))
            groups.append(df.loc[df[strategy_col] == s])

    # Collect rows: variable label -> list of formatted strings (one per column).
    row_order = (
        ["age", "is_male", "biomarker_hba1c_ngsp_blood", "systolic_blood_pressure",
         "diastolic_blood_pressure", "non_hdl_cholesterol", "hdl_cholesterol",
         "ldl_cholesterol", "estimated_glomerular_filtration_rate",
         "score_questionnaire_HAD", "is_smoker", "has_car"]
    )
    continuous = set(BASELINE_CONTINUOUS)
    available = [c for c in row_order if c in df.columns]
    if not available:
        raise ValueError(
            "No baseline columns found in DataFrame. Expected at least one of: %s."
            % ", ".join(row_order)
        )

    table_rows = {}
    for var in row_order:
        if var not in df.columns:
            continue
        label = BASELINE_LABELS.get(var, var)
        values = []
        for g in groups:
            sub = g[var].dropna()
            n = len(sub)
            if var in continuous:
                if n == 0:
                    values.append("—")
                else:
                    mean_val = sub.mean()
                    sd_val = sub.std()
                    if pd.isna(sd_val) or n == 1:
                        sd_val = 0.0
                    values.append("%.1f (%.1f)" % (mean_val, sd_val))
            else:
                # Categorical: count and percentage (e.g. level 1 or True).
                if n == 0:
                    values.append("0 (0.0%)")
                else:
                    count = (sub.astype(int) == 1).sum()
                    pct = 100.0 * count / n
                    values.append("%d (%.1f%%)" % (count, pct))
        table_rows[label] = values

    # Build DataFrame: index = variable labels, columns = group labels.
    index_labels = [BASELINE_LABELS[var] for var in row_order if var in df.columns]
    rows_data = [table_rows[BASELINE_LABELS[var]] for var in row_order if var in df.columns]
    table_df = pd.DataFrame(rows_data, index=index_labels, columns=group_labels)
    table_df.index.name = "Characteristic"

    # Prepend sample size row (N per column).
    n_row = [len(g) for g in groups]
    n_df = pd.DataFrame([n_row], index=["N"], columns=group_labels)
    table_df = pd.concat([n_df, table_df], axis=0)

    # Save LaTeX (index = Characteristic) and CSV under data/.
    tex_path = data_dir / "table_1_baseline.tex"
    table_df.to_latex(tex_path, index=True, escape=False)
    csv_path = data_dir / "table_1_baseline.csv"
    table_df.to_csv(csv_path)

    return table_df
