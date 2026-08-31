#!/usr/bin/env python3
"""
Build a tabular feature CSV for MIMIC-IV demo in-hospital mortality prediction.

Output: /home/aqsa/sok/mimic.csv
  - One row per ICU stay (first stay per patient, adults, >=24h)
  - Static demographics + admission features
  - Aggregated (min/max/mean/std/count) vitals and labs over first 24h of ICU stay
  - Binary label: mortality (= admissions.hospital_expire_flag)
"""

import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path("/shared_data/mimiciv/mimic-iv-3.1")
HOSP = ROOT / "hosp"
ICU = ROOT / "icu"
OUT_PATH = Path("/home/aqsa/sok/mimic_full.csv")
CHECKPOINT_DIR = Path("/home/aqsa/sok/mimic_full_checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

OBS_HOURS = 24       # observation window length
MIN_LOS_HOURS = 24   # require at least this much ICU time
MIN_AGE = 18

# ----------------------------------------------------------------------------
# Curated lists of clinically standard variables.
# We resolve their itemids from d_items / d_labitems by case-insensitive
# substring match on `label` -- robust to the demo's exact spellings.
# ----------------------------------------------------------------------------
VITAL_PATTERNS = {
    "heart_rate":        ["heart rate"],
    "sbp":               ["systolic blood pressure", "arterial blood pressure systolic", "non invasive blood pressure systolic"],
    "dbp":               ["diastolic blood pressure", "arterial blood pressure diastolic", "non invasive blood pressure diastolic"],
    "mbp":               ["mean blood pressure", "arterial blood pressure mean", "non invasive blood pressure mean"],
    "resp_rate":         ["respiratory rate"],
    "spo2":              ["o2 saturation pulseoxymetry", "spo2"],
    "temperature":       ["temperature celsius", "temperature fahrenheit"],
    "gcs_total":         ["gcs - total", "glasgow coma scale total"],
}

LAB_PATTERNS = {
    "glucose":           ["glucose"],
    "creatinine":        ["creatinine"],
    "bun":               ["urea nitrogen"],
    "sodium":            ["sodium"],
    "potassium":         ["potassium"],
    "bicarbonate":       ["bicarbonate"],
    "chloride":          ["chloride"],
    "hemoglobin":        ["hemoglobin"],
    "hematocrit":        ["hematocrit"],
    "platelets":         ["platelet count"],
    "wbc":               ["white blood cells"],
    "lactate":           ["lactate"],
}

AGG_FUNCS = ["min", "max", "mean", "std", "count"]


def banner(msg):
    print("\n" + "=" * 78)
    print(msg)
    print("=" * 78)


def resolve_itemids(d_table, patterns, label_col="label"):
    """Map each variable name to the list of itemids whose label matches any pattern."""
    d = d_table.copy()
    d[label_col] = d[label_col].astype(str).str.lower()
    out = {}
    for varname, pats in patterns.items():
        mask = np.zeros(len(d), dtype=bool)
        for p in pats:
            mask |= d[label_col].str.contains(p.lower(), na=False, regex=False)
        ids = d.loc[mask, "itemid"].unique().tolist()
        out[varname] = ids
    return out


def temp_f_to_c(series, itemid_series, fahrenheit_ids):
    """Convert temperature values to Celsius if their itemid is in fahrenheit_ids."""
    is_f = itemid_series.isin(fahrenheit_ids)
    out = series.copy()
    out.loc[is_f] = (series.loc[is_f] - 32.0) * 5.0 / 9.0
    return out


def aggregate_long_to_wide(long_df, varname_col, value_col, group_keys, agg_funcs=AGG_FUNCS):
    """Pivot a long-format (group_keys, varname, value) frame to wide aggregate features."""
    g = long_df.groupby(group_keys + [varname_col])[value_col]
    pieces = []
    for fn in agg_funcs:
        s = getattr(g, fn)()
        s.name = fn
        pieces.append(s)
    agg = pd.concat(pieces, axis=1).reset_index()
    # one column per (variable, statistic)
    wide = agg.pivot_table(
        index=group_keys,
        columns=varname_col,
        values=agg_funcs,
        aggfunc="first",
    )
    wide.columns = [f"{var}_{stat}" for stat, var in wide.columns]
    return wide.reset_index()


# ============================================================================
# 1. Build cohort
# ============================================================================
banner("LOADING CORE TABLES")
patients = pd.read_csv(HOSP / "patients.csv.gz")
admissions = pd.read_csv(HOSP / "admissions.csv.gz", parse_dates=["admittime", "dischtime", "deathtime"])
icustays = pd.read_csv(ICU / "icustays.csv.gz", parse_dates=["intime", "outtime"])
print(f"patients   : {patients.shape}")
print(f"admissions : {admissions.shape}")
print(f"icustays   : {icustays.shape}")

banner("BUILDING COHORT")
# one row per icu stay, join admission + patient
cohort = icustays.merge(
    admissions[["subject_id", "hadm_id", "admittime", "admission_type",
                "hospital_expire_flag",
                "insurance", "marital_status", "race", "language",
                "admission_location"]],
    on=["subject_id", "hadm_id"], how="left"
).merge(
    patients[["subject_id", "gender", "anchor_age"]],
    on="subject_id", how="left"
)
print(f"After join (icu + adm + pat): {cohort.shape}")

# adults only
before = len(cohort)
cohort = cohort[cohort["anchor_age"] >= MIN_AGE]
print(f"After age >= {MIN_AGE}            : {len(cohort)} (dropped {before - len(cohort)})")

# ICU LOS in hours
cohort["icu_los_hours"] = (cohort["outtime"] - cohort["intime"]).dt.total_seconds() / 3600.0
before = len(cohort)
cohort = cohort[cohort["icu_los_hours"] >= MIN_LOS_HOURS]
print(f"After ICU LOS >= {MIN_LOS_HOURS} hr        : {len(cohort)} (dropped {before - len(cohort)})")

# first ICU stay per patient
cohort = cohort.sort_values(["subject_id", "intime"]).drop_duplicates("subject_id", keep="first")
print(f"After first ICU stay per pt   : {len(cohort)}")

# observation window end
cohort["window_end"] = cohort["intime"] + pd.Timedelta(hours=OBS_HOURS)

# the label
cohort = cohort.rename(columns={"hospital_expire_flag": "mortality"})
print(f"Final cohort: {len(cohort)} ICU stays")
print(f"Class balance: \n{cohort['mortality'].value_counts(dropna=False)}")
print(f"Mortality rate: {cohort['mortality'].mean():.3%}")

cohort_keys = cohort[["subject_id", "hadm_id", "stay_id", "intime", "window_end"]].copy()
cohort.to_parquet(CHECKPOINT_DIR / "cohort.parquet")
print(f"  [checkpoint saved] {CHECKPOINT_DIR / 'cohort.parquet'}")

# ============================================================================
# 2. Static features
# ============================================================================
banner("STATIC FEATURES")
static = cohort[[
    "subject_id", "hadm_id", "stay_id",
    "anchor_age", "gender", "admission_type", "first_careunit",
    "insurance", "marital_status", "race", "language", "admission_location",
]].copy()

# Bucket rare categories so we don't end up with single-patient one-hot columns.
RARE_THRESHOLD = 5
for col in ["race", "marital_status", "language", "admission_location",
            "insurance", "admission_type", "first_careunit"]:
    if col in static.columns:
        counts = static[col].value_counts(dropna=False)
        rare = counts[counts < RARE_THRESHOLD].index.tolist()
        if rare:
            print(f"  [rare-bucketing] {col}: collapsing {len(rare)} rare value(s) -> 'OTHER'")
            static[col] = static[col].where(~static[col].isin(rare), "OTHER")
print(static.head(3).to_string(index=False))

# ============================================================================
# 3. Vitals from chartevents
# ============================================================================
banner("RESOLVING VITAL ITEMIDS")
d_items = pd.read_csv(ICU / "d_items.csv.gz")
vital_ids = resolve_itemids(d_items, VITAL_PATTERNS)
for k, v in vital_ids.items():
    print(f"  {k:14s} -> {len(v)} itemid(s): {v[:6]}{'...' if len(v)>6 else ''}")
fahrenheit_ids = d_items.loc[
    d_items["label"].astype(str).str.lower().str.contains("fahrenheit", na=False),
    "itemid"
].unique().tolist()
print(f"  (Fahrenheit itemids that will be converted to C: {fahrenheit_ids})")

banner("READING chartevents (this is the biggest table)")
all_vital_ids = sorted({i for ids in vital_ids.values() for i in ids})
ce_iter = pd.read_csv(
    ICU / "chartevents.csv.gz",
    usecols=["subject_id", "hadm_id", "stay_id", "charttime", "itemid", "valuenum"],
    parse_dates=["charttime"],
    chunksize=200_000,
    low_memory=False,
)
ce_chunks = []
for chunk in ce_iter:
    chunk = chunk[chunk["itemid"].isin(all_vital_ids)]
    chunk = chunk[chunk["valuenum"].notna()]
    chunk = chunk.merge(cohort_keys, on=["subject_id", "hadm_id", "stay_id"], how="inner")
    chunk = chunk[(chunk["charttime"] >= chunk["intime"]) & (chunk["charttime"] < chunk["window_end"])]
    ce_chunks.append(chunk)
ce = pd.concat(ce_chunks, ignore_index=True) if ce_chunks else pd.DataFrame()
print(f"chartevents rows in window for cohort: {len(ce)}")

if len(ce):
    # convert F -> C for any temperature rows charted in F
    ce.loc[ce["itemid"].isin(vital_ids["temperature"]), "valuenum"] = temp_f_to_c(
        ce.loc[ce["itemid"].isin(vital_ids["temperature"]), "valuenum"],
        ce.loc[ce["itemid"].isin(vital_ids["temperature"]), "itemid"],
        fahrenheit_ids,
    )
    # map itemid -> variable name
    id_to_var = {}
    for var, ids in vital_ids.items():
        for i in ids:
            id_to_var[i] = var
    ce["varname"] = ce["itemid"].map(id_to_var)
    vitals_wide = aggregate_long_to_wide(
        ce, varname_col="varname", value_col="valuenum",
        group_keys=["subject_id", "hadm_id", "stay_id"],
    )
else:
    vitals_wide = pd.DataFrame(columns=["subject_id", "hadm_id", "stay_id"])

print(f"vitals_wide shape: {vitals_wide.shape}")
vitals_wide.to_parquet(CHECKPOINT_DIR / "vitals_wide.parquet")
print(f"  [checkpoint saved] {CHECKPOINT_DIR / 'vitals_wide.parquet'}")

# ============================================================================
# 4. Labs from labevents
# ============================================================================
banner("RESOLVING LAB ITEMIDS")
d_labitems = pd.read_csv(HOSP / "d_labitems.csv.gz")
lab_ids = resolve_itemids(d_labitems, LAB_PATTERNS)
for k, v in lab_ids.items():
    print(f"  {k:14s} -> {len(v)} itemid(s)")

banner("READING labevents")
all_lab_ids = sorted({i for ids in lab_ids.values() for i in ids})
le_iter = pd.read_csv(
    HOSP / "labevents.csv.gz",
    usecols=["subject_id", "hadm_id", "charttime", "itemid", "valuenum"],
    parse_dates=["charttime"],
    chunksize=200_000,
    low_memory=False,
)
le_chunks = []
for chunk in le_iter:
    chunk = chunk[chunk["itemid"].isin(all_lab_ids)]
    chunk = chunk[chunk["valuenum"].notna()]
    chunk = chunk.merge(
        cohort_keys[["subject_id", "hadm_id", "stay_id", "intime", "window_end"]],
        on=["subject_id", "hadm_id"], how="inner",
    )
    chunk = chunk[(chunk["charttime"] >= chunk["intime"]) & (chunk["charttime"] < chunk["window_end"])]
    le_chunks.append(chunk)
le = pd.concat(le_chunks, ignore_index=True) if le_chunks else pd.DataFrame()
print(f"labevents rows in window for cohort: {len(le)}")

if len(le):
    id_to_var = {}
    for var, ids in lab_ids.items():
        for i in ids:
            id_to_var[i] = var
    le["varname"] = le["itemid"].map(id_to_var)
    labs_wide = aggregate_long_to_wide(
        le, varname_col="varname", value_col="valuenum",
        group_keys=["subject_id", "hadm_id", "stay_id"],
    )
else:
    labs_wide = pd.DataFrame(columns=["subject_id", "hadm_id", "stay_id"])

print(f"labs_wide shape: {labs_wide.shape}")
labs_wide.to_parquet(CHECKPOINT_DIR / "labs_wide.parquet")
print(f"  [checkpoint saved] {CHECKPOINT_DIR / 'labs_wide.parquet'}")

# ============================================================================
# 5. Merge everything, impute, one-hot encode, save
# ============================================================================
banner("MERGING + ENCODING")
df = (
    cohort[["subject_id", "hadm_id", "stay_id", "mortality"]]
    .merge(static, on=["subject_id", "hadm_id", "stay_id"], how="left")
    .merge(vitals_wide, on=["subject_id", "hadm_id", "stay_id"], how="left")
    .merge(labs_wide, on=["subject_id", "hadm_id", "stay_id"], how="left")
)

# drop id columns (we keep stay_id as a row identifier for now then drop)
ids = ["subject_id", "hadm_id", "stay_id"]
label = "mortality"

# split into numeric / categorical
cat_cols = ["gender", "admission_type", "first_careunit", "insurance", "marital_status", "race", "language", "admission_location"]
num_cols = [c for c in df.columns if c not in ids + [label] + cat_cols]

# numeric: median impute
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
    med = df[c].median()
    df[c] = df[c].fillna(med if pd.notna(med) else 0.0)

# count features can stay as ints but fillna(0) is more meaningful than median
for c in num_cols:
    if c.endswith("_count"):
        df[c] = df[c].fillna(0)

# categorical: fill + one-hot
for c in cat_cols:
    df[c] = df[c].fillna("missing").astype(str)
df = pd.get_dummies(df, columns=cat_cols, drop_first=False)

# drop id columns now
df = df.drop(columns=ids)

# move label to last column for convenience
df = df[[c for c in df.columns if c != label] + [label]]

print(f"Final feature matrix shape: {df.shape}")
print(f"NaNs remaining: {df.isna().sum().sum()}")
print(f"Label distribution:")
print(df[label].value_counts())

df.to_csv(OUT_PATH, index=False)
print(f"\nWrote: {OUT_PATH}")
print(f"Columns ({len(df.columns)}):")
for c in df.columns:
    print(f"  {c}")
