# explore_mimic_demo.py
import os
import pandas as pd
from pathlib import Path

ROOT = Path("/home/aqsa/sok/mimic-iv-clinical-database-demo-2.2")
HOSP = ROOT / "hosp"
ICU = ROOT / "icu"

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

print("=" * 80)
print("MIMIC-IV CLINICAL DATABASE DEMO 2.2 — STRUCTURE")
print("=" * 80)

# -----------------------------------------------------------------------------
# 1. Top-level files
# -----------------------------------------------------------------------------
print("\n--- Top-level files ---")
for p in sorted(ROOT.iterdir()):
    size = p.stat().st_size if p.is_file() else "(dir)"
    print(f"  {p.name:40s} {size}")

# -----------------------------------------------------------------------------
# 2. demo_subject_id.csv — the cohort
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("demo_subject_id.csv")
print("=" * 80)
demo_ids = pd.read_csv(ROOT / "demo_subject_id.csv")
print(f"Shape: {demo_ids.shape}")
print(f"Columns: {demo_ids.columns.tolist()}")
print(demo_ids.head())

# -----------------------------------------------------------------------------
# 3. Inventory of hosp/ and icu/
# -----------------------------------------------------------------------------
def inventory(folder, label):
    print("\n" + "=" * 80)
    print(f"{label}/  inventory")
    print("=" * 80)
    rows = []
    for p in sorted(folder.glob("*.csv*")):
        size_mb = p.stat().st_size / (1024 * 1024)
        rows.append((p.name, f"{size_mb:.2f} MB"))
    for name, size in rows:
        print(f"  {name:40s} {size}")
    return [p for p in sorted(folder.glob("*.csv*"))]

hosp_files = inventory(HOSP, "hosp")
icu_files = inventory(ICU, "icu")

# -----------------------------------------------------------------------------
# 4. For each table: shape, columns, dtypes, head(2), missing summary
# -----------------------------------------------------------------------------
def summarize_table(path):
    print("\n" + "-" * 80)
    print(f"TABLE: {path.parent.name}/{path.name}")
    print("-" * 80)
    try:
        # Read a small sample first to get columns / dtypes cheaply
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        print(f"  [ERROR reading: {e}]")
        return
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print("\nDtypes:")
    print(df.dtypes)
    print("\nHead(2):")
    with pd.option_context("display.max_colwidth", 80):
        print(df.head(2))
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing):
        print("\nMissing values (cols with >0 NaN):")
        print(missing.sort_values(ascending=False))
    else:
        print("\nMissing values: none")
    # Show unique counts for low-cardinality cols (potential labels/categoricals)
    low_card = []
    for c in df.columns:
        try:
            n = df[c].nunique(dropna=True)
            if 2 <= n <= 20:
                low_card.append((c, n))
        except Exception:
            pass
    if low_card:
        print("\nLow-cardinality columns (2–20 unique, candidate categoricals/labels):")
        for c, n in low_card:
            vc = df[c].value_counts(dropna=False).head(10)
            print(f"  {c} ({n} unique):")
            for k, v in vc.items():
                print(f"      {k!r:40s} {v}")

print("\n" + "#" * 80)
print("# HOSP TABLES")
print("#" * 80)
for p in hosp_files:
    summarize_table(p)

print("\n" + "#" * 80)
print("# ICU TABLES")
print("#" * 80)
for p in icu_files:
    summarize_table(p)

# -----------------------------------------------------------------------------
# 5. Quick cohort sanity check from admissions
# -----------------------------------------------------------------------------
adm_path = HOSP / "admissions.csv.gz"
if not adm_path.exists():
    adm_path = HOSP / "admissions.csv"
if adm_path.exists():
    print("\n" + "=" * 80)
    print("COHORT SANITY CHECK (from admissions)")
    print("=" * 80)
    adm = pd.read_csv(adm_path, low_memory=False)
    print(f"Total admissions: {len(adm)}")
    print(f"Unique patients : {adm['subject_id'].nunique()}")
    if "hospital_expire_flag" in adm.columns:
        print("\nhospital_expire_flag (in-hospital mortality) distribution:")
        print(adm["hospital_expire_flag"].value_counts(dropna=False))
        rate = adm["hospital_expire_flag"].mean()
        print(f"In-hospital mortality rate: {rate:.3%}")
    if "admission_type" in adm.columns:
        print("\nadmission_type:")
        print(adm["admission_type"].value_counts(dropna=False))
