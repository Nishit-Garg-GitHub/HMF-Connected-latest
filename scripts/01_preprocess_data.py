# ============================================================
# HMF ELECTROOXIDATION DATASET
# COMPLETE DATA READING + PREPROCESSING
# Python 3.12
# ============================================================
import pandas as pd
import numpy as np
import re
from pathlib import Path

from descriptors import (
    ANODE_SPLIT, anode_text, counter_electrode_text, extract_cell_voltage,
    extract_current, extract_electrolyte_anion, extract_electrolyte_cation,
    extract_electrolyte_concentration, extract_electrolyte_ph, extract_first_number,
    extract_potential, extract_temperature,
)
# ============================================================
# 1. FILE PATH
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
# The script normally lives in <project>/scripts. Keep a fallback for running
# it from the project root without changing the user's established filenames.
ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name.lower() == "scripts" else SCRIPT_DIR
DATA_DIR = ROOT / "data" / "processed"
RAW_DIR = ROOT / "data" / "raw"
SOURCE_NAME = "Nishith_Summary Table of different cell types.xlsx"
FILE_CANDIDATES = [
    RAW_DIR / SOURCE_NAME,   # canonical location
    ROOT / SOURCE_NAME,
    SCRIPT_DIR / SOURCE_NAME,
]
FILE_PATH = next((path for path in FILE_CANDIDATES if path.exists()), FILE_CANDIDATES[0])
OUTPUT_FULL = DATA_DIR / "HMF_Cleaned_Full_Dataset.xlsx"
OUTPUT_ML = DATA_DIR / "HMF_ML_Preprocessed.xlsx"
OUTPUT_CSV = DATA_DIR / "HMF_ML_Preprocessed.csv"
DATA_DIR.mkdir(parents=True, exist_ok=True)
if not FILE_PATH.exists():
    raise FileNotFoundError(
        f"Source workbook not found. Checked: {', '.join(map(str, FILE_CANDIDATES))}"
    )
# ============================================================
# 2. READ EXCEL DATA
# ============================================================
print("=" * 70)
print("READING DATASET")
print("=" * 70)
df = pd.read_excel(FILE_PATH)
print("Dataset loaded successfully.")
print("Shape:", df.shape)
print("\nColumns:")
for i, col in enumerate(df.columns, 1):
    print(f"{i}. {col}")
# ============================================================
# 3. CLEAN COLUMN NAMES
# ============================================================
df.columns = (
    df.columns
    .astype(str)
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
)
# ============================================================
# 4. RENAME IMPORTANT COLUMNS
# ============================================================
rename_dict = {
    "HMF Concentration (mM)": "HMF_Concentration_mM",
    "Potential (V vs. RHE)": "Potential_raw",
    "Conversion (%)": "Conversion_pct",
    "Product/Sel. (%)": "Selectivity_pct",
    "TEMPO Used (Yes/No)": "TEMPO",
    "Yield (%)": "Yield_pct",
    "FE (%)": "FE_pct",
    "Reactant monomer": "Reactant"
}
df.rename(
    columns=rename_dict,
    inplace=True
)
# ============================================================
# 5. STANDARDIZE MISSING VALUES
# ============================================================
missing_values = [
    "",
    " ",
    "NA",
    "N/A",
    "n/a",
    "na",
    "Not specified",
    "Not specified in paper",
    "Not reported",
    "Not mentioned",
    "NR",
    "Unknown",
    "-",
    "--"
]
df.replace(
    missing_values,
    np.nan,
    inplace=True
)
# Strip whitespace from text columns
for col in df.select_dtypes(
    include="object"
).columns:
    df[col] = df[col].apply(
        lambda x: x.strip()
        if isinstance(x, str)
        else x
    )
# ============================================================
# 6. HELPER FUNCTION
# ============================================================
# ============================================================
# 7. HMF CONCENTRATION
# ============================================================
df["HMF_Concentration_mM"] = (
    df["HMF_Concentration_mM"]
    .apply(extract_first_number)
)
print("\n" + "=" * 70)
print("HMF CONCENTRATION")
print("=" * 70)
print(
    df["HMF_Concentration_mM"]
    .describe()
)
# ============================================================
# 8. POTENTIAL AND CURRENT
# ============================================================




# Potential_V is retained as a backward-compatible alias, but now contains
# only explicitly RHE-referenced electrode potentials.
df["Potential_vs_RHE_V"] = df["Potential_raw"].apply(extract_potential)
df["Potential_V"] = df["Potential_vs_RHE_V"]
df["Cell_Voltage_V"] = df["Potential_raw"].apply(extract_cell_voltage)
df["Current_mA"] = (
    df["Potential_raw"]
    .apply(extract_current)
)
# ------------------------------------------------------------
# Potential vs RHE
# ------------------------------------------------------------
df["Potential_vs_RHE"] = (
    df["Potential_raw"]
    .astype(str)
    .str.contains(
        "RHE",
        case=False,
        na=False
    )
    .astype(int)
)
# ------------------------------------------------------------
# Current-controlled experiment
# ------------------------------------------------------------
df["Current_Controlled"] = (
    df["Potential_raw"]
    .astype(str)
    .str.contains(
        r"constant\s+current|galvanostatic|"
        r"current\s+density|mA",
        case=False,
        na=False,
        regex=True
    )
    .astype(int)
)
# ------------------------------------------------------------
# Cell voltage reported
# ------------------------------------------------------------
df["Cell_Voltage"] = (
    df["Potential_raw"]
    .astype(str)
    .str.contains(
        "cell voltage",
        case=False,
        na=False
    )
    .astype(int)
)
print("\nPotential / Current information:")
print(
    df[
        [
            "Potential_raw",
            "Potential_V",
            "Current_mA",
            "Potential_vs_RHE",
            "Current_Controlled",
            "Cell_Voltage"
        ]
    ].head(25)
)
# ============================================================
# 9. CONVERSION
# ============================================================
df["Conversion_pct"] = (
    df["Conversion_pct"]
    .apply(extract_first_number)
)
# ============================================================
# 10. SELECTIVITY
# ============================================================
df["Selectivity_pct"] = (
    df["Selectivity_pct"]
    .apply(extract_first_number)
)
# ============================================================
# 11. YIELD
# ============================================================
df["Yield_pct"] = (
    df["Yield_pct"]
    .apply(extract_first_number)
)
## ============================================================
# 12. FARADAIC EFFICIENCY
# ============================================================
df["FE_pct"] = (
    df["FE_pct"]
    .apply(extract_first_number)
)
# ============================================================
# 13. TEMPERATURE
# ============================================================
df["Temperature_C"] = (
    df["Temperature"]
    .apply(extract_temperature)
)
# Flag values where 25 °C was inferred
df["Temperature_Imputed"] = (
    df["Temperature"]
    .astype(str)
    .str.contains(
        "room temperature|ambient",
        case=False,
        na=False
    )
    .astype(int)
)
# ============================================================
# 14. TEMPO
# ============================================================
df["TEMPO_binary"] = (
    df["TEMPO"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({
        "yes": 1,
        "no": 0
    })
)
# ============================================================
# 15. ELECTROLYTE FEATURES
# ============================================================


df["Electrolyte_Concentration_M"] = (
    df["Electrolyte"]
    .apply(extract_electrolyte_concentration)
)
df["Electrolyte_Cation"] = (
    df["Electrolyte"]
    .apply(extract_electrolyte_cation)
)
df["Electrolyte_Anion"] = (
    df["Electrolyte"]
    .apply(extract_electrolyte_anion)
)
# ============================================================
# 16. ALKALINE ELECTROLYTE
# ============================================================
df["Alkaline_Electrolyte"] = (
    df["Electrolyte"]
    .astype(str)
    .str.contains(
        r"KOH|NaOH|LiOH|RbOH|CsOH",
        case=False,
        na=False,
        regex=True
    )
    .astype(int)
)
# ============================================================
# 17. CATALYST FEATURES (ANODE / WORKING ELECTRODE ONLY)
# ============================================================
# The "Catalysts" field describes the whole cell, e.g.
# "Anode: Ni3N@C on Ni foam. Cathode: Pt foil". HMF is oxidised at the
# anode, so counter/reference electrodes (very often Pt, graphite or
# Ag/AgCl) must not be counted as the catalyst. Element symbols are matched
# case-sensitively on chemical formulas (Ni3S2, CoNiP, VN, Co9S8), with
# element names as a fallback (nickel foam, platinum gauze).





df["Catalyst_Anode_Text"] = df["Catalysts"].apply(anode_text)
anode_raw = df["Catalyst_Anode_Text"].fillna("").astype(str)
catalyst_text = anode_raw.str.lower()

# Symbol: capitalised symbol not followed by a lower-case letter, so "Co" in
# "Co3O4" matches but "Co" in "Commercial" does not.
catalyst_elements = {
    "Ni": (r"Ni(?![a-z])", r"nickel"),
    "Co": (r"Co(?![a-z])", r"cobalt"),
    "Fe": (r"Fe(?![a-z])", r"\biron\b"),
    "Cu": (r"Cu(?![a-z])", r"copper"),
    "Pt": (r"Pt(?![a-z])", r"platin"),
    "Pd": (r"Pd(?![a-z])", r"palladium"),
    "Au": (r"Au(?![a-z])", r"\bgold\b"),
    "Ru": (r"Ru(?![a-z])", r"ruthenium"),
    "Ir": (r"Ir(?![a-z])", r"iridium"),
    "Rh": (r"Rh(?![a-z])", r"rhodium"),
    "Mn": (r"Mn(?![a-z])", r"manganese"),
    "Mo": (r"Mo(?![a-z])", r"molybden"),
    "W": (r"(?<![A-Z])W(?![a-z])", r"tungsten"),
    "V": (r"(?<![A-Z])V(?![a-z])", r"vanadium"),
    "Bi": (r"Bi(?![a-z])", r"bismuth"),
    "Ce": (r"Ce(?![a-z])", r"cerium"),
    "Zn": (r"Zn(?![a-z])", r"\bzinc\b"),
    "Ag": (r"Ag(?![a-z])", r"\bsilver\b"),
}

for element, (symbol, name) in catalyst_elements.items():
    df[f"Catalyst_{element}"] = (
        anode_raw.str.contains(symbol, regex=True, na=False)
        | catalyst_text.str.contains(name, regex=True, na=False)
    ).astype(int)

TRANSITION = ["Ni", "Co", "Fe", "Cu", "Mn", "Mo", "W", "V", "Zn"]
NOBLE = ["Pt", "Pd", "Au", "Ru", "Ir", "Rh", "Ag"]
df["Catalyst_N_Elements"] = df[
    [f"Catalyst_{e}" for e in catalyst_elements]
].sum(axis=1)
df["Catalyst_Noble_Metal"] = df[[f"Catalyst_{e}" for e in NOBLE]].max(axis=1)
df["Catalyst_Bimetallic"] = (
    df[[f"Catalyst_{e}" for e in TRANSITION]].sum(axis=1) >= 2
).astype(int)

# ============================================================
# 18. CATALYST FAMILY AND SUBSTRATE FEATURES
# ============================================================
# Each family is (case-sensitive formula pattern, lower-case word pattern).
# A non-metal symbol counts only when it directly follows the lower-case
# letter of a metal symbol, a stoichiometric digit/subscript, "x", ")" or "-"
# (Ni3S2, CoNiP, CuxS, Ni₂P, Co-P). This keeps "NF" (nickel foam), "CFP"
# (carbon fibre paper), "FTO" and "DES" from being read as N, P, O or S.
FORMULA_PREFIX = r"(?<=[a-z0-9x\)\-\u2080-\u2089])"
catalyst_families = {
    "Catalyst_Oxide":
        (FORMULA_PREFIX + r"(?<!H\d)(?<!H)O[\d\u2080-\u2089x]*(?![a-zA-Z\(H])|[VWT]i?O\d",
         r"oxide(?!.*hydr)|\bnio\b"),
    "Catalyst_Hydroxide":
        (r"\(OH\)\d|OH(?![a-z])(?<!OOH)", r"hydroxide"),
    "Catalyst_Oxyhydroxide":
        (r"OOH", r"oxyhydroxide"),
    "Catalyst_LDH":
        (r"LDH", r"layered double hydroxide"),
    "Catalyst_Phosphide":
        (FORMULA_PREFIX + r"P[\d\u2080-\u2089]*(?![a-zA-Z])|_P\b", r"phosph"),
    "Catalyst_Nitride":
        (r"(?<=[A-Za-z0-9\)])N[\d\u2080-\u2089]*(?![a-zA-Z])", r"nitride"),
    "Catalyst_Sulfide":
        (FORMULA_PREFIX + r"S(?:x|[\d\u2080-\u2089])*(?![a-zA-Z])", r"sulfide|sulphide"),
    "Catalyst_Selenide":
        (r"Se(?![a-z])", r"selenide"),
    "Catalyst_Boride":
        (FORMULA_PREFIX + r"B(?:x|\d)*(?![a-zA-Z])", r"boride"),
    "Catalyst_MOF":
        (r"MOF", r"metal[- ]organic framework"),
    "Catalyst_Carbon":
        (r"@C\b|/C\b|NC\b|CNT", r"carbon|graphit|graphene|nanotube"),
    "Catalyst_SingleAtom":
        (r"-SA\b|SA/", r"single[- ]atom"),
    "Catalyst_Nanoparticle":
        (r"NPs?\b", r"nano"),
    "Catalyst_Metal":
        (r"$^", r"metallic|\bbulk\b|\bfoil\b|\bwire\b|plain"),
    # Electrode substrate / architecture descriptors.
    "Catalyst_Foam_Substrate":
        (r"\bNF\b|\bCF\b|\bNiF\b", r"foam"),
    "Catalyst_Conductive_Glass":
        (r"FTO|ITO|BDD", r"glass"),
}

for feature, (formula, words) in catalyst_families.items():
    df[feature] = (
        anode_raw.str.contains(formula, regex=True, na=False)
        | catalyst_text.str.contains(words, regex=True, na=False)
    ).astype(int)

# Counter electrode (cathode) descriptors belong to the cell-configuration
# input group: a Pt counter in a divided cell differs from a bifunctional
# two-electrode electrolyser with the same material on both sides.


counter_raw = df["Catalysts"].apply(counter_electrode_text)
df["Counter_Electrode_Reported"] = (counter_raw.str.strip().str.len() > 0).astype(int)
df["Counter_Electrode_Pt"] = counter_raw.str.contains(
    r"Pt(?![a-z])|platin", case=False, regex=True, na=False).astype(int)
df["Counter_Electrode_Carbon"] = counter_raw.str.contains(
    r"carbon|graphite", case=False, regex=True, na=False).astype(int)
df["Counter_Electrode_Ni"] = counter_raw.str.contains(
    r"Ni(?![a-z])|nickel", regex=True, na=False).astype(int)

# Electrolyte pH: explicit "pH = 13" first, otherwise inferred from a
# hydroxide concentration (pH = 14 + log10[OH-]); neutral salts -> 7.


df["Electrolyte_pH"] = df["Electrolyte"].apply(extract_electrolyte_ph)

# ============================================================
# 19. CELL TYPE FEATURES
# ============================================================
cell_text = (
    df["Cell Type"]
    .fillna("")
    .astype(str)
    .str.lower()
)
df["Divided_Cell"] = (
    cell_text
    .str.contains(
        r"divided|h-type|h shaped|h-shaped",
        regex=True,
        na=False
    )
    .astype(int)
)
df["Undivided_Cell"] = (
    cell_text
    .str.contains(
        "undivided",
        na=False
    )
    .astype(int)
)
df["Flow_Cell"] = (
    cell_text
    .str.contains(
        "flow",
        na=False
    )
    .astype(int)
)
df["Three_Electrode"] = (
    cell_text
    .str.contains(
        r"three[- ]electrode",
        regex=True,
        na=False
    )
    .astype(int)
)
# ============================================================
# 20. MAJOR PRODUCT
# ============================================================
df["Major_Product_Clean"] = (
    df["Major Product"]
    .fillna("")
    .astype(str)
    .str.upper()
    .str.strip()
)
# ============================================================
# 21. FDCA TARGET FLAG
# ============================================================
df["FDCA_Target"] = (
    df["Major_Product_Clean"]
    .str.contains(
        "FDCA",
        na=False
    )
    .astype(int)
)
# ============================================================
# 22. CHECK WHETHER REACTANT IS HMF
# ============================================================
df["Is_HMF"] = (
    df["Reactant"]
    .astype(str)
    .str.contains(
        r"HMF|5-Hydroxymethylfurfural",
        case=False,
        na=False,
        regex=True
    )
    .astype(int)
)
# ============================================================
# 22b. DERIVE MISSING FDCA YIELD FROM CONVERSION x SELECTIVITY
# ============================================================
# Yield = conversion x selectivity holds by definition for the major product.
# It is applied only when the reported yield is missing, conversion and
# selectivity are both reported (0-100) and the major product is FDCA. Derived
# rows are flagged; no value is ever imputed from a median.
df["Yield_pct_reported"] = df["Yield_pct"]
derivable = (
    df["Yield_pct"].isna()
    & df["Conversion_pct"].between(0, 100)
    & df["Selectivity_pct"].between(0, 100)
    & (df["FDCA_Target"] == 1)
)
df.loc[derivable, "Yield_pct"] = (
    df.loc[derivable, "Conversion_pct"] * df.loc[derivable, "Selectivity_pct"] / 100.0
)
df["Yield_Derived"] = derivable.astype(int)
print(f"Yield derived from conversion x selectivity for {int(derivable.sum())} records.")
# ============================================================
# 23. DATA CONSISTENCY CHECKS
# ============================================================
# ------------------------------------------------------------
# Yield > conversion
# ------------------------------------------------------------
df["Yield_Greater_Than_Conversion"] = (
    (
        df["Yield_pct"] >
        df["Conversion_pct"]
    )
    &
    df["Yield_pct"].notna()
    &
    df["Conversion_pct"].notna()
).astype(int)
# ------------------------------------------------------------
# FE > 100
# ------------------------------------------------------------
df["FE_Above_100"] = (
    (
        df["FE_pct"] > 100
    )
    &
    df["FE_pct"].notna()
).astype(int)
# ------------------------------------------------------------
# Conversion > 100
# ------------------------------------------------------------
df["Conversion_Above_100"] = (
    (
        df["Conversion_pct"] > 100
    )
    &
    df["Conversion_pct"].notna()
).astype(int)
# ------------------------------------------------------------
# Yield > 100
# ------------------------------------------------------------
df["Yield_Above_100"] = (
    (
        df["Yield_pct"] > 100
    )
    &
    df["Yield_pct"].notna()
).astype(int)
# ============================================================
# 24. DEFINE ML FEATURES
# ============================================================
ml_features = [
    # --------------------------------------------------------
    # PROCESS VARIABLES
    # --------------------------------------------------------
    "HMF_Concentration_mM",
    "Potential_vs_RHE_V",
    "Potential_V",
    "Current_mA",
    "Cell_Voltage",
    "Cell_Voltage_V",
    "Current_Controlled",
    "Potential_vs_RHE",
    "Temperature_C",
    "TEMPO_binary",
    # --------------------------------------------------------
    # ELECTROLYTE
    # --------------------------------------------------------
    "Electrolyte_Concentration_M",
    "Electrolyte_pH",
    "Alkaline_Electrolyte",
    # --------------------------------------------------------
    # CELL
    # --------------------------------------------------------
    "Divided_Cell",
    "Undivided_Cell",
    "Flow_Cell",
    "Three_Electrode",
    "Counter_Electrode_Reported",
    "Counter_Electrode_Pt",
    "Counter_Electrode_Carbon",
    "Counter_Electrode_Ni",
    # --------------------------------------------------------
    # CATALYST ELEMENTS
    # --------------------------------------------------------
    "Catalyst_Ni",
    "Catalyst_Co",
    "Catalyst_Fe",
    "Catalyst_Cu",
    "Catalyst_Pt",
    "Catalyst_Pd",
    "Catalyst_Au",
    "Catalyst_Ru",
    "Catalyst_Ir",
    "Catalyst_Mn",
    "Catalyst_Mo",
    "Catalyst_W",
    "Catalyst_Ce",
    "Catalyst_Zn",
    "Catalyst_Ag",
    "Catalyst_Rh",
    "Catalyst_V",
    "Catalyst_Bi",
    "Catalyst_N_Elements",
    "Catalyst_Noble_Metal",
    "Catalyst_Bimetallic",
    # --------------------------------------------------------
    # CATALYST FAMILIES
    # --------------------------------------------------------
    "Catalyst_Oxide",
    "Catalyst_Hydroxide",
    "Catalyst_Oxyhydroxide",
    "Catalyst_LDH",
    "Catalyst_Phosphide",
    "Catalyst_Nitride",
    "Catalyst_Sulfide",
    "Catalyst_Selenide",
    "Catalyst_Boride",
    "Catalyst_MOF",
    "Catalyst_Carbon",
    "Catalyst_SingleAtom",
    "Catalyst_Nanoparticle",
    "Catalyst_Metal",
    "Catalyst_Foam_Substrate",
    "Catalyst_Conductive_Glass",
]
# Keep only columns that exist
ml_features = [
    col
    for col in ml_features
    if col in df.columns
]
# ============================================================
# 25. DEFINE TARGET VARIABLES
# ============================================================
targets = [
    "Conversion_pct",
    "Yield_pct",
    "FE_pct",
    "Selectivity_pct"
]
# ============================================================
# 26. CREATE ML DATAFRAME
# ============================================================
ml_df = df[
    ml_features + targets
].copy()
# ============================================================
# 27. ONE-HOT ENCODE ELECTROLYTE
# ============================================================
electrolyte_features = pd.get_dummies(
    df[
        [
            "Electrolyte_Cation",
            "Electrolyte_Anion"
        ]
    ],
    dummy_na=True,
    dtype=int
)
# Rename generated columns
electrolyte_features.columns = [
    f"Electrolyte_{col}"
    for col in electrolyte_features.columns
]
# Combine
ml_df = pd.concat(
    [
        ml_df.reset_index(drop=True),
        electrolyte_features.reset_index(drop=True)
    ],
    axis=1
)
# ============================================================
# 28. REMOVE DUPLICATE COLUMNS
# ============================================================
ml_df = ml_df.loc[
    :,
    ~ml_df.columns.duplicated()
]
# ============================================================
# 29. CREATE DATA QUALITY REPORT
# ============================================================
missing_report = (
    ml_df
    .isna()
    .sum()
    .sort_values(
        ascending=False
    )
)
missing_report = missing_report[
    missing_report > 0
]
# ============================================================
# 30. PRINT MISSING VALUE REPORT
# ============================================================
print("\n")
print("=" * 70)
print("MISSING VALUE REPORT")
print("=" * 70)
if len(missing_report) == 0:
    print("No missing values.")
else:
    print(missing_report)
# ============================================================
# 31. TARGET SUMMARY
# ============================================================
print("\n")
print("=" * 70)
print("TARGET VARIABLE SUMMARY")
print("=" * 70)
print(
    ml_df[
        [
            "Conversion_pct",
            "Yield_pct",
            "FE_pct",
            "Selectivity_pct"
        ]
    ].describe()
)
# ============================================================
# 32. DATA QUALITY FLAGS
# ============================================================
print("\n")
print("=" * 70)
print("DATA QUALITY CHECKS")
print("=" * 70)
print(
    "Yield > Conversion:",
    df["Yield_Greater_Than_Conversion"].sum()
)
print(
    "FE > 100:",
    df["FE_Above_100"].sum()
)
print(
    "Conversion > 100:",
    df["Conversion_Above_100"].sum()
)
print(
    "Yield > 100:",
    df["Yield_Above_100"].sum()
)
# ============================================================
# 33. PROCESS VARIABLE SUMMARY
# ============================================================
print("\n")
print("=" * 70)
print("PROCESS VARIABLE SUMMARY")
print("=" * 70)
process_columns = [
    "HMF_Concentration_mM",
    "Potential_V",
    "Current_mA",
    "Temperature_C",
    "Electrolyte_Concentration_M"
]
print(
    df[
        process_columns
    ].describe()
)
# ============================================================
# 34. CATEGORICAL SUMMARY
# ============================================================
print("\n")
print("=" * 70)
print("ELECTROLYTE CATION COUNTS")
print("=" * 70)
print(
    df["Electrolyte_Cation"]
    .value_counts(dropna=False)
)
print("\n")
print("=" * 70)
print("ELECTROLYTE ANION COUNTS")
print("=" * 70)
print(
    df["Electrolyte_Anion"]
    .value_counts(dropna=False)
)
print("\n")
print("=" * 70)
print("TEMPO COUNTS")
print("=" * 70)
print(
    df["TEMPO"]
    .value_counts(dropna=False)
)
# ============================================================
# 35. CATALYST FAMILY COUNTS
# ============================================================
print("\n")
print("=" * 70)
print("CATALYST FAMILY COUNTS")
print("=" * 70)
catalyst_family_columns = [
    "Catalyst_Oxide",
    "Catalyst_Hydroxide",
    "Catalyst_Oxyhydroxide",
    "Catalyst_LDH",
    "Catalyst_Phosphide",
    "Catalyst_Nitride",
    "Catalyst_Sulfide",
    "Catalyst_Selenide",
    "Catalyst_MOF",
    "Catalyst_Carbon",
    "Catalyst_SingleAtom",
    "Catalyst_Nanoparticle",
    "Catalyst_Metal"
]
for col in catalyst_family_columns:
    if col in df.columns:
        print(
            f"{col}:",
            df[col].sum()
        )
# ============================================================
# 36. SAVE FULL CLEANED DATASET
# ============================================================
df.to_excel(
    OUTPUT_FULL,
    index=False
)
# ============================================================
# 37. SAVE ML DATASET
# ============================================================
ml_df.to_excel(
    OUTPUT_ML,
    index=False
)
# ============================================================
# 38. SAVE CSV
# ============================================================
ml_df.to_csv(
    OUTPUT_CSV,
    index=False
)
# ============================================================
# 39. FINAL SUMMARY
# ============================================================
print("\n")
print("=" * 70)
print("PREPROCESSING COMPLETE")
print("=" * 70)
print(
    "Original dataset shape:",
    df.shape
)
print(
    "ML dataset shape:",
    ml_df.shape
)
print(
    "Number of ML features:",
    len(ml_features)
)
print(
    "Number of target variables:",
    len(targets)
)
print("\nFiles created:")
print(
    OUTPUT_FULL
)
print(
    OUTPUT_ML
)
print(
    OUTPUT_CSV
)
# ============================================================
# 40. SHOW FIRST 10 ML ROWS
# ============================================================
print("\n")
print("=" * 70)
print("FIRST 10 ROWS OF ML DATASET")
print("=" * 70)
print(
    ml_df.head(10).to_string()
)
# ============================================================
# END
# ============================================================
print("\n")
print("=" * 70)
print("READY FOR NEXT STAGE: EDA + ML")
print("=" * 70)
