# ============================================================
# HMF ELECTROOXIDATION DATASET
# COMPLETE DATA READING + PREPROCESSING
# Python 3.12
# ============================================================
import pandas as pd
import numpy as np
import re
from pathlib import Path
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
    SCRIPT_DIR / SOURCE_NAME,
    ROOT / SOURCE_NAME,
    RAW_DIR / SOURCE_NAME,
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
def extract_first_number(value):
    """
    Extract first numerical value from a string.
    Examples:
    '5 mM'              -> 5
    '1.47 V vs. RHE'    -> 1.47
    '~100%'             -> 100
    '99.5%'             -> 99.5
    """
    if pd.isna(value):
        return np.nan
    text = str(value)
    match = re.search(
        r"-?\d+(?:\.\d+)?",
        text
    )
    if match:
        return float(
            match.group()
        )
    return np.nan
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
def extract_potential(value):
    """
    Extract a numerical electrode potential only when the text explicitly
    references RHE. Cell voltage, vs Ag/AgCl, vs SCE, current and current
    density are not interchangeable with potential vs RHE and remain missing.
    """
    if pd.isna(value):
        return np.nan
    text = str(value)
    if re.search(r"constant\s+current|galvanostatic|current\s+density", text, re.I):
        return np.nan
    if not re.search(r"\bRHE\b", text, re.I):
        return np.nan
    match = re.search(r"(-?\d+(?:\.\d+)?)\s*V", text, re.I)
    return float(match.group(1)) if match else np.nan


def extract_cell_voltage(value):
    """Extract cell voltage only when the record labels it as cell voltage."""
    if pd.isna(value):
        return np.nan
    text = str(value)
    if not re.search(r"cell\s+voltage|full[- ]cell\s+voltage", text, re.I):
        return np.nan
    match = re.search(r"(-?\d+(?:\.\d+)?)\s*V", text, re.I)
    return float(match.group(1)) if match else np.nan


def extract_current(value):
    """
    Extract current in mA when explicitly reported.
    """
    if pd.isna(value):
        return np.nan
    text = str(value)
    match = re.search(
        r"(-?\d+(?:\.\d+)?)\s*mA",
        text,
        flags=re.IGNORECASE
    )
    if match:
        return float(
            match.group(1)
        )
    return np.nan
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
def extract_temperature(value):
    if pd.isna(value):
        return np.nan
    text = str(value).lower()
    # Explicit temperature, e.g. 25 °C, 55°C
    match = re.search(
        r"(-?\d+(?:\.\d+)?)\s*°?\s*c",
        text
    )
    if match:
        return float(match.group(1))
    # Room temperature
    if "room temperature" in text:
        return 25.0
    # Ambient
    if "ambient" in text:
        return 25.0
    return np.nan
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
def extract_electrolyte_concentration(value):
    if pd.isna(value):
        return np.nan
    text = str(value)
    # Examples:
    # 1 M KOH
    # 0.1 M KOH
    # 2 mM electrolyte
    match = re.search(
        r"(\d+(?:\.\d+)?)\s*(mM|M)\b",
        text,
        flags=re.IGNORECASE
    )
    if not match:
        return np.nan
    concentration = float(
        match.group(1)
    )
    unit = match.group(2).lower()
    # Convert everything to M
    if unit == "mm":
        concentration /= 1000
    return concentration
def extract_electrolyte_cation(value):
    """Identify common alkali/ammonium cations in written salt formulas."""
    if pd.isna(value):
        return np.nan
    text = str(value)
    # Formula-aware patterns handle attached formulas such as KOH and Na2SO4.
    ion_patterns = {
        "NH4": r"NH4(?:\+)?",
        "Li": r"Li(?:[0-9]|\+|OH|2SO4|Cl|NO3)",
        "Na": r"Na(?:[0-9]|\+|OH|2SO4|Cl|NO3|HCO3|2CO3)",
        "K": r"K(?:[0-9]|\+|OH|2SO4|Cl|NO3|HCO3|2CO3)",
        "Rb": r"Rb(?:[0-9]|\+|OH|2SO4|Cl|NO3)",
        "Cs": r"Cs(?:[0-9]|\+|OH|2SO4|Cl|NO3)",
    }
    for ion, pattern in ion_patterns.items():
        if re.search(pattern, text, flags=re.IGNORECASE):
            return ion
    return np.nan


def extract_electrolyte_anion(value):
    if pd.isna(value):
        return np.nan
    text = str(value)
    anions = [
        "OH",
        "SO4",
        "HCO3",
        "CO3",
        "ClO4",
        "NO3",
        "PO4",
        "Cl"
    ]
    for anion in anions:
        if re.search(
            anion,
            text,
            flags=re.IGNORECASE
        ):
            return anion
    return np.nan
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
# 17. CATALYST FEATURES
# ============================================================
catalyst_text = (
    df["Catalysts"]
    .fillna("")
    .astype(str)
    .str.lower()
)
# ------------------------------------------------------------
# Metal / element indicators
# ------------------------------------------------------------
catalyst_elements = {
    "Ni":
        r"\bni\b|niooh|ni\(oh\)2|nico|nife|nifo|nifep",
    "Co":
        r"\bco\b|coooh|co3o4|co\(oh\)2|nico",
    "Fe":
        r"\bfe\b|feooh|fe2o3|fe3o4|nife",
    "Cu":
        r"\bcu\b|cuo|cu2o|cu\(oh\)2",
    "Pt":
        r"\bpt\b|pt/c",
    "Pd":
        r"\bpd\b|pd/c",
    "Au":
        r"\bau\b|au/c",
    "Ru":
        r"\bru\b|ruo2",
    "Ir":
        r"\bir\b|iro2",
    "Mn":
        r"\bmn\b|mno2|mno",
    "Mo":
        r"\bmo\b|moo",
    "W":
        r"\bw\b|wo3",
    "Ce":
        r"\bce\b|ceo2",
    "Zn":
        r"\bzn\b|zno",
    "Ag":
        r"\bag\b"
}
for element, pattern in catalyst_elements.items():
    df[f"Catalyst_{element}"] = (
        catalyst_text
        .str.contains(
            pattern,
            regex=True,
            na=False
        )
        .astype(int)
    )
# ============================================================
# 18. CATALYST FAMILY FEATURES
# ============================================================
catalyst_families = {
    "Catalyst_Oxide":
        r"oxide|nio|co3o4|iro2|ruo2|zno|ceo2|feo",
    "Catalyst_Hydroxide":
        r"hydroxide|\(oh\)2",
    "Catalyst_Oxyhydroxide":
        r"oxyhydroxide|ooh",
    "Catalyst_LDH":
        r"\bldh\b|layered double hydroxide",
    "Catalyst_Phosphide":
        r"phosphide",
    "Catalyst_Nitride":
        r"nitride",
    "Catalyst_Sulfide":
        r"sulfide|sulphide",
    "Catalyst_Selenide":
        r"selenide",
    "Catalyst_MOF":
        r"\bmof\b|metal[- ]organic framework",
    "Catalyst_Carbon":
        r"carbon|graphene|cnt|carbon nanotube",
    "Catalyst_SingleAtom":
        r"single atom|single-atom",
    "Catalyst_Nanoparticle":
        r"nanoparticle|nano",
    "Catalyst_Metal":
        r"metallic"
}
for feature, pattern in catalyst_families.items():
    df[feature] = (
        catalyst_text
        .str.contains(
            pattern,
            regex=True,
            na=False
        )
        .astype(int)
    )
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
    "Alkaline_Electrolyte",
    # --------------------------------------------------------
    # CELL
    # --------------------------------------------------------
    "Divided_Cell",
    "Undivided_Cell",
    "Flow_Cell",
    "Three_Electrode",
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
    "Catalyst_MOF",
    "Catalyst_Carbon",
    "Catalyst_SingleAtom",
    "Catalyst_Nanoparticle",
    "Catalyst_Metal"
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
