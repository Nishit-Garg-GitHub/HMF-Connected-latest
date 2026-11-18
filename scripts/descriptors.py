"""Pure text-to-descriptor parsers for the HMF literature table (stage 01).

Kept free of side effects so they can be unit-tested (see tests/). Each parser
takes one free-text cell and returns a number, a string or NaN. Behaviour is
identical to the code that used to live inside 01_preprocess_data.py.
"""
import re

import numpy as np
import pandas as pd


ANODE_SPLIT = re.compile(
    r"(?:\bcathode\b|\bcatholyte\b|counter[- ]electrode|\bcounter\b|"
    r"reference[- ]electrode|\breference\b)",
    flags=re.IGNORECASE,
)


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


def extract_temperature(value):
    if pd.isna(value):
        return np.nan
    text = str(value).lower()
    # Drop "± 2" style uncertainties so "23 ± 2 °C" parses as 23 °C.
    text = re.sub(r"(?:±|\+/-)\s*\d+(?:\.\d+)?", "", text)
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


def anode_text(value):
    """Return the anode/working-electrode part of a catalyst description."""
    if pd.isna(value):
        return ""
    text = str(value)
    # "Cathode: X ... Anode: Y" ordering is rare but handled.
    anode_match = re.search(r"(?:anode|working electrode)[^:]*:", text, re.I)
    if anode_match:
        text = text[anode_match.end():]
    return ANODE_SPLIT.split(text, maxsplit=1)[0]


def counter_electrode_text(value):
    if pd.isna(value):
        return ""
    match = re.search(r"(?:cathode|counter[- ]electrode|\bcounter\b)[^:]*:?(.*)",
                      str(value), flags=re.IGNORECASE)
    return match.group(1) if match else ""


def extract_electrolyte_ph(value):
    if pd.isna(value):
        return np.nan
    text = str(value)
    explicit = re.search(r"pH\s*(?:=|~|of|maintained at)?\s*(\d+(?:\.\d+)?)", text, re.I)
    if explicit:
        return float(explicit.group(1))
    hydroxide = re.search(
        r"(\d+(?:\.\d+)?)\s*(mM|M)\s*(?:KOH|NaOH|LiOH|CsOH|RbOH)", text, re.I
    )
    if hydroxide:
        conc = float(hydroxide.group(1))
        if hydroxide.group(2).lower() == "mm":
            conc /= 1000
        return float(np.clip(14 + np.log10(max(conc, 1e-4)), 7, 14.7))
    if re.search(r"H2SO4|HClO4|HNO3|\bHCl\b", text):
        return 1.0
    if re.search(r"HCO3", text):
        return 8.3
    if re.search(r"borate", text, re.I):
        return 9.2
    if re.search(r"SO4|ClO4|NO3|Cl\b|phosphate buffer", text, re.I):
        return 7.0
    return np.nan
