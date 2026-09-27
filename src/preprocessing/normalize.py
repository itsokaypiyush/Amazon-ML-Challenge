"""
Module 1: Preprocessing & Text Normalization
=============================================

This module handles all text-cleaning logic shared by every downstream stage
(blocking, feature engineering, cross-encoder input formatting). Keeping a
SINGLE normalization path is critical for entity resolution: if Source 1 and
Source 2 are normalized differently, string-distance features become noisy
and blocking recall silently degrades.

Design notes
------------
- Normalization is deliberately conservative: we standardize legal-entity
  suffixes and street abbreviations (which vary a lot across vendor feeds)
  but we do NOT stem or remove tokens that carry disambiguating signal
  (e.g. we keep "north"/"n" separate from being dropped entirely, we just
  standardize its form).
- All functions are pure (no I/O) and vectorizable via `.apply`, so they can
  be swapped for a faster batch implementation (e.g. via `pandas.Series.str`
  chains or `polars`) without touching the rest of the pipeline.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

import pandas as pd

try:
    import jellyfish  # phonetic blocking (Soundex / NYSIIS)
except ImportError:  # pragma: no cover
    jellyfish = None


# -----------------------------------------------------------------------------
# Static lookup tables
# -----------------------------------------------------------------------------

# Legal-entity suffixes -> canonical expanded form.
# Order matters: longer/more specific keys should be matched first, so we sort
# by descending token length at lookup time.
LEGAL_SUFFIX_MAP = {
    "inc": "incorporated",
    "incorp": "incorporated",
    "incorporated": "incorporated",
    "corp": "corporation",
    "corporation": "corporation",
    "co": "company",
    "company": "company",
    "ltd": "limited",
    "limited": "limited",
    "llc": "limited liability company",
    "l l c": "limited liability company",
    "llp": "limited liability partnership",
    "lp": "limited partnership",
    "plc": "public limited company",
    "pvt": "private",
    "pte": "private",
    "gmbh": "gesellschaft mit beschrankter haftung",
    "ag": "aktiengesellschaft",
    "sa": "sociedad anonima",
    "sarl": "societe a responsabilite limitee",
    "bv": "besloten vennootschap",
    "nv": "naamloze vennootschap",
    "kg": "kommanditgesellschaft",
    "and": "and",
    "&": "and",
}

# Street / address abbreviations -> canonical expanded form.
STREET_SUFFIX_MAP = {
    "st": "street",
    "str": "street",
    "rd": "road",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "ct": "court",
    "cir": "circle",
    "pl": "place",
    "sq": "square",
    "ter": "terrace",
    "pkwy": "parkway",
    "hwy": "highway",
    "hway": "highway",
    "fwy": "freeway",
    "expy": "expressway",
    "aly": "alley",
    "byp": "bypass",
    "cswy": "causeway",
    "trl": "trail",
    "trce": "trace",
    "xing": "crossing",
    "grn": "green",
    "hts": "heights",
    "jct": "junction",
    "lk": "lake",
    "mtn": "mountain",
    "vly": "valley",
    "byu": "bayou",
    "apt": "apartment",
    "ste": "suite",
    "fl": "floor",
    "bldg": "building",
    "rm": "room",
    "unit": "unit",
    "no": "number",
    "num": "number",
    "n": "north",
    "s": "south",
    "e": "east",
    "w": "west",
    "ne": "northeast",
    "nw": "northwest",
    "se": "southeast",
    "sw": "southwest",
    "po": "post office",
    "p o": "post office",
    "pobox": "post office box",
    "box": "box",
    "us": "united states",
    "usa": "united states",
}

# Common punctuation / noise characters we strip outright (not replaced).
_NOISE_CHARS_RE = re.compile(r"[^\w\s&]")
_MULTI_SPACE_RE = re.compile(r"\s+")
_LEADING_TRAILING_PUNCT_RE = re.compile(r"^\W+|\W+$")


def _strip_accents(text: str) -> str:
    """Normalize unicode (e.g. 'Café' -> 'Cafe') so accented vendor data
    matches cleanly against ASCII reference data."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def _apply_token_map(tokens: list[str], mapping: dict) -> list[str]:
    """Replace tokens using an abbreviation map. Applied token-by-token so we
    never accidentally rewrite substrings inside longer words (e.g. 'st'
    should not match inside 'street' or 'coast')."""
    return [mapping.get(tok, tok) for tok in tokens]


def _base_clean(text: Optional[str]) -> str:
    """Shared first pass: handle NaN/None, casing, accents, punctuation,
    and whitespace collapsing. Returns "" for null/empty input (never None),
    so downstream `.str` operations never blow up on NaN."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    text = str(text).strip()
    if not text:
        return ""

    text = _strip_accents(text)
    text = text.lower()

    # Normalize ampersand spacing before punctuation stripping removes it
    text = text.replace("&", " & ")

    # Replace punctuation (except & and word chars) with a space so that
    # "Smith,Jones" doesn't collapse into "smithjones".
    text = _NOISE_CHARS_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def normalize_company_name(name: Optional[str]) -> str:
    """
    Normalize a raw company name string into a canonical form.

    Steps:
      1. Lowercase, strip accents, strip punctuation noise.
      2. Expand legal-entity suffix abbreviations (Inc -> Incorporated, etc).
      3. Collapse whitespace.

    Examples
    --------
    >>> normalize_company_name("Acme Corp.")
    'acme corporation'
    >>> normalize_company_name("Smith & Jones, LLC")
    'smith and jones limited liability company'
    >>> normalize_company_name("O'Brien Pvt. Ltd.")
    'obrien private limited'
    """
    cleaned = _base_clean(name)
    if not cleaned:
        return ""

    tokens = cleaned.split(" ")
    tokens = _apply_token_map(tokens, LEGAL_SUFFIX_MAP)
    return " ".join(tokens).strip()


def normalize_address(address: Optional[str]) -> str:
    """
    Normalize a raw address string into a canonical form.

    Steps mirror `normalize_company_name` but use the street-suffix map and
    additionally normalize numeric tokens (e.g. "1st" -> "1", ordinal
    suffixes stripped) so that "123 1st St" and "123 First Street" line up
    better under fuzzy matching.

    Examples
    --------
    >>> normalize_address("123 Main St, Apt 4B")
    '123 main street apartment 4b'
    >>> normalize_address("45 N. Broadway Ave., Ste 200")
    '45 north broadway avenue suite 200'
    """
    cleaned = _base_clean(address)
    if not cleaned:
        return ""

    tokens = cleaned.split(" ")
    tokens = _apply_token_map(tokens, STREET_SUFFIX_MAP)

    # Strip ordinal suffixes glued to numbers: "1st" -> "1", "3rd" -> "3"
    tokens = [re.sub(r"^(\d+)(st|nd|rd|th)$", r"\1", tok) for tok in tokens]

    return " ".join(tokens).strip()


def normalize_zip(zip_code: Optional[str]) -> str:
    """Keep digits only, and use the first 5 for US-style ZIP+4 codes so
    '10001-1234' and '10001' can still block together."""
    cleaned = _base_clean(zip_code)
    digits = re.sub(r"\D", "", cleaned)
    return digits[:5] if len(digits) >= 5 else digits


def phonetic_code(name: str) -> str:
    """NYSIIS phonetic code of the first significant token of a cleaned
    company name. Used as one of several blocking keys — it groups
    "Kaufman" / "Coffman" style spelling variants that edit-distance alone
    can miss. Falls back to a truncated raw token if `jellyfish` is
    unavailable."""
    if not name:
        return ""
    tokens = [t for t in name.split(" ") if t not in ("the", "and")]
    if not tokens:
        return ""
    first_token = tokens[0]
    if jellyfish is not None:
        try:
            return jellyfish.nysiis(first_token)
        except Exception:
            return first_token[:4]
    return first_token[:4]


def build_blocking_key(
    name: str,
    city: Optional[str] = None,
    zip_code: Optional[str] = None,
    zip_prefix_len: int = 3,
) -> str:
    """
    Build a coarse exact-match blocking key from normalized fields.

    Combines the phonetic code of the company name with a geo key (zip
    prefix, falling back to city) so that records for genuinely different
    businesses in different regions never collide, while allowing modest
    spelling variation within the same name/region bucket.
    """
    phon = phonetic_code(name)
    geo = ""
    if zip_code:
        z = normalize_zip(zip_code)
        geo = z[:zip_prefix_len] if z else ""
    if not geo and city:
        geo = _base_clean(city)[:5]
    return f"{phon}|{geo}"


def preprocess_dataframe(
    df: pd.DataFrame,
    name_col: str = "company_name",
    address_col: str = "address",
    city_col: str = "city",
    zip_col: str = "zip",
    zip_prefix_len: int = 3,
) -> pd.DataFrame:
    """
    Apply full normalization to a raw dataframe and return an augmented copy
    with new columns: `clean_name`, `clean_address`, `clean_zip`,
    `phonetic_code`, `blocking_key`.

    Missing optional columns (city/zip) degrade gracefully — the blocking
    key just relies on whichever geo signal is available.
    """
    out = df.copy()

    for preferred, aliases in {
        name_col: ("business_name", "company_name"),
        address_col: ("business_address", "address"),
        city_col: ("city", "city_name"),
        zip_col: ("zip", "postal_code", "zipcode"),
    }.items():
        if preferred == name_col and preferred not in out.columns:
            resolved = next((a for a in aliases if a in out.columns), None)
            if resolved is not None and resolved != preferred:
                out = out.rename(columns={resolved: preferred})
        elif preferred == address_col and preferred not in out.columns:
            resolved = next((a for a in aliases if a in out.columns), None)
            if resolved is not None and resolved != preferred:
                out = out.rename(columns={resolved: preferred})
        elif preferred == city_col and preferred not in out.columns:
            resolved = next((a for a in aliases if a in out.columns), None)
            if resolved is not None and resolved != preferred:
                out = out.rename(columns={resolved: preferred})
        elif preferred == zip_col and preferred not in out.columns:
            resolved = next((a for a in aliases if a in out.columns), None)
            if resolved is not None and resolved != preferred:
                out = out.rename(columns={resolved: preferred})

    out["clean_name"] = out.get(name_col, "").apply(normalize_company_name)
    out["clean_address"] = out.get(address_col, "").apply(normalize_address) \
        if address_col in out.columns else ""

    if zip_col in out.columns:
        out["clean_zip"] = out[zip_col].apply(normalize_zip)
    else:
        out["clean_zip"] = ""

    city_series = out[city_col] if city_col in out.columns else pd.Series([""] * len(out))

    out["phonetic_code"] = out["clean_name"].apply(phonetic_code)
    out["blocking_key"] = [
        build_blocking_key(n, c, z, zip_prefix_len)
        for n, c, z in zip(out["clean_name"], city_series, out["clean_zip"])
    ]
    return out
