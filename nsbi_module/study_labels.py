"""Reusable study/dataset label formatting.

This module centralizes the mapping between raw dataset keys (e.g.
``reymermet2018``) and the display labels used in figures (e.g.
``Rey-Mermet 2018``). Other plotting scripts should import from here rather
than maintaining their own local mappings.
"""

import re

STUDY_NAME_MAP = {
    "eisenberg2019": "Eisenberg 2019",
    "hedge2018": "Hedge 2018",
    "reymermet2018": "Rey-Mermet 2018",
    "whitehead2019": "Whitehead 2019",
    "clayson2025": "Clayson 2025",
    "ulrich2015": "Ulrich 2015",
}


def format_author_year(text: str) -> str:
    """Capitalize and insert a space between author and year.

    Idempotent: safe to call on already-formatted names.

    Examples
    --------
    >>> format_author_year("eisenberg2019")
    'Eisenberg 2019'
    >>> format_author_year("Eisenberg 2019")
    'Eisenberg 2019'
    """
    text = text.capitalize()
    text = re.sub(r'(\D)(\d)', r'\1 \2', text)
    # Collapse any double spaces introduced by re-running on already-formatted names
    text = text.replace("  ", " ")
    return text


def format_study_label(study: str) -> str:
    """Return a display label for a dataset key.

    Falls back to :func:`format_author_year` if the key is not in the explicit
    map.

    Examples
    --------
    >>> format_study_label("reymermet2018")
    'Rey-Mermet 2018'
    >>> format_study_label("someNewStudy2026")
    'SomeNewStudy 2026'
    """
    return STUDY_NAME_MAP.get(study, format_author_year(study))


# ---------------------------------------------------------------------------
# Abbreviated labels (e.g. "Eisenberg 2019" → "E19")
# Based on 32fig2_v8_combined.py AUTHOR_ABBREV
# ---------------------------------------------------------------------------
STUDY_ABBREV_MAP = {
    "Clayson 2024": "C24",
    "Clayson 2025": "C25",
    "Eisenberg 2019": "E19",
    "Hedge 2018": "H18",
    "Kucina 2023": "K23",
    "Lee 2025": "L25",
    "Rey-Mermet 2018": "R18",
    "Reymermet 2018": "R18",  # before hyphen replacement
    "Ulrich 2015": "U15",
    "Whitehead 2019": "W19",
}


def format_study_abbrev(study: str) -> str:
    """Return an abbreviated label for a dataset key.

    Handles both raw keys and already-formatted display names.

    Examples
    --------
    >>> format_study_abbrev("eisenberg2019")
    'E19'
    >>> format_study_abbrev("Eisenberg 2019")
    'E19'
    >>> format_study_abbrev("reymermet2018")
    'R18'
    """
    # Try direct lookup first (handles already-formatted display names)
    if study in STUDY_ABBREV_MAP:
        return STUDY_ABBREV_MAP[study]
    display = format_study_label(study)
    return STUDY_ABBREV_MAP.get(display, display)
