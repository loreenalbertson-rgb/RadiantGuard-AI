from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping, Sequence


@dataclass(frozen=True)
class ReportAudit:
    score: int
    rating: str
    flags: tuple[str, ...]
    positive_signals: tuple[str, ...]
    suggested_footer: str


OVERCONFIDENCE_PATTERNS: Mapping[str, str] = {
    r"\bdefinitely\b": (
        "Uses absolute language ('definitely') that may overstate model certainty."
    ),
    r"\bproves?\b": (
        "Claims the image proves a conclusion; imaging findings usually require "
        "qualified interpretation and clinical context."
    ),
    r"\bconfirm(?:s|ed|ing)?\b": (
        "Uses confirmation language that may be too strong for an unvalidated AI output."
    ),
    r"\bno doubt\b": "Uses absolute certainty language ('no doubt').",
    r"\brules? out\b": (
        "Claims to rule out a condition without documenting validated scope and limitations."
    ),
    r"\b100\s*%\b": (
        "Reports 100% certainty, which is inappropriate for this prototype."
    ),
}


# These patterns identify wording that discourages or dismisses qualified review.
REVIEW_DISCOURAGEMENT_PATTERNS: Mapping[str, str] = {
    r"\bno\s+(?:doctor|physician|radiologist)(?:'s)?\s+(?:review\s+)?"
    r"(?:is\s+)?(?:needed|required|necessary)\b": (
        "Suggests professional review is unnecessary. RadiantGuard requires "
        "qualified physician or radiologist oversight."
    ),
    r"\b(?:doctor|physician|radiologist)(?:'s)?\s+review\s+"
    r"(?:is\s+)?not\s+(?:needed|required|necessary)\b": (
        "Suggests professional review is unnecessary. RadiantGuard requires "
        "qualified physician or radiologist oversight."
    ),
    r"\b(?:does\s+not|doesn't|do\s+not|don't)\s+(?:need|require)\s+"
    r"(?:a\s+)?(?:doctor|physician|radiologist)(?:'s)?(?:\s+review)?\b": (
        "Discourages qualified professional review."
    ),
    r"\breplaces?\s+(?:a\s+|the\s+)?(?:doctor|physician|radiologist)\b": (
        "Claims AI can replace a qualified professional."
    ),
}


# These patterns require affirmative review language rather than simply detecting
# the word "radiologist". This prevents a phrase such as "no radiologist review
# is necessary" from being rewarded as a positive signal.
AFFIRMATIVE_REVIEW_PATTERNS: Sequence[str] = (
    r"\b(?:requires?|needs?|recommends?|advises?)\s+(?:a\s+)?(?:qualified\s+)?"
    r"(?:doctor|physician|radiologist)(?:'s)?\s+review\b",
    r"\bshould\s+be\s+reviewed\s+by\s+(?:a\s+)?(?:qualified\s+)?"
    r"(?:doctor|physician|radiologist)\b",
    r"\breview(?:ed)?\s+by\s+(?:a\s+)?(?:qualified\s+)?"
    r"(?:doctor|physician|radiologist)\b",
    r"\b(?:doctor|physician|radiologist)(?:'s)?\s+review\s+"
    r"(?:is\s+)?(?:required|recommended|necessary|needed)\b",
    r"\bconsult\s+(?:a\s+)?(?:qualified\s+)?"
    r"(?:doctor|physician|radiologist)\b",
)


UNCERTAINTY_PATTERNS: Sequence[str] = (
    r"\bpossible\b",
    r"\bpossibly\b",
    r"\bmay\b",
    r"\bmight\b",
    r"\bcould\b",
    r"\bsuggest(?:s|ed|ive)?\b",
    r"\buncertain(?:ty)?\b",
    r"\bconfidence\b",
    r"\bnot diagnostic\b",
    r"\bnot a diagnosis\b",
)


CONTEXT_PATTERNS: Sequence[str] = (
    r"\bclinical context\b",
    r"\bsymptoms?\b",
    r"\bhistory\b",
    r"\bcorrelat(?:e|es|ed|ion)\b",
    r"\blaborator(?:y|ies)\b",
)


def _matches_any(text: str, patterns: Sequence[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def audit_generated_report(text: str) -> ReportAudit:
    normalized = " ".join(text.lower().split())
    flags: list[str] = []
    positives: list[str] = []

    for pattern, message in OVERCONFIDENCE_PATTERNS.items():
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            flags.append(message)

    review_discouraged = False
    for pattern, message in REVIEW_DISCOURAGEMENT_PATTERNS.items():
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            flags.append(message)
            review_discouraged = True

    if _matches_any(normalized, UNCERTAINTY_PATTERNS):
        positives.append("Communicates at least one uncertainty or possibility signal.")
    else:
        flags.append("Does not communicate uncertainty or the limits of the AI output.")

    affirmative_review = _matches_any(normalized, AFFIRMATIVE_REVIEW_PATTERNS)
    if affirmative_review and not review_discouraged:
        positives.append("Directs the output to qualified physician or radiologist review.")
    elif not review_discouraged:
        flags.append(
            "Does not explicitly direct the result to qualified physician or radiologist review."
        )

    if _matches_any(normalized, CONTEXT_PATTERNS):
        positives.append("Acknowledges the importance of clinical context.")

    raw_score = 100 - (18 * len(flags)) + (4 * len(positives))
    score = max(0, min(100, raw_score))

    if score >= 85:
        rating = "Cautious wording"
    elif score >= 65:
        rating = "Review recommended"
    else:
        rating = "High-priority safety revision"

    suggested_footer = (
        "Prototype research output only. This result is not a diagnosis and has not "
        "been clinically validated. Interpretation may be incomplete or incorrect and "
        "should be reviewed alongside the full study and clinical context by a qualified "
        "physician or radiologist."
    )

    return ReportAudit(
        score=score,
        rating=rating,
        flags=tuple(dict.fromkeys(flags)),
        positive_signals=tuple(dict.fromkeys(positives)),
        suggested_footer=suggested_footer,
    )
