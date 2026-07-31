from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class ReportAudit:
    score: int
    rating: str
    flags: tuple[str, ...]
    positive_signals: tuple[str, ...]
    suggested_footer: str


OVERCONFIDENCE_PATTERNS = {
    r"\bdefinitely\b": "Uses absolute language ('definitely') that may overstate model certainty.",
    r"\bproves?\b": "Claims the image proves a conclusion; imaging findings usually require qualified interpretation and clinical context.",
    r"\bconfirm(?:s|ed)?\b": "Uses confirmation language that may be too strong for an unvalidated AI output.",
    r"\bno doubt\b": "Uses absolute certainty language ('no doubt').",
    r"\brules? out\b": "Claims to rule out a condition without documenting validated scope and limitations.",
    r"\b100\s*%\b": "Reports 100% certainty, which is inappropriate for this prototype.",
}

REPLACEMENT_PATTERNS = {
    r"\bno (?:doctor|physician|radiologist) (?:is )?(?:needed|required|necessary)\b": (
        "Suggests professional review is unnecessary. RadiantGuard requires qualified physician or radiologist oversight."
    ),
    r"\breplaces? (?:a |the )?(?:doctor|physician|radiologist)\b": (
        "Claims AI can replace a qualified professional."
    ),
}


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def audit_generated_report(text: str) -> ReportAudit:
    normalized = " ".join(text.lower().split())
    flags: list[str] = []
    positives: list[str] = []

    for pattern, message in {**OVERCONFIDENCE_PATTERNS, **REPLACEMENT_PATTERNS}.items():
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            flags.append(message)

    uncertainty_terms = ("possible", "may", "could", "suggest", "uncertain", "confidence")
    if _contains_any(normalized, uncertainty_terms):
        positives.append("Communicates at least one uncertainty or possibility signal.")
    else:
        flags.append("Does not communicate uncertainty or the limits of the AI output.")

    review_terms = ("radiologist", "qualified physician", "doctor review", "clinical review")
    if _contains_any(normalized, review_terms):
        positives.append("References review by a qualified clinician or radiologist.")
    else:
        flags.append("Does not explicitly direct the result to qualified physician or radiologist review.")

    context_terms = ("clinical context", "symptoms", "history", "correlat", "laboratory")
    if _contains_any(normalized, context_terms):
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
        "Prototype research output only. This result is not a diagnosis and has not been clinically validated. "
        "Interpretation may be incomplete or incorrect and should be reviewed alongside the full study and "
        "clinical context by a qualified physician or radiologist."
    )

    return ReportAudit(
        score=score,
        rating=rating,
        flags=tuple(dict.fromkeys(flags)),
        positive_signals=tuple(dict.fromkeys(positives)),
        suggested_footer=suggested_footer,
    )
