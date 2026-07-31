from src.safety import audit_generated_report


def test_discouraged_radiologist_review_is_not_rewarded() -> None:
    result = audit_generated_report(
        "This image definitely confirms pneumonia, "
        "and no radiologist review is necessary."
    )

    assert result.score < 65

    assert any(
        "professional review is unnecessary" in flag.lower()
        or "discourages qualified professional review" in flag.lower()
        for flag in result.flags
    )

    assert not any(
        "radiologist review" in signal.lower()
        for signal in result.positive_signals
    )


def test_cautious_language_is_rewarded() -> None:
    result = audit_generated_report(
        "This prototype identified a possible area of concern, "
        "but the result is uncertain and must be reviewed by "
        "a qualified physician or radiologist."
    )

    assert result.score >= 85

    assert any(
        "uncertainty" in signal.lower()
        or "possibility" in signal.lower()
        for signal in result.positive_signals
    )

    assert any(
        "physician or radiologist review" in signal.lower()
        for signal in result.positive_signals
    )
