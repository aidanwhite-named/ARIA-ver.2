from backend.models.schemas import ClaimElement, ParsedClaim
from backend.services.quantitative_assessment import assess_claim, assess_claims, format_assessment_markdown


def _claim():
    return ParsedClaim(
        claim_number=1, text="generic claim",
        elements=[
            ClaimElement(label="A", text="first limitation", importance="5"),
            ClaimElement(label="B", text="second limitation", importance="3"),
        ],
    )


def test_high_average_does_not_hide_critical_gap():
    cache = {"1": [
        {"label": "A", "judgment": "대응 없음", "quote": ""},
        {"label": "B", "judgment": "동일", "quote": "quoted", "chunk_id": "p1"},
    ]}
    result = assess_claim(_claim(), [cache], [0])
    assert result["status"] == "critical_gap"
    assert result["critical_uncovered_labels"] == ["A"]


def test_secondary_complement_is_reported_separately():
    primary = {"1": [
        {"label": "A", "judgment": "동일", "quote": "a", "chunk_id": "p1"},
        {"label": "B", "judgment": "대응 없음"},
    ]}
    secondary = {"1": [
        {"label": "A", "judgment": "일부 유사", "quote": "x"},
        {"label": "B", "judgment": "동일", "quote": "b", "chunk_id": "p2"},
    ]}
    result = assess_claim(_claim(), [primary, secondary], [0, 1])
    assert result["metrics"]["combined_coverage"] == 100.0
    assert result["metrics"]["primary_coverage"] < 100.0
    assert result["metrics"]["complement_dependency"] == 37.5


def test_missing_evidence_is_discounted_without_changing_coverage():
    unsupported = {"1": [{"label": "A", "judgment": "동일"}]}
    supported = {"1": [{"label": "A", "judgment": "동일", "quote": "text",
                        "chunk_id": "p1", "similarity_reason": "direct"}]}
    claim = ParsedClaim(
        claim_number=1, text="claim",
        elements=[ClaimElement(label="A", text="limitation", importance="3")],
    )
    low = assess_claim(claim, [unsupported], [0])
    high = assess_claim(claim, [supported], [0])
    assert low["metrics"]["combined_coverage"] == high["metrics"]["combined_coverage"]
    assert low["metrics"]["evidence_strength"] < high["metrics"]["evidence_strength"]


def test_markdown_uses_neutral_metric_labels():
    text = format_assessment_markdown(assess_claim(_claim(), [], []))
    assert "[정량평가 - 분석 보조지표]" in text
    assert "법적 결론을 포함하지 않습니다" in text
    assert "거절 성공" not in text


def test_dependent_assessment_separates_inherited_and_added_documents():
    claim = ParsedClaim(
        claim_number=2,
        claim_type="dependent",
        parent_claim=1,
        text="dependent claim",
        elements=[ClaimElement(label="A", text="added limitation", importance="5")],
    )
    inherited = {"2": [{"label": "A", "judgment": "대응 없음"}]}
    added = {"2": [{
        "label": "A", "judgment": "동일", "quote": "text",
        "chunk_id": "p1", "similarity_reason": "direct",
    }]}
    result = assess_claims(
        [claim],
        [inherited, added],
        {"2": {"total": [0, 1], "inherited": [0], "added": [1]}},
    )["2"]
    assert result["dependency"]["assessment_scope"] == "additional_limitations"
    assert result["dependency"]["coverage_gain"] == 100.0
    assert result["dependency"]["remaining_uncovered_labels"] == []
