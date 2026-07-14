from backend.services.citation_chain import _dependent_added_inv


def _item(label, judgment, quote=""):
    return {
        "label": label,
        "judgment": judgment,
        "quote": quote,
        "chunk_id": "p1" if quote else "",
        "similarity_reason": "direct" if quote else "",
    }


def test_one_document_must_cover_all_remaining_additional_limitations_for_full_basis():
    none = "대응 없음"
    same = "동일"
    caches = {
        0: {"2": [_item("A", none), _item("B", none)]},
        1: {"2": [_item("A", same, "a"), _item("B", same, "b")]},
        2: {"2": [_item("A", same, "a"), _item("B", none)]},
    }
    added, trace = _dependent_added_inv(
        "2", [0], caches, 3,
        expected_labels={"A", "B"},
        expected_importance_by_label={"A": 5, "B": 3},
    )
    assert added == [1]
    assert trace["selection_basis"] == "single_document_full_gap_filler"
    assert trace["selected_document"] == 1


def test_partial_support_is_not_reported_as_full_gap_filling():
    none = "대응 없음"
    same = "동일"
    caches = {
        0: {"2": [_item("A", none), _item("B", none)]},
        1: {"2": [_item("A", same, "a"), _item("B", none)]},
    }
    added, trace = _dependent_added_inv(
        "2", [0], caches, 2,
        expected_labels={"A", "B"},
        expected_importance_by_label={"A": 5, "B": 3},
    )
    assert added == [1]
    assert trace["selection_basis"] == "partial_support_only"
    assert trace["candidate_scores"][0]["full_cover"] is False


def test_inherited_coverage_does_not_add_redundant_document():
    same = "동일"
    caches = {
        0: {"2": [_item("A", same, "a")]},
        1: {"2": [_item("A", same, "b")]},
    }
    added, trace = _dependent_added_inv(
        "2", [0], caches, 2,
        expected_labels={"A"},
        expected_importance_by_label={"A": 5},
    )
    assert added == []
    assert trace["selection_basis"] == "covered_by_inherited"


def test_difference_quote_is_not_promoted_to_dependent_reference():
    caches = {
        0: {"2": [_item("A", "대응 없음")]},
        1: {"2": [_item("A", "차이", "only related background")]},
    }

    added, trace = _dependent_added_inv(
        "2", [0], caches, 2,
        expected_labels={"A"},
        expected_importance_by_label={"A": 5},
    )

    assert added == []
    assert trace["selection_basis"] == "no_candidate"
