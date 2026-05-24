"""Unit tests for the evaluation framework (no API calls required)."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure scripts/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.evaluate_agents import check_structural_requirements


def test_structural_check_passes_for_good_brief() -> None:
    brief = (
        "## Company Overview\n"
        "## Funding History\n"
        "## Competitors\n"
        "## Risks\n"
        "## Bull Case\n"
        "## Bear Case\n"
        "## Sources"
    )
    test_case: dict = {"expected_in_brief": [], "min_sections": 6}
    result = check_structural_requirements(brief, test_case)
    assert result["structural_pass"] is True
    assert result["sections_present"] >= 6


def test_structural_check_fails_for_incomplete_brief() -> None:
    brief = "Just some text without sections"
    test_case: dict = {"expected_in_brief": [], "min_sections": 6}
    result = check_structural_requirements(brief, test_case)
    assert result["structural_pass"] is False


def test_structural_check_phrase_coverage_full() -> None:
    brief = "Razorpay was founded by Harshil Mathur in Bangalore as a fintech payment gateway."
    test_case: dict = {
        "expected_in_brief": ["payment gateway", "Harshil Mathur", "fintech", "Bangalore"],
        "min_sections": 0,
    }
    result = check_structural_requirements(brief, test_case)
    assert result["phrases_coverage"] == 1.0
    assert result["expected_phrases_found"] == 4


def test_structural_check_phrase_coverage_partial() -> None:
    brief = "Razorpay is a fintech company."
    test_case: dict = {
        "expected_in_brief": ["payment gateway", "Harshil Mathur", "fintech", "Bangalore"],
        "min_sections": 0,
    }
    result = check_structural_requirements(brief, test_case)
    assert result["phrases_coverage"] == 0.25
    assert result["expected_phrases_found"] == 1


def test_structural_check_no_expected_phrases() -> None:
    brief = "Some brief content here."
    test_case: dict = {"expected_in_brief": [], "min_sections": 0}
    result = check_structural_requirements(brief, test_case)
    assert result["phrases_coverage"] == 1.0


def test_structural_check_counts_all_sections() -> None:
    brief = (
        "Company Overview section.\n"
        "Funding rounds.\n"
        "Competitor analysis.\n"
        "Bull case for investors.\n"
        "Bear case risks.\n"
        "Sources cited here.\n"
        "General risk factors.\n"
    )
    test_case: dict = {"expected_in_brief": [], "min_sections": 6}
    result = check_structural_requirements(brief, test_case)
    assert result["sections_present"] >= 6
    assert result["structural_pass"] is True


def test_eval_dataset_exists_and_valid() -> None:
    import json

    path = Path("data/eval/v1_companies.json")
    assert path.exists(), "Eval dataset file must exist"

    with open(path, encoding="utf-8") as f:
        dataset = json.load(f)

    assert len(dataset) >= 8, "Dataset should have at least 8 test cases"

    for case in dataset:
        assert "id" in case
        assert "query" in case
        assert "category" in case

    categories = {c["category"] for c in dataset}
    assert "well_known" in categories
    assert "harmful_refused" in categories


def test_eval_dataset_has_harmful_case() -> None:
    import json

    with open("data/eval/v1_companies.json", encoding="utf-8") as f:
        dataset = json.load(f)

    harmful = [c for c in dataset if c.get("should_be_refused")]
    assert len(harmful) >= 1, "Dataset must have at least one harmful query"
