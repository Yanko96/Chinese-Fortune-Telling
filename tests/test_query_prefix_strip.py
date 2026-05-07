"""Regression tests for _strip_query_prefix.

This is the fix for the Production Shadow Eval gap documented in
docs/BENCHMARK_REPORT.md Appendix A: the production BaZi/Forecast paths
prepend an English wrapper to the Chinese question, which pollutes HyDE.
The strip function reverses that for the retrieval stage only — the
generation prompt still sees the full wrapped query.
"""

from __future__ import annotations

import pytest

# conftest already patches sys.path so flat imports work
from fortune_langchain_utils import _strip_query_prefix  # noqa: E402


def test_strips_bazi_prefix():
    q = "BaZi analysis for someone born on 1990-07-22 14:15, gender: female. 什么是正财格？"
    assert _strip_query_prefix(q) == "什么是正财格？"


def test_strips_bazi_prefix_male():
    q = "BaZi analysis for someone born on 1985-03-12 08:30, gender: male. 七杀格的喜忌？"
    assert _strip_query_prefix(q) == "七杀格的喜忌？"


def test_strips_bazi_prefix_date_only():
    """Production sends 'YYYY-MM-DD HH:MM' but should also handle 'YYYY-MM-DD'."""
    q = "BaZi analysis for someone born on 1990-07-22, gender: female. 正官与七杀如何区分？"
    assert _strip_query_prefix(q) == "正官与七杀如何区分？"


def test_strips_forecast_prefix():
    q = "Yearly forecast for 2026 for Dragon (龙) with the question: 今年事业运如何？"
    assert _strip_query_prefix(q) == "今年事业运如何？"


def test_passes_through_pure_chinese_question():
    """If no wrapper is detected, return the query unchanged."""
    q = "正财格如何判断？"
    assert _strip_query_prefix(q) == q


def test_passes_through_pure_english_question():
    """English-only question (no wrapper match) should not be modified."""
    q = "What is the Day Master in BaZi?"
    assert _strip_query_prefix(q) == q


def test_does_not_strip_unrelated_english_prefix():
    """Only the specific BaZi/Forecast wrappers are stripped — arbitrary
    English shouldn't accidentally match."""
    q = "Random text about something else. 正财格如何判断？"
    assert _strip_query_prefix(q) == q, "non-matching prefix must be preserved"


def test_strip_is_case_insensitive():
    q = "BAZI analysis for someone born on 1990-07-22 14:15, GENDER: FEMALE. 什么是正财格？"
    assert _strip_query_prefix(q) == "什么是正财格？"


@pytest.mark.parametrize(
    "prefixed,expected",
    [
        (
            "BaZi analysis for someone born on 1980-12-08 06:10, gender: male. 印绶格如何取用神？",
            "印绶格如何取用神？",
        ),
        (
            "Yearly forecast for 2030 for Snake (蛇) with the question: 健康方面要注意什么？",
            "健康方面要注意什么？",
        ),
        ("纯中文问题", "纯中文问题"),
    ],
)
def test_strip_param(prefixed, expected):
    assert _strip_query_prefix(prefixed) == expected
