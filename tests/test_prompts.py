"""Prompt template rendering tests.

The prompts in api/fortune_prompts.py are ChatPromptTemplate / string
templates that the RAG chain expects. These tests just verify they
render without errors when given representative inputs — catches the
class of bugs where someone renames a placeholder but forgets to
update the template.
"""

from __future__ import annotations

import pytest

from fortune_prompts import (  # noqa: E402  (sys.path patched by conftest)
    fortune_contextualize_prompt,
    fortune_qa_prompt,
    birthday_analysis_prompt,
    yearly_forecast_prompt,
    bench_qa_prompt,
    bench_qa_prompt_concise,
    bench_qa_prompt_balanced,
    RAG_ANSWER_PROMPT_CONCISE,
    RAG_ANSWER_PROMPT_BALANCED,
)


SAMPLE_CONTEXT = "《三命通会》云：正财格者，月令藏正财而得令也。"
SAMPLE_HISTORY: list = []  # an empty history is a valid input


# ── Chat-style prompts ────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "prompt, input_text",
    [
        (fortune_qa_prompt, "What is my BaZi?"),
        (birthday_analysis_prompt, "1990-01-15 14:30"),
        (yearly_forecast_prompt, "Dragon (龙)"),
        (bench_qa_prompt, "正财格如何判断"),
        (bench_qa_prompt_concise, "正财格如何判断"),
        (bench_qa_prompt_balanced, "正财格如何判断"),
    ],
)
def test_chat_prompt_renders_without_error(prompt, input_text):
    messages = prompt.format_messages(
        context=SAMPLE_CONTEXT,
        chat_history=SAMPLE_HISTORY,
        input=input_text,
    )
    assert len(messages) >= 2, "chat prompt must produce at least system+user messages"
    full_text = "\n".join(m.content for m in messages if hasattr(m, "content"))
    assert SAMPLE_CONTEXT in full_text, "context must be present in rendered prompt"


def test_contextualize_prompt_uses_input_and_history():
    """The history-aware retriever's contextualize prompt should accept
    a chat_history list + input and pass through cleanly."""
    messages = fortune_contextualize_prompt.format_messages(
        chat_history=SAMPLE_HISTORY,
        input="follow up question",
    )
    rendered = "\n".join(m.content for m in messages if hasattr(m, "content"))
    assert "follow up question" in rendered


# ── String-template prompts (multihop benchmark variants) ────────────────────

@pytest.mark.parametrize(
    "template_str",
    [RAG_ANSWER_PROMPT_CONCISE, RAG_ANSWER_PROMPT_BALANCED],
)
def test_string_template_format(template_str):
    rendered = template_str.format(
        context=SAMPLE_CONTEXT,
        question="跨书：正官与七杀格局如何区分？",
    )
    assert SAMPLE_CONTEXT in rendered
    assert "正官与七杀" in rendered


def test_bench_qa_prompt_is_chinese_and_evidence_grounded():
    """The benchmark QA prompt must instruct evidence-based Chinese answers
    (this is the key methodological difference from the production prompt,
    which is English and role-played)."""
    messages = bench_qa_prompt.format_messages(
        context=SAMPLE_CONTEXT,
        input="正财格如何判断",
    )
    system_text = "\n".join(m.content for m in messages if m.type == "system")
    # Smoke check: these Chinese tokens must appear in the system instructions
    assert "原文" in system_text, "bench prompt must instruct evidence grounding (原文)"
    assert "中文" in system_text, "bench prompt must instruct Chinese output"
