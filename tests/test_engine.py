"""Tests for secdd.engine — pure functions only (no RLM/API calls)."""

import pytest
from secdd.engine import estimate_cost, _clean_rlm_response


# ── estimate_cost ──


class TestEstimateCost:
    def test_gpt54_pricing(self):
        # 1M input tokens at $2.50 + 1M output at $15.00 = $17.50
        assert estimate_cost(1_000_000, 1_000_000, "gpt-5.4") == pytest.approx(17.50)

    def test_gpt54_nano_pricing(self):
        # 1M input at $0.20 + 1M output at $1.25 = $1.45
        assert estimate_cost(1_000_000, 1_000_000, "gpt-5.4-nano") == pytest.approx(1.45)

    def test_gpt54_mini_pricing(self):
        # 1M input at $0.75 + 1M output at $4.50 = $5.25
        assert estimate_cost(1_000_000, 1_000_000, "gpt-5.4-mini") == pytest.approx(5.25)

    def test_unknown_model_falls_back_to_gpt54(self):
        assert estimate_cost(1_000_000, 1_000_000, "unknown-model") == pytest.approx(17.50)

    def test_zero_tokens(self):
        assert estimate_cost(0, 0, "gpt-5.4") == 0.0

    def test_typical_usage(self):
        # 50k input, 2k output on gpt-5.4
        cost = estimate_cost(50_000, 2_000, "gpt-5.4")
        expected = (50_000 / 1e6) * 2.50 + (2_000 / 1e6) * 15.00
        assert cost == pytest.approx(expected)

    def test_empty_model_string_uses_fallback(self):
        assert estimate_cost(1_000_000, 1_000_000, "") == pytest.approx(17.50)


# ── _clean_rlm_response ──


class TestCleanRlmResponse:
    def test_extracts_final_content(self):
        response = 'Some code here\nFINAL(## Revenue\nTotal was $10B)'
        result = _clean_rlm_response(response)
        assert result == "## Revenue\nTotal was $10B"

    def test_final_with_nested_parens(self):
        response = 'FINAL(Revenue was $10B (up 15% YoY) from operations)'
        result = _clean_rlm_response(response)
        assert result == "Revenue was $10B (up 15% YoY) from operations"

    def test_final_with_backtick_dollar(self):
        """Backtick content like `$1.2B` shouldn't break paren matching."""
        response = 'FINAL(Revenue was `$1.2B` (up 10%))'
        result = _clean_rlm_response(response)
        assert "`$1.2B`" in result
        assert "(up 10%)" in result

    def test_removes_python_code_blocks(self):
        response = "Here is the answer\n```python\nprint('hello')\n```\nEnd."
        result = _clean_rlm_response(response)
        assert "print('hello')" not in result
        assert "Here is the answer" in result
        assert "End." in result

    def test_removes_print_statements(self):
        response = "print(context[:500])\n## Analysis\nRevenue grew 10%"
        result = _clean_rlm_response(response)
        assert "print(" not in result
        assert "## Analysis" in result

    def test_removes_for_loops_not_prose(self):
        response = "for item in items:\n  print(item)\nFor the fiscal year, revenue was $5B."
        result = _clean_rlm_response(response)
        assert "for item in items:" not in result
        assert "For the fiscal year" in result

    def test_removes_if_statements_not_prose(self):
        response = 'if x > 5:\n  pass\nIf applicable, the company must disclose.'
        result = _clean_rlm_response(response)
        assert "if x > 5:" not in result
        assert "If applicable" in result

    def test_removes_variable_assignments(self):
        response = "revenue_data = [1, 2, 3]\n## Findings\nRevenue grew."
        result = _clean_rlm_response(response)
        assert "revenue_data = [" not in result
        assert "## Findings" in result

    def test_removes_example_queries_bleed(self):
        response = "The answer is 42.\n\nExample Queries\nsome UI stuff\nmore stuff"
        result = _clean_rlm_response(response)
        assert "Example Queries" not in result
        assert "The answer is 42." in result

    def test_collapses_excess_blank_lines(self):
        response = "Line 1\n\n\n\n\n\nLine 2"
        result = _clean_rlm_response(response)
        assert "\n\n\n\n" not in result
        assert "Line 1" in result
        assert "Line 2" in result

    def test_empty_input_returns_as_is(self):
        assert _clean_rlm_response("") == ""
        assert _clean_rlm_response("   ") == "   "

    def test_none_input_returns_as_is(self):
        assert _clean_rlm_response(None) is None

    def test_all_code_stripped_returns_original(self):
        """If cleaning removes everything, fall back to original."""
        response = "print(x)\nprint(y)\nprint(z)"
        result = _clean_rlm_response(response)
        # All lines are print() so they all get stripped; fallback returns original
        assert result == response.strip()

    def test_removes_llm_query_lines(self):
        response = 'query = "what is revenue"\nllm_query(query, context)\n## Answer\nRevenue was $5B'
        result = _clean_rlm_response(response)
        assert "llm_query(" not in result
        assert "query = " not in result
        assert "## Answer" in result

    def test_removes_else_lines(self):
        response = "else:\n  something\n## Real content here"
        result = _clean_rlm_response(response)
        assert "else:" not in result
        assert "## Real content here" in result

    def test_removes_elif_lines(self):
        response = "elif x > 10:\n  handle()\n## Summary\nDone."
        result = _clean_rlm_response(response)
        assert "elif " not in result
        assert "## Summary" in result

    def test_keeps_separator_after_heading(self):
        """Decorative === lines under markdown headings should be kept."""
        response = "## Section\n" + "=" * 40 + "\nContent here"
        result = _clean_rlm_response(response)
        assert "=" * 40 in result

    def test_strips_separator_not_after_heading(self):
        """Decorative === lines not under a heading should be stripped."""
        response = "Some text\n" + "=" * 40 + "\nMore text"
        result = _clean_rlm_response(response)
        assert "=" * 40 not in result
        assert "Some text" in result
        assert "More text" in result

    def test_real_world_rlm_output(self):
        """Simulate a realistic RLM output with mixed code and answer."""
        response = """```python
import re
matches = re.findall(r'Item 1A', context)
print(len(matches))
```

print(context[50000:55000])

googl_filings = context[:10000]

FINAL(## Risk Factors Summary

Apple identified **32 risk factors** in their latest 10-K filing [Item 1A].

Key categories:
- **Macroeconomic**: Global economic uncertainty (up from 2023)
- **Competition**: Intense competitive pressure in all segments
- **Supply Chain**: Concentration risk in semiconductor supply

> "The Company's business can be impacted by political events, trade policies..." [Item 1A, p. 15]

### Sources
- Item 1A: Risk Factors (chars 50,000-80,000))"""
        result = _clean_rlm_response(response)
        assert "## Risk Factors Summary" in result
        assert "**32 risk factors**" in result
        assert "import re" not in result
        assert "print(context" not in result
        assert "googl_filings" not in result
