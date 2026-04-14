"""
Core RLM engine.

Uses the RLM library (https://github.com/alexzhang13/rlm) by Alex L. Zhang et al.
Paper: "Recursive Language Models" (arXiv:2512.24601)

Architecture:
- Root LM: GPT-5.4 — orchestrates the analysis, writes code to navigate filings
- Sub LM: GPT-5.4-nano — handles recursive sub-calls over chunks (fast + cheap)

The RLM scaffold does the heavy lifting: the root model writes Python code
to navigate 200+ page SEC filings, and sub-models analyze extracted sections.

Why RLM can be slow:
- Each completion runs up to max_iterations sequential root-LM rounds (default 15).
- Each round sends the full conversation history (system + all prior turns + code results).
- Root model uses reasoning_effort (GPT-5 "thinking") which adds latency.
- Every llm_query() in REPL code is a sub-call (gpt-5.4-nano); many sub-calls per round.
- Docker REPL adds per-code-execution overhead; use RLM_ENVIRONMENT=local for dev.

How to speed up:
- Use Speed/Fast presets: max_depth=1, max_iterations=5–10, reasoning_effort=low.
- Lower default max_iterations (e.g. 8–10) if you prefer faster, less thorough runs.
- Prefer local REPL (RLM_ENVIRONMENT=local) when safe; Docker is for production isolation.
"""

import asyncio
import concurrent.futures
import io
import os
import re
import sys
import logging
from dataclasses import dataclass

from rlm import RLM
from rlm.logger import RLMLogger

from .prompts import FINANCE_RLM_SYSTEM_PROMPT, GOVERNMENT_RLM_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# ── Monkey-patch RLM's OpenAI client to forward kwargs (max_tokens, reasoning, etc.) ──
# The upstream rlm library hardcodes only model+messages in chat.completions.create(),
# silently dropping max_tokens, reasoning_effort, etc.  This patch reads self.kwargs
# (populated from backend_kwargs) and passes them through to the API call.
from rlm.clients.openai import OpenAIClient as _OrigOpenAIClient, DEFAULT_PRIME_INTELLECT_BASE_URL

def _prepare_api_call(self, prompt, model=None):
    """Shared logic for sync/async completion: validate prompt, resolve model, build kwargs."""
    if isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    elif isinstance(prompt, list) and all(isinstance(item, dict) for item in prompt):
        messages = prompt
    else:
        raise ValueError(f"Invalid prompt type: {type(prompt)}")
    model = model or self.model_name
    if not model:
        raise ValueError("Model name is required for OpenAI client.")
    extra_body = {}
    if self.client.base_url == DEFAULT_PRIME_INTELLECT_BASE_URL:
        extra_body["usage"] = {"include": True}
    api_kwargs = {}
    for k, v in getattr(self, "kwargs", {}).items():
        if k in ("max_tokens", "max_completion_tokens", "temperature", "top_p", "reasoning_effort"):
            api_kwargs[k] = v
        elif k == "reasoning" and isinstance(v, dict):
            api_kwargs["reasoning"] = v
    return model, messages, extra_body, api_kwargs

def _patched_completion(self, prompt, model=None):
    model, messages, extra_body, api_kwargs = _prepare_api_call(self, prompt, model)
    response = self.client.chat.completions.create(
        model=model, messages=messages, extra_body=extra_body, **api_kwargs
    )
    self._track_cost(response, model)
    return response.choices[0].message.content

async def _patched_acompletion(self, prompt, model=None):
    model, messages, extra_body, api_kwargs = _prepare_api_call(self, prompt, model)
    response = await self.async_client.chat.completions.create(
        model=model, messages=messages, extra_body=extra_body, **api_kwargs
    )
    self._track_cost(response, model)
    return response.choices[0].message.content

_OrigOpenAIClient.completion = _patched_completion
_OrigOpenAIClient.acompletion = _patched_acompletion
logger.info("Patched RLM OpenAIClient to forward max_tokens/reasoning_effort to API calls")

# Bounded thread pool for RLM completion — limited workers to prevent memory/CPU spikes
# RLM runs are heavy; each can use 1-2GB RAM. Adjust via RLM_MAX_WORKERS env var:
# - 8GB RAM: RLM_MAX_WORKERS=2
# - 16GB RAM: RLM_MAX_WORKERS=3-4 (default 3)
# - 32GB+ RAM: RLM_MAX_WORKERS=5-8
# Monitor with `htop` or Activity Monitor and reduce if you see swapping/OOM
_rlm_executor: concurrent.futures.ThreadPoolExecutor | None = None

def _get_executor() -> concurrent.futures.ThreadPoolExecutor:
    global _rlm_executor
    if _rlm_executor is None:
        _rlm_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=int(os.getenv("RLM_MAX_WORKERS", "3")),
            thread_name_prefix="rlm",
        )
    return _rlm_executor


@dataclass
class AnalysisResult:
    answer: str
    total_tokens_in: int
    total_tokens_out: int
    estimated_cost_usd: float
    num_sub_calls: int
    trajectory_log: str | None = None


# Pricing per 1M tokens
# Source: https://platform.openai.com/docs/pricing
MODEL_PRICING = {
    # OpenAI GPT-5.4 family
    "gpt-5.4": {"input": 2.50, "output": 15.00},
    "gpt-5.4-mini": {"input": 0.75, "output": 4.50},
    "gpt-5.4-nano": {"input": 0.20, "output": 1.25},
}
DEFAULT_PRICING = {"input": 2.50, "output": 15.00}  # fallback to GPT-5.4 pricing


def _clean_rlm_response(response: str) -> str:
    """
    Clean RLM response: extract FINAL() content, strip leaked Python code,
    and remove debug output that sometimes appears in the final answer.
    """
    import re
    if not response or not response.strip():
        return response

    original_response = response
    text = response.strip()

    # 1. Extract content from FINAL(...) if present (handle nested parens in markdown)
    idx = text.find("FINAL(")
    if idx != -1:
        start = idx + 6  # character after "FINAL("
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            # Skip parens inside markdown code backticks so "(e.g. `$1.2B`)"
            # doesn't break balance
            if i < len(text) - 1 and text[i : i + 2] == "`":
                j = text.find("`", i + 1)
                if j != -1:
                    i = j + 1
                    continue
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                depth -= 1
            i += 1
        if depth == 0:
            return text[start : i - 1].strip()

    # 2. Remove Python code blocks (```python ... ```)
    text = re.sub(r"```python\s*.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"```\s*.*?```", "", text, flags=re.DOTALL)

    # 3. Remove lines that look like Python code (print, for, if, loops, etc.)
    # but keep Markdown and normal prose (e.g. "For the fiscal year..." or "If the company...")
    lines = text.split("\n")
    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()
        # Skip obvious Python code / debug output (be precise to avoid stripping valid prose)
        if stripped.startswith("print("):
            continue
        if re.match(r"^for\s+\w+\s+in\s+", stripped):  # "for x in items:" not "For the year..."
            continue
        if re.match(r"^if\s+.+:\s*$", stripped) or stripped.startswith("elif "):  # "if x:" not "If applicable..."
            continue
        if stripped.startswith("else:"):
            continue
        if any(stripped.startswith(p) for p in ("googl_", "revenue_", "stmt_", "googl_filings")):
            continue
        if re.match(r"^\w+\s*=\s*\[", stripped) or re.match(r"^\w+\s*=\s*re\.", stripped):
            continue
        if re.match(r"^\w+\.group\(", stripped) or "llm_query(" in stripped or stripped.startswith("query = "):
            continue
        # Skip lines that are mostly = or - (decorative from print output)
        if re.match(r"^[=\-]{20,}$", stripped) and len(cleaned) > 0:
            prev = cleaned[-1].strip() if cleaned else ""
            if not prev.startswith("#") and not prev.startswith("##"):
                continue
        cleaned.append(line)

    text = "\n".join(cleaned)

    # 4. Remove "Example Queries" / UI bleed (can happen if model output was truncated)
    text = re.sub(r"\n+\s*Example Queries\s*\n.*", "", text, flags=re.DOTALL)
    text = re.sub(r"\n+\s*AAPL\s*·\s*10-K\s*\n.*", "", text, flags=re.DOTALL)

    # 5. Collapse excess blank lines
    text = re.sub(r"\n{4,}", "\n\n\n", text)

    # If cleaning removed everything, return original
    cleaned = text.strip()
    if not cleaned and original_response.strip():
        logger.warning(f"[RLM] Cleaning removed all content! Returning original response")
        return original_response.strip()

    return cleaned or response


def estimate_cost(tokens_in: int, tokens_out: int, model: str = "") -> float:
    pricing = MODEL_PRICING.get(model, DEFAULT_PRICING)
    return (tokens_in / 1_000_000) * pricing["input"] + \
           (tokens_out / 1_000_000) * pricing["output"]


def create_rlm(
    log_dir: str = "./logs",
    environment: str = "docker",  # Use docker for safety in production
    root_model: str = "gpt-5.4",
    sub_model: str | None = "gpt-5.4-nano",
    verbose: bool = True,
    use_local: bool = False,  # Use local vLLM server instead of OpenAI
    max_iterations: int = 15,
    max_depth: int = 2,
    reasoning_effort: str = "low",  # "low", "medium", "high" — controls GPT-5 thinking time
    custom_system_prompt: str | None = None,  # Per-request system prompt (e.g. finance vs government)
) -> RLM:
    """
    Create an RLM instance configured for SEC filing analysis.

    Args:
        log_dir: Where to save trajectory logs (useful for debugging)
        environment: REPL sandbox type. "local" for dev, "docker" for prod.
        root_model: Model for the root LM (orchestrator)
        sub_model: Model for recursive sub-calls. Defaults to gpt-5.4-nano.
        verbose: Print trajectory to console
        use_local: Use local vLLM server (Qwen 3-8B) instead of OpenAI API
        max_iterations: Max REPL iterations before forcing a final answer (5=fast, 30=thorough)
        max_depth: Max recursion depth for sub-calls (1=fast/flat, 5=deep/thorough)
        reasoning_effort: GPT-5 reasoning depth — "low" for speed, "medium" for quality
        custom_system_prompt: Override the default RLM system prompt (required for finance/government analysis).
    """
    rlm_logger = RLMLogger(log_dir=log_dir)

    if use_local:
        # Configure for local Ollama server
        local_base_url = os.getenv("LOCAL_MODEL_BASE_URL", "http://localhost:11434/v1")
        local_root_model = os.getenv("LOCAL_ROOT_MODEL_NAME", "qwen3:8b")
        local_sub_model = os.getenv("LOCAL_SUB_MODEL_NAME", "qwen3:4b")

        safe_url = re.sub(r"://[^@]+@", "://<redacted>@", local_base_url)
        logger.info(f"Using local Ollama models: root={local_root_model}, sub={local_sub_model} at {safe_url}")

        kwargs = dict(
            backend="openai",
            backend_kwargs={
                "model_name": local_root_model,
                "max_completion_tokens": 16384,
                "base_url": local_base_url,
                "api_key": "EMPTY",  # Ollama doesn't require an API key
            },
            environment=environment,
            max_iterations=max_iterations,
            max_depth=max_depth,
            verbose=verbose,
            logger=rlm_logger,
            custom_system_prompt=custom_system_prompt,
        )

        # Use smaller local model for sub-calls (faster + cheaper)
        if sub_model:
            kwargs["other_backends"] = ["openai"]
            kwargs["other_backend_kwargs"] = [{
                "model_name": local_sub_model,
                "max_completion_tokens": 8192,
                "base_url": local_base_url,
                "api_key": "EMPTY",
            }]
    else:
        # Configure for OpenAI API
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable not set")

        kwargs = dict(
            backend="openai",
            backend_kwargs={
                "model_name": root_model,
                "max_completion_tokens": 16384,  # Allow longer responses (default is ~8192, which truncates comparative analyses)
                "reasoning_effort": reasoning_effort,
            },
            environment=environment,
            max_iterations=max_iterations,
            max_depth=max_depth,
            verbose=verbose,
            logger=rlm_logger,
            custom_system_prompt=custom_system_prompt,
        )

        # Sub-model for recursive calls via other_backends.
        if sub_model:
            kwargs["other_backends"] = ["openai"]
            kwargs["other_backend_kwargs"] = [{
                "model_name": sub_model,
                "max_completion_tokens": 8192,  # Sub-calls typically shorter than root responses
            }]

    rlm = RLM(**kwargs)

    return rlm


async def analyze_filing(
    query: str,
    filing_text: str,
    filing_info: str = "",
    max_depth: int | None = None,
    max_iterations: int | None = None,
    use_local: bool = False,
    reasoning_effort: str | None = None,
    mode: str = "finance",
) -> AnalysisResult:
    """
    Run an RLM query against a filing.

    Creates a fresh RLM instance per request so each analysis runs with its
    own config (no shared-state locks).

    Args:
        query: User's question
        filing_text: Full text of the SEC filing(s)
        filing_info: Metadata string (company name, form type, date) for context
        max_depth: Override recursion depth (1=fast, 5=thorough). None uses RLM default.
        max_iterations: Override iteration cap (5=fast, 30=thorough). None uses RLM default.
        reasoning_effort: GPT-5 reasoning depth, "low" for speed, "medium" for quality.
    """
    # Build the system prompt with filing structure context.
    from datetime import date
    current_date = date.today().strftime("%B %d, %Y")  # e.g., "February 09, 2026"

    if mode == "government":
        system_context = GOVERNMENT_RLM_SYSTEM_PROMPT.format(
            context_len=len(filing_text),
            current_date=current_date,
        )
        full_query = f"""Document info: {filing_info}

Question: {query}

**CRITICAL: The document text is ALREADY LOADED in the `context` variable. It contains {len(filing_text):,} characters. DO NOT claim the document doesn't exist — you have it right now.**

Your task:
1. Use Python code to navigate the `context` variable efficiently — e.g. `print(context[50000:80000])`, `print(re.findall(...))`, slice by section boundaries
2. Use `llm_query()` for semantic analysis of specific sections
3. Be precise with data points — extract exact values from the text

Format your final answer as rich Markdown with headers, lists, tables where appropriate, and inline citations (e.g. [Section 3.2 — Performance Requirements]).
End with a Sources section listing the document sections you consulted.

When done, output ONLY: FINAL(your complete markdown answer here) — no code, no print statements."""
    else:
        system_context = FINANCE_RLM_SYSTEM_PROMPT.format(
            context_len=len(filing_text),
            current_date=current_date,
        )
        full_query = f"""Filing info: {filing_info}

Question: {query}

**CRITICAL: The SEC filing text is ALREADY LOADED in the `context` variable. It contains {len(filing_text):,} characters of filing data. DO NOT claim the filing doesn't exist — you have it right now.**

Your task:
1. Use Python code to navigate the `context` variable efficiently — e.g. `print(context[50000:80000])`, `print(re.findall(...))`, slice by section boundaries
2. Use `llm_query()` for semantic analysis of specific sections
3. For numbers from financial statements (income statement, balance sheet, etc.): locate the exact row (line item) and column (period) in the raw text, then report ONLY the value that appears in that cell. The digit string you report must appear literally in the document — never substitute a value from a different row, column, or table.

Format your final answer as rich Markdown with headers, lists, tables where appropriate, and inline citations for every data point (e.g. [Item 7 — MD&A]).
End with a Sources section listing the filing sections you consulted.

When done, output ONLY: FINAL(your complete markdown answer here) — no code, no print statements."""

    # Always create a per-request RLM with our system prompt. The upstream library uses
    # self.system_prompt (set via custom_system_prompt in __init__), so we must pass it
    # here; mutating .custom_system_prompt after creation has no effect.
    env = os.getenv("RLM_ENVIRONMENT", "local")
    has_overrides = (max_depth is not None) or (max_iterations is not None) or (reasoning_effort is not None)

    def _run_rlm():
        request_rlm = create_rlm(
            environment=env,
            verbose=False,
            use_local=use_local,
            max_depth=max_depth if max_depth is not None else 2,
            max_iterations=max_iterations if max_iterations is not None else 15,
            reasoning_effort=reasoning_effort or "low",
            custom_system_prompt=system_context,
        )

        # Redirect sys.stdout/stderr to safe in-memory buffers during RLM execution.
        # LocalREPL._capture_output() globally replaces and restores sys.stdout/stderr;
        # in a threaded web server the process stdout pipe can raise [Errno 5] (EIO).
        # By setting safe buffers first, _capture_output saves/restores these instead.
        _orig_out, _orig_err = sys.stdout, sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        try:
            return request_rlm.completion(filing_text, full_query)
        finally:
            sys.stdout, sys.stderr = _orig_out, _orig_err

    # Run in bounded thread pool to avoid blocking event loop and prevent machine overload
    import time as _time
    t_start = _time.time()
    depth_label = f"max_depth={max_depth}, max_iter={max_iterations}, reasoning={reasoning_effort}" if has_overrides else "default"
    logger.info(f"[RLM] starting completion — context={len(filing_text):,} chars, query={query[:80]!r}, {depth_label}")

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        _get_executor(),
        _run_rlm,
    )

    t_elapsed = _time.time() - t_start
    logger.info(f"[RLM] completion finished in {t_elapsed:.1f}s")

    # Extract usage stats from usage_summary (per-model for accurate cost)
    tokens_in = 0
    tokens_out = 0
    num_sub_calls = 0
    total_cost = 0.0
    if result.usage_summary and result.usage_summary.model_usage_summaries:
        for model, usage in result.usage_summary.model_usage_summaries.items():
            tokens_in += usage.total_input_tokens
            tokens_out += usage.total_output_tokens
            num_sub_calls += usage.total_calls
            total_cost += estimate_cost(usage.total_input_tokens, usage.total_output_tokens, model)
            logger.info(f"[RLM] model={model} calls={usage.total_calls} tokens_in={usage.total_input_tokens:,} tokens_out={usage.total_output_tokens:,}")

    raw_response = result.response
    logger.debug(f"[RLM] raw response length={len(raw_response) if raw_response else 0} chars, first 500: {(raw_response or '')[:500]!r}")

    # If response is very short or empty, return it as-is without cleaning
    if not raw_response or len(raw_response.strip()) < 50:
        logger.warning(f"[RLM] Response is too short ({len(raw_response) if raw_response else 0} chars), returning as-is")
        answer = raw_response or "No response generated."
    else:
        answer = _clean_rlm_response(raw_response)
        logger.debug(f"[RLM] cleaned response length={len(answer)} chars, first 200: {answer[:200]!r}")
    return AnalysisResult(
        answer=answer,
        total_tokens_in=tokens_in,
        total_tokens_out=tokens_out,
        estimated_cost_usd=total_cost,
        num_sub_calls=max(0, num_sub_calls - 1),  # exclude root call
    )
