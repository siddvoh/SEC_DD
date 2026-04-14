#!/usr/bin/env python3
"""
secdd: Analyze SEC filings and documents using Recursive Language Models.

Built on RLM (https://github.com/alexzhang13/rlm) by Alex L. Zhang et al.

Usage:
  secdd "What are Apple's main risk factors in their latest 10-K?"
  secdd filing.pdf "What are the main risk factors?"
  secdd 10k_2023.htm 10k_2024.htm "Compare revenue recognition policies"
  secdd --ticker AAPL --form 10-K "Summarize the risk factors"

Supported file types: .txt, .htm, .html, .pdf
"""

import argparse
import asyncio
import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()

MODEL = "gpt-5.4"

MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB


class SecddError(Exception):
    """User-facing error in CLI operations."""
    pass


def read_file_content(file_path: str) -> str:
    """Read file as text. Supports .txt, .htm/.html (with encoding fallbacks), and .pdf (via pypdf)."""
    file_size = os.path.getsize(file_path)
    if file_size > MAX_FILE_SIZE_BYTES:
        raise SecddError(f"File too large: {file_path} is {file_size // (1024*1024)} MB (max 50 MB)")

    path_lower = file_path.lower()
    if path_lower.endswith(".pdf"):
        try:
            from pypdf import PdfReader
        except ImportError:
            raise SecddError("PDF support requires pypdf. Run: pip install pypdf")
        reader = PdfReader(file_path)
        parts = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                parts.append(t)
        return "\n\n".join(parts).strip() or "(No text extracted from PDF)"
    if path_lower.endswith((".htm", ".html")):
        for encoding in ("utf-8", "cp1252", "latin-1"):
            try:
                with open(file_path, "r", encoding=encoding, errors="strict") as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _run_simple(query: str, file_paths: list[str], reasoning_effort: str):
    """Simple mode: send query + files directly to OpenAI (no RLM)."""
    import openai
    from .engine import estimate_cost

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SecddError("OPENAI_API_KEY environment variable not set")

    if file_paths:
        parts = []
        total_chars = 0
        for file_path in file_paths:
            if not os.path.isfile(file_path):
                raise SecddError(f"File not found: {file_path}")
            content = read_file_content(file_path)
            total_chars += len(content)
            name = os.path.basename(file_path)
            parts.append(f"--- Document: {name} ---\n\n{content}")
        combined = "\n\n".join(parts)
        user_content = f"Documents:\n\n{combined}\n\n---\n\nQuestion: {query}"
        print(f"Files: {len(file_paths)} ({total_chars:,} chars total)")
        for p in file_paths:
            print(f"  {p}")
    else:
        user_content = query
        print("Query only (no files)")

    print(f"Query: {query[:80]}{'...' if len(query) > 80 else ''}")
    print(f"Model: {MODEL}")
    print("-" * 60)
    print("Sending request...")

    client = openai.OpenAI(api_key=api_key)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful analyst. Answer the user's question thoroughly. "
                "If a document is provided, use it and cite it. Use markdown where appropriate."
            ),
        },
        {"role": "user", "content": user_content},
    ]

    start = time.time()
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_completion_tokens=16000,
        reasoning_effort=reasoning_effort,
    )
    elapsed = time.time() - start

    answer = response.choices[0].message.content
    usage = response.usage
    tokens_in = usage.prompt_tokens
    tokens_out = usage.completion_tokens
    total_tokens = usage.total_tokens
    cost = estimate_cost(tokens_in, tokens_out, MODEL)

    print(f"\n{'=' * 60}")
    print("ANSWER")
    print(f"{'=' * 60}\n")
    print(answer)
    print(f"\n{'=' * 60}")
    print("USAGE")
    print(f"{'=' * 60}")
    print(f"  Time:          {elapsed:.1f}s")
    print(f"  Input tokens:  {tokens_in:,}")
    print(f"  Output tokens: {tokens_out:,}")
    print(f"  Total tokens:  {total_tokens:,}")
    print(f"  Est. cost:     ${cost:.4f}")


def _run_rlm(query: str, file_paths: list[str], ticker: str | None,
             form_type: str, reasoning_effort: str, depth: int, iterations: int,
             use_local: bool = False):
    """RLM mode: use Recursive Language Models for deep document analysis."""
    # Import here to avoid slow startup for simple mode
    from .engine import analyze_filing

    if not use_local:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise SecddError("OPENAI_API_KEY environment variable not set")
    else:
        if not os.getenv("LOCAL_MODEL_BASE_URL"):
            raise SecddError(
                "LOCAL_MODEL_BASE_URL environment variable not set. "
                "Set it to your Ollama URL, e.g.: LOCAL_MODEL_BASE_URL=http://localhost:11434/v1"
            )

    if file_paths:
        # Analyze local files with RLM
        parts = []
        filenames = []
        for file_path in file_paths:
            if not os.path.isfile(file_path):
                raise SecddError(f"File not found: {file_path}")
            content = read_file_content(file_path)
            filenames.append(os.path.basename(file_path))
            parts.append(content)

        if len(parts) == 1:
            combined_text = parts[0]
            filing_info = f"Uploaded file: {filenames[0]}"
        else:
            sections = []
            for i, (fname, text) in enumerate(zip(filenames, parts)):
                sections.append(
                    f"\n\n{'=' * 80}\n"
                    f"## DOCUMENT {i + 1}: {fname}\n"
                    f"{'=' * 80}\n\n{text}"
                )
            combined_text = "\n".join(sections)
            filing_info = f"{len(filenames)} document(s): {', '.join(filenames)}"

        total_chars = sum(len(p) for p in parts)
        print(f"Files: {len(file_paths)} ({total_chars:,} chars total)")
        for p in file_paths:
            print(f"  {p}")

    elif ticker:
        # Fetch from EDGAR
        from .edgar import get_filing
        print(f"Fetching {form_type} for {ticker} from EDGAR...")
        try:
            filing = asyncio.run(get_filing(ticker=ticker, form_type=form_type))
        except Exception as e:
            raise SecddError(f"Fetching filing failed: {e}")
        combined_text = filing.text
        filing_info = f"{filing.company_name} ({filing.ticker}) | {filing.form_type} | Filed: {filing.filed_date}"
        print(f"Filing: {filing_info} ({len(combined_text):,} chars)")
    else:
        raise SecddError("RLM mode requires either files or --ticker")

    print(f"Query: {query[:80]}{'...' if len(query) > 80 else ''}")
    mode_label = "RLM + local Ollama" if use_local else "RLM + OpenAI"
    print(f"Mode: {mode_label} (depth={depth}, iterations={iterations}, reasoning={reasoning_effort})")
    print("-" * 60)
    print("Running RLM analysis (this may take 30-90s)...")

    start = time.time()
    result = asyncio.run(analyze_filing(
        query=query,
        filing_text=combined_text,
        filing_info=filing_info,
    ))
    elapsed = time.time() - start

    print(f"\n{'=' * 60}")
    print("ANSWER")
    print(f"{'=' * 60}\n")
    print(result.answer)
    print(f"\n{'=' * 60}")
    print("USAGE")
    print(f"{'=' * 60}")
    print(f"  Time:          {elapsed:.1f}s")
    print(f"  Sub-calls:     {result.num_sub_calls}")
    print(f"  Tokens in:     {result.total_tokens_in:,}")
    print(f"  Tokens out:    {result.total_tokens_out:,}")
    print(f"  Est. cost:     ${result.estimated_cost_usd:.4f}")


def _main_inner():
    parser = argparse.ArgumentParser(
        prog="secdd",
        description=(
            "SEC Deep Dive: Analyze SEC filings and documents using Recursive Language Models.\n"
            "Built on RLM (https://github.com/alexzhang13/rlm) by Alex L. Zhang et al."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            '  secdd "What are Apple\'s main risk factors?"\n'
            '  secdd filing.pdf "Summarize the key points"\n'
            '  secdd --rlm --ticker AAPL "List all related-party transactions"\n'
            '  secdd --rlm 10k.htm "Compare revenue recognition" --depth 3\n'
        ),
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Document paths and/or query. Last argument is always the query.",
    )
    parser.add_argument(
        "--rlm", action="store_true",
        help="Use RLM (Recursive Language Model) for deep multi-pass analysis",
    )
    parser.add_argument(
        "--ticker", "-t",
        help="Fetch filing from EDGAR by ticker (e.g. AAPL, MSFT). Requires --rlm.",
    )
    parser.add_argument(
        "--form", "-f", default="10-K",
        help="SEC form type (default: 10-K)",
    )
    parser.add_argument(
        "--reasoning-effort", "-r",
        default="medium",
        choices=["none", "low", "medium", "high", "xhigh"],
        help="Reasoning effort level (default: medium)",
    )
    parser.add_argument(
        "--depth", "-d", type=int, default=2,
        help="RLM recursion depth: 1=fast, 5=thorough (default: 2)",
    )
    parser.add_argument(
        "--iterations", "-i", type=int, default=15,
        help="RLM max iterations: 5=fast, 30=thorough (default: 15)",
    )
    parser.add_argument(
        "--local", action="store_true",
        help="Use local Ollama model instead of OpenAI (requires LOCAL_MODEL_BASE_URL in .env)",
    )
    parser.add_argument(
        "--environment", "-e",
        default=os.getenv("RLM_ENVIRONMENT", "docker"),
        choices=["local", "docker"],
        help="REPL sandbox: 'docker' (safe, default) or 'local' (fast, dev only)",
    )
    parser.add_argument(
        "--version", "-v", action="version",
        version=f"secdd {__import__('secdd').__version__}",
    )
    args = parser.parse_args()

    if not args.inputs:
        parser.error("provide a query (and optionally files or --ticker)")

    # Parse inputs: last arg is query, rest are file paths.
    if len(args.inputs) == 1:
        file_paths = []
        query = args.inputs[0]
    else:
        file_paths = [os.path.expanduser(p) for p in args.inputs[:-1]]
        query = args.inputs[-1]

    if args.ticker and not args.rlm:
        args.rlm = True  # --ticker implies --rlm

    # Set environment for engine to pick up
    os.environ["RLM_ENVIRONMENT"] = args.environment
    if args.environment == "local":
        print("WARNING: Running with local REPL (no sandbox). Use --environment docker for untrusted inputs.")

    if args.rlm:
        _run_rlm(query, file_paths, args.ticker, args.form,
                  args.reasoning_effort, args.depth, args.iterations, args.local)
    else:
        _run_simple(query, file_paths, args.reasoning_effort)


def main():
    try:
        _main_inner()
    except SecddError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
