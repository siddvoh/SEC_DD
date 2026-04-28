# SEC Deep Dive

Analyze entire SEC filings with questions that require reading the *whole document*, not just retrieved chunks. Powered by [Recursive Language Models](https://arxiv.org/abs/2512.24601) by Alex L. Zhang, Tim Kraska, and Omar Khattab.

Standard RAG retrieves a few chunks and hopes the answer is in there. SEC Deep Dive stores the entire filing as a variable in a Python REPL and lets GPT-5 **programmatically navigate** it: grepping sections, slicing tables, and spawning recursive sub-LM calls over specific parts. It handles 10-Ks that are 200+ pages without truncation.

**Good for questions like:**
- "List every related-party transaction anywhere in this filing" (full-document scan)
- "Compare revenue recognition across these 3 filings" (cross-document reasoning)
- "What's the ratio of non-performing loans to total assets?" (multi-section financial math)

## Install

Clone and install in editable mode. There is no PyPI package: everything runs from source.

```bash
git clone https://github.com/siddvoh/SEC_DD.git
cd SEC_DD
pip install -e .
```

That's it. `pip install -e .` installs the dependencies from `pyproject.toml` and wires up the `secdd` CLI.

### Prerequisites

- Python 3.11+
- An [OpenAI API key](https://platform.openai.com/api-keys) with access to GPT-5 models (or run fully local with [Ollama](#local-models-with-ollama-optional))

Set the key in your shell or in a `.env` file (copied from `.env.example`):

```bash
cp .env.example .env   # then edit .env and paste your key
# or:
export OPENAI_API_KEY=sk-...
```

Your key stays on your machine. `secdd` reads it at runtime and passes it straight to OpenAI.

## CLI

```bash
# Simple query (single LLM call)
secdd "What are the main risk factors in retail?"

# Analyze local files
secdd filing.pdf "What are the main risk factors?"
secdd 10k_2023.htm 10k_2024.htm "Compare revenue recognition policies"

# RLM mode: deep multi-pass analysis with recursive sub-calls
secdd --rlm filing.pdf "List every related-party transaction"

# Fetch directly from EDGAR by ticker
secdd --ticker AAPL "Summarize the risk factors in the latest 10-K"
secdd --ticker MSFT --form 10-Q "What changed in revenue recognition?"
```

**Options:**

| Flag | Description |
|------|-------------|
| `--rlm` | Enable Recursive Language Model mode (deep analysis, 30-90s) |
| `--ticker, -t` | Fetch filing from EDGAR by ticker (implies `--rlm`) |
| `--form, -f` | SEC form type (default: `10-K`) |
| `--reasoning-effort, -r` | `none`, `low`, `medium` (default), `high`, `xhigh` |
| `--depth, -d` | RLM recursion depth, 1=fast, 5=thorough (default: 2) |
| `--iterations, -i` | RLM max iterations, 5=fast, 30=thorough (default: 15) |
| `--environment, -e` | REPL sandbox: `docker` (default, safe) or `local` (fast, dev only) |
| `--local` | Use local Ollama model instead of OpenAI |
| `--version, -v` | Print version and exit |

## Python Library

```python
import asyncio
from secdd.engine import analyze_filing

result = asyncio.run(analyze_filing(
    query="What are Apple's biggest risk factors?",
    filing_text=open("apple_10k.txt").read(),
    filing_info="Apple Inc (AAPL) | 10-K | Filed: 2024-11-01",
))

print(result.answer)
print(f"Cost: ${result.estimated_cost_usd:.3f}")
print(f"Sub-calls: {result.num_sub_calls}")
```

### Fetch from EDGAR programmatically

```python
from secdd.edgar import get_filing

filing = asyncio.run(get_filing(ticker="AAPL", form_type="10-K"))
print(f"{filing.company_name} | {filing.form_type} | {len(filing.text):,} chars")
```

## Configuration

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes (unless `--local`) | Your OpenAI API key |
| `EDGAR_USER_AGENT` | Recommended | Contact info the SEC wants in the User-Agent, e.g. `Jane Doe <jane@example.com>` |
| `RLM_ENVIRONMENT` | No | `docker` (default, sandboxed) or `local` (in-process, faster but unsafe with untrusted inputs) |
| `RLM_MAX_WORKERS` | No | Max parallel RLM completions (default `3`). Lower if you see OOM. |
| `LOCAL_MODEL_BASE_URL` | No | Ollama URL (e.g. `http://localhost:11434/v1`) |
| `LOCAL_ROOT_MODEL_NAME` | No | Local root model (e.g. `qwen3:8b`) |
| `LOCAL_SUB_MODEL_NAME` | No | Local sub model (e.g. `qwen3:4b`) |

Set these in a `.env` file (loaded automatically) or export them in your shell. `.env` is gitignored: your key never leaves your machine.

## Local Models with Ollama (Optional)

Run analysis using local models via [Ollama](https://ollama.com) instead of OpenAI. No API key needed.

```bash
# 1. Install and start Ollama
ollama serve
ollama pull qwen3:8b && ollama pull qwen3:4b

# 2. Set environment variables
export LOCAL_MODEL_BASE_URL=http://localhost:11434/v1
export LOCAL_ROOT_MODEL_NAME=qwen3:8b
export LOCAL_SUB_MODEL_NAME=qwen3:4b

# 3. Use --local flag
secdd --local --rlm filing.pdf "Summarize the risk factors"
```

In Python, pass `use_local=True` to `analyze_filing()`.

## Architecture

```
User question + ticker
        |
        v
   EDGAR API --> Fetch full filing text (free, no key needed)
        |
        v
   RLM Engine (github.com/alexzhang13/rlm)
        |
        |-- Root LM: GPT-5.4
        |     Receives: query + metadata (NOT the filing text)
        |     Writes: Python code to navigate the filing
        |     Calls: llm_query() for semantic sub-analysis
        |
        +-- Sub LM: GPT-5.4-nano
              Receives: specific section/chunk from root LM's code
              Returns: structured analysis back to root LM
        |
        v
   FINAL(answer) with section citations
```

The filing text lives in the REPL as a Python variable `context`. The root LM never has it in its context window: it writes code like `context[50000:80000]` or `re.findall(r'Item 7', context)` to navigate, then uses `llm_query()` to reason about specific chunks.

## Cost

A typical single-filing query costs **$0.10 to $0.30**. Cross-document queries with many sub-calls can hit $0.50 to $1.00.

| Model | Role | Input | Output |
|-------|------|-------|--------|
| GPT-5.4 | Root (orchestration) | $2.50/1M tokens | $15.00/1M tokens |
| GPT-5.4-nano | Sub (bulk processing) | $0.20/1M tokens | $1.25/1M tokens |

## Project Structure

```
secdd/
├── secdd/              # Main package (CLI + core library)
│   ├── __init__.py
│   ├── cli.py          # `secdd` command entry point
│   ├── engine.py       # RLM wrapper, model config, cost tracking
│   ├── edgar.py        # EDGAR API (fetch filings by ticker)
│   └── prompts.py      # Finance + government RLM system prompts
├── tests/              # Pytest suite (pure-function tests, no network)
├── pyproject.toml
├── requirements.txt
├── .env.example
└── LICENSE
```

## Acknowledgments

This project is a thin SEC-filing wrapper around the [Recursive Language Models (RLM)](https://github.com/alexzhang13/rlm) library by **Alex L. Zhang**, **Tim Kraska**, and **Omar Khattab**. RLM is what makes full-document analysis possible.

Please cite their paper if you use this:

> Alex L. Zhang, Tim Kraska, Omar Khattab. "Recursive Language Models." arXiv:2512.24601, 2026.

```bibtex
@misc{zhang2026recursivelanguagemodels,
      title={Recursive Language Models},
      author={Alex L. Zhang and Tim Kraska and Omar Khattab},
      year={2026},
      eprint={2512.24601},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
}
```

## Limitations

- EDGAR rate limit: ~10 requests/second. The code handles this but bulk fetching is slow.
- Some older SEC filings are in SGML/ASCII format and parse poorly.
- RLM queries take 30 to 90 seconds due to multiple LM calls. Not suitable for autocomplete-style UX.
- Use `RLM_ENVIRONMENT=docker` in production. The `local` REPL uses in-process code execution, which is fine for dev but not safe with untrusted inputs.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT. See [LICENSE](LICENSE). The full source for the CLI, EDGAR fetcher, and prompts is in this repo.
