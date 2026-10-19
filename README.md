# Temporal Knowledge Graph QA — LLM Benchmark

Evaluate the supplied **497-question** dataset using any OpenAI-compatible chat endpoint (Groq, OpenAI, Ollama, or another compatible provider). Python 3.10+; no third-party packages required.

The uploaded `train_test_dataset500(9).json` is included as `data/questions.json` (497 questions). The benchmark uses this path by default; no manual dataset copy is needed.

## Windows / Anaconda quick start

```bat
conda create -n tkgqa python=3.11 -y
conda activate tkgqa
cd TKGQA_LLMs
python benchmark.py --dry-run --limit 3 --output results/check
set GROQ_API_KEY=YOUR_KEY
set LLM_MODEL=YOUR_AVAILABLE_MODEL_ID
python benchmark.py --modes io cot sc --limit 5 --output results/pilot
python benchmark.py --modes io cot sc --output results/full
```

For PowerShell use `$env:GROQ_API_KEY="YOUR_KEY"` and `$env:LLM_MODEL="YOUR_AVAILABLE_MODEL_ID"`. For Linux/macOS use `export`. Select a model ID supported by your provider; availability changes. Never commit credentials.

Other endpoints:

```bat
python benchmark.py --base-url https://api.openai.com/v1 --api-key-env OPENAI_API_KEY --model YOUR_MODEL --modes io --output results/openai
python benchmark.py --base-url http://localhost:11434/v1 --api-key-env OLLAMA_API_KEY --model YOUR_INSTALLED_MODEL --modes io --output results/ollama
```

Ollama must be running and the model already downloaded. Its local API normally needs no key. Some models require different token-limit or temperature parameters; adapt `call_api` to the model API if rejected. Keys are read only from environment variables.

## Experiments

| Mode | Behavior |
|---|---|
| `io` | Direct JSON answer |
| `cot` | Temporal constraint checking with a brief explanatory justification; not a captured private reasoning trace |
| `sc` | Five independent calls by default, temperature 0.7; vote on normalized complete answer sets |
| `rag` | BM25 retrieval over independent local evidence, then evidence-only answering |

SC ties choose the first encountered candidate; `strict_majority` explicitly reports whether the winner exceeds half of all requested samples. Consistency uses valid samples only; errors are recorded separately. Budget: all 497 questions in IO + CoT + SC with five samples require 3,479 logical calls, plus retries. Start with a pilot. No real API evaluation results are included.

## RAG requires factual evidence

The question dataset is not an evidence corpus. Create a UTF-8 JSONL file with one independent source document/event per line:

```json
{"id":"source-event-1","text":"An independently sourced dated event statement.","source":"citation or source URL"}
```

```bat
python benchmark.py --modes rag --corpus data/events.jsonl --top-k 5 --output results/rag
```

Prefer the original temporal graph/event corpus used to construct the dataset. Exact graph actor names, dates, and first/last ordering may not be recoverable from general model knowledge or a generic web corpus. BM25 does lexical retrieval, not an exhaustive temporal graph query; top-k evidence may be insufficient for first/last or complete multi-answer questions. Record corpus provenance and keep evaluation labels out of evidence. QA-shaped records are rejected, but provenance still needs manual review.

## Outputs and metrics

Each unique `--output` prefix produces:

- `.jsonl`: per-question answers, brief explanations, individual SC samples, errors, evidence IDs/text/scores, latency, dataset SHA256, and token usage.
- `.csv`: Excel-readable question-level results.
- `.summary.json`: accuracy (exact answer-set match), macro F1, row count, and error count by method.

Precision/recall/F1 operate on answer **sets**, with Unicode NFKC, case and whitespace normalization. All listed reference answers are treated as required targets, not aliases. No fuzzy alias or automatic date conversion is applied. API/parse failures score zero; partial SC failures remain visible. Dry runs perform no requests and produce no accuracy estimate. Retrieval hit stays null without independently labeled relevant document IDs. Evidence token counts, relevance ratings, and monetary costs are not fabricated. Usage totals reflect returned API usage for successfully received responses, including parse failures, and may exclude failed/ambiguous network attempts.

Each invocation starts a new output prefix; existing JSONL files are protected. Completed records flush immediately, so interrupted runs retain partial JSONL; automatic resume and final CSV recovery are not implemented. No train/test partition is inferred from the filename; all 497 records are evaluated unless `--limit` is set. Do not tune prompts on a held-out test set. Corpus and endpoint settings should be archived alongside results for reproducibility; stochastic sampling is not guaranteed deterministic.

## Tests

```bat
python -m unittest discover -s tests -v
```

Tests cover scoring, label isolation, parsing, retrieval, mocked full execution, SC usage and voting, and failed-call scoring. They do not establish live provider compatibility or model accuracy.
