# Data-Analyst-Agent

Ask questions about the [Olist Brazilian e-commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (about 100k orders, 2016–2018) in plain English, and get answers computed from the real data:

- **Direct answers** for narrow questions: "How many orders were delivered late?" gives **7,827**.
- **Reports** for a category, city, state, seller or the whole marketplace: KPI tiles, a monthly trend, breakdown charts and a written analysis.
- **"I don't know"** for off-topic questions, and **"I don't have enough data"** for things the dataset doesn't contain (profit, product names, forecasts).

Every answer shows the SQL it ran, so the numbers can be checked. The model runs locally with [Ollama](https://ollama.com) (default `gemma2:2b`) or on NVIDIA's API (`nvidia/nemotron-3.5-lightning-30b-a3b`).

**New to coding?** [BUILD_GUIDE.md](BUILD_GUIDE.md) walks through installing everything and building the project step by step, with no coding knowledge needed.

## Quick start

Requires Python 3.12+ (tested on 3.14) and [Ollama](https://ollama.com/download).

```bash
python -m venv .venv
.venv\Scripts\activate             # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
ollama pull gemma2:2b
cp .env.example .env               # Windows: copy .env.example .env
streamlit run app.py
```

Put the 9 Olist CSVs from Kaggle in `data/raw/`. On first run the app builds `data/olist.duckdb` from them (about 30 seconds). Rebuild it with `python -m analyst.db`.

### Using NVIDIA Nemotron instead of the local model

Add these lines to `.env` (get a key at [build.nvidia.com](https://build.nvidia.com)):

```
NVIDIA_API_KEY=nvapi-...
ANALYST_PROVIDER=nvidia
```

Nemotron is faster and more accurate than Gemma 2B, but questions and query results are sent to NVIDIA. Its thinking mode is off by default (`ANALYST_THINKING=true` turns it on; it's 10–100× slower). All settings and their defaults are in [.env.example](.env.example).

## How it works

```
question → scope guard → router ─┬─ report: resolve entity → prewritten queries → model writes narrative
                                 └─ direct: SQL (built in code or by the model, auto-fixed) → model writes answer
```

1. **Scope guard:**
   - Keyword checks run before any model call. Off-topic questions get "I don't know", and questions about missing data (profit, margins, product names, marketing, forecasts) get "I don't have enough data".
   - The model can also flag a question as out of scope. Its judgement is used only when the question contains no data words.
2. **Router:** the model decides between a report and a direct answer, with keyword rules that correct common misroutes.
3. **Reports:**
   - The city, state or category named in the question is matched to the stored value (accents, typos, state names → codes).
   - Prewritten, verified queries compute the KPIs, charts and key facts.
   - The model only writes the narrative.
4. **Direct answers:**
   - Common ranking and counting questions ("most selling product", "slowest states", "how many orders were canceled") get SQL built in code.
   - Other questions get one model-written query, cleaned to a single statement, and sent back to the model to fix if it fails.
5. **Answer checks:**
   - Single values are written by code.
   - Model-written answers may only contain numbers from the result, and "R$" only on money columns. If an answer fails, the result rows are listed instead.
   - Products have no names in Olist, so they're shown with their category.

Small models write unreliable SQL, so report numbers never come from the model, and the most common question shapes don't depend on it either.

| File | Role |
|---|---|
| `analyst/db.py` | Loads the CSVs into DuckDB, adds the joined `sales` view (one row per order item with order, customer, English category, seller, review score, delivery days and lateness), runs read-only single `SELECT`s |
| `analyst/agent.py` | Prompts, scope guard, `classify` (routing), `retrieve` (SQL), `write_answer`, `write_narrative`; Ollama and NVIDIA providers; follow-up history |
| `analyst/reports.py` | Report queries per entity, name detection and fuzzy matching, SQL templates for rankings and counts |
| `app.py` | Streamlit chat UI: KPI tiles, Plotly charts, tables, narrative, "SQL used" panel, error messages |
| `evals/` | Per-component test datasets, deepeval metrics, LLM judge, run logs and run comparison |

## Evaluation

Each component has its own dataset in `evals/datasets/` and is scored with [deepeval](https://github.com/confident-ai/deepeval):

| Component | Dataset | What is checked | Metric |
|---|---|---|---|
| Scope guard | `scope_guard.json` (20) | In scope / "I don't know" / "not enough data" | Exact match |
| Router | `router.json` (18) | Report vs direct answer | Exact match |
| Entity resolver | `entity_resolver.json` (18) | Finds and matches the city/state/category | Exact match |
| Retriever | `retriever.json` (18) | The SQL returns the same data as a reference query | Result match (values within 1%) |
| Answer generator | `answer_generator.json` (16) | The written answer, given the correct data | GEval correctness, Faithfulness, Answer relevancy |
| Report generator | `report_generator.json` (15) | The report narrative, given the computed facts | Numbers grounded, Faithfulness, GEval report quality |

```bash
pip install -r requirements-dev.txt
python -m evals.run_evals                                   # all components
python -m evals.run_evals retriever router                  # or just some
EVAL_RUN=my_run python -m evals.run_evals                   # name the run (default: <date>_<model>)
python -m evals.compare baseline_gemma2-2b my_run           # compare two runs (also writes markdown)
```

The generators are tested on correct inputs, so their scores don't include retriever mistakes. The LLM judge is `nvidia/nemotron-3-ultra-550b-a55b` via NVIDIA's API (needs `NVIDIA_API_KEY`). Without a key it falls back to a local Ollama model (`EVAL_JUDGE=ollama`, `EVAL_OLLAMA_MODEL=mistral`), which is much slower. Each run is logged to `evals/runs/<run>/` with per-case outputs, scores, the judge's reasons and a `summary.json`.

### Results

Cases passed, from the runs logged in `evals/runs/`:

| Component | Gemma 2B (baseline) | Nemotron 30B | Gemma 2B (prompt fix) | Nemotron 30B (prompt fix) |
|---|---|---|---|---|
| Scope guard | 20/20 | 20/20 | 20/20 | 20/20 |
| Router | 18/18 | 18/18 | 18/18 | 18/18 |
| Entity resolver | 17/18 | 17/18 | 18/18 | 18/18 |
| Retriever | 13/18 | 10/18 | 17/18 | 18/18 |
| Answer generator | 7/16 | 11/16 | not re-run | not re-run |
| Report generator | 1/15 | 7/15 | not re-run | not re-run |

The "prompt fix" runs added SQL templates for common rankings and counts. Many retriever test questions now use those templates rather than model-written SQL, so the retriever gain overstates how much better the models are at new questions. The judge is from the same model family as Nemotron 30B, which may slightly favour it on the judged metrics. Details: [evals/runs/compare_baseline_gemma2-2b_vs_nemotron-3.5-lightning-30b.md](evals/runs/compare_baseline_gemma2-2b_vs_nemotron-3.5-lightning-30b.md).

## Known limitations

- Gemma 2B still gets harder questions wrong when they fall outside the templates (e.g. revenue totals that should exclude canceled orders, "per month" breakdowns). Nemotron handles these better.
- Olist products have no names, only IDs and categories.
- The data covers Sep 2016 – Oct 2018 only.
