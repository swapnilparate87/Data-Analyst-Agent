# Data-Analyst-Agent

Ask questions about the [Olist Brazilian e-commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (about 100k orders, 2016–2018) in plain English, and get answers computed from the real data:

- **Direct answers** for narrow questions: "How many orders were delivered late?" gives **7,827**.
- **Reports shaped by the question:**
  - They cover a category, city, state, seller or the whole marketplace.
  - The topic sets the KPIs, charts and analysis: sales, delivery, reviews, sellers or payments. "How is RJ doing on deliveries?" gets a delivery report.
  - A time period ("in 2018", "Q1 2018", "March 2018") filters the report and compares it with the same months a year earlier.
  - Extra filters work too: "health_beauty in SP".
  - Every figure is compared with the marketplace and with peers (e.g. "#21 of 27 states"). The written analysis starts by answering the question and ends with a recommendation.
- **Comparisons:** "Compare SP and RJ", "health_beauty vs watches_gifts reviews" or "Compare revenue in 2017 and 2018" get a side-by-side table and overlaid trends.
- **Several questions in one message:** each one is answered separately. Follow-ups such as "and in 2017?" or "what about deliveries?" reuse the previous question.
- **"I don't know"** for off-topic questions, and **"I don't have enough data"** for things the dataset doesn't contain (profit, product names, forecasts) or periods it doesn't cover.

Every answer shows the SQL it ran, so the numbers can be checked. The model runs on NVIDIA's API (`nvidia/nemotron-3.5-lightning-30b-a3b`, the default when `NVIDIA_API_KEY` is set) or locally with [Ollama](https://ollama.com) (`gemma2:2b`). A **Model** choice in the sidebar switches between them without restarting.

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

### Choosing the model: Nemotron or Gemma

Add your key to `.env` (get one at [build.nvidia.com](https://build.nvidia.com)):

```
NVIDIA_API_KEY=nvapi-...
```

With a key, the app starts with Nemotron 30B; without one, it starts with Gemma 2B. Switch at any time with **Model** in the sidebar (follow-ups keep working), or set the starting model with `ANALYST_PROVIDER=nvidia` or `ollama`. The evaluations use `ANALYST_PROVIDER`.

Nemotron is faster and more accurate than Gemma 2B, but questions and query results are sent to NVIDIA. Its thinking mode is off by default (`ANALYST_THINKING=true` turns it on; it's 10–100× slower). All settings and their defaults are in [.env.example](.env.example).

## How it works

```
message → query analyser ─ for each question ─┬─ out of scope / not in data → "I don't know" / "not enough data"
                                              ├─ compare → side-by-side queries → model writes the comparison
                                              ├─ report  → focused queries + benchmarks → model writes the analysis
                                              ├─ direct  → SQL (code template or model, auto-fixed) → answer
                                              └─ unsure  → model router decides
```

1. **Query analyser** ([analyst/query.py](analyst/query.py), pure code):
   - **Splits the message:** at "?", sentence ends, new lines, numbered items, and "and" / "also" / "then" when a new question starts.
   - **For each question it finds:**
     - every city, state and category mentioned
     - the time period, clipped to the months the data covers
     - the topic: sales, delivery, reviews, sellers or payments
   - **Sets the intent:** out_of_scope, not_in_data, compare, report, direct, or unsure.
   - **Carries context:** "How many orders and how much revenue came from Curitiba?" applies Curitiba to both parts. Follow-ups reuse the previous question.
2. **Scope check:**
   - Questions that mention nothing in the data get "I don't know" without a model call. Questions about missing data (profit, margins, product names, marketing, forecasts) get "I don't have enough data".
   - For unclear cases, the model router can also flag a question as out of scope.
3. **Model router:** only for questions the analyser marks "unsure". It also catches names the analyser didn't recognise, such as typos ("Sao Paolo" → `sao paulo`).
4. **Reports and comparisons** ([analyst/reports.py](analyst/reports.py)):
   - Prewritten, verified queries chosen by topic compute the KPIs, charts and facts.
   - The facts include comparisons: against the marketplace (or the filter, e.g. "all of SP"), the rank among peers, and the same months a year earlier. Year comparisons use the same months for both years, because the data ends in August 2018.
   - The model writes only the narrative: an **Answer** to the question, **Key findings** that interpret the comparisons, and one **Recommendation**. The facts are shown under each report.
5. **Direct answers:**
   - Common ranking and counting questions ("most selling product", "slowest states", "how many orders were canceled") get SQL built in code.
   - Other questions get one model-written query, cleaned to a single statement, and sent back to the model to fix if it fails.
6. **Answer checks:**
   - Single values are written by code.
   - Model-written answers may only contain numbers from the result, and "R$" only on money columns. If an answer fails, the result rows are listed instead.
   - Products have no names in Olist, so they're shown with their category.

Small models write unreliable SQL, so report numbers never come from the model, and the most common question shapes don't depend on it either.

| File | Role |
|---|---|
| `analyst/db.py` | Loads the CSVs into DuckDB, adds the joined `sales` view (one row per order item with order, customer, English category, seller, review score, delivery days and lateness), runs read-only single `SELECT`s |
| `analyst/query.py` | Query analyser: splits messages into questions; finds entities, period and topic; decides each question's intent; handles follow-ups |
| `analyst/agent.py` | Prompts, `ask` (runs each analysed question), `classify` (model router for unclear cases), `retrieve` (SQL), `write_answer`, `write_narrative`; Ollama and NVIDIA providers; history |
| `analyst/reports.py` | Name detection and fuzzy matching, SQL templates for rankings and counts, focused report queries with benchmarks and ranks, comparison queries |
| `app.py` | Streamlit chat UI: model switch (Nemotron / Gemma), one section per question, KPI tiles, Plotly charts, tables, narrative, "Facts behind this analysis" and "SQL used" panels, error messages |
| `evals/` | Per-component test datasets, deepeval metrics, LLM judge, run logs and run comparison |

## Evaluation

Each component has its own dataset in `evals/datasets/` and is scored with [deepeval](https://github.com/confident-ai/deepeval):

| Component | Dataset | What is checked | Metric |
|---|---|---|---|
| Query analyser | `query_analyzer.json` (21) | Splitting multi-question messages; each question's intent, entities, topic and period; follow-ups | Plan match (field by field) |
| Scope guard | `scope_guard.json` (20) | In scope / "I don't know" / "not enough data" | Exact match |
| Router | `router.json` (19) | Report vs comparison vs direct answer | Exact match |
| Entity resolver | `entity_resolver.json` (18) | Finds and matches the city/state/category | Exact match |
| Retriever | `retriever.json` (18) | The SQL returns the same data as a reference query | Result match (values within 1%) |
| Answer generator | `answer_generator.json` (16) | The written answer, given the correct data | GEval correctness, Faithfulness, Answer relevancy |
| Report generator | `report_generator.json` (15) | The narrative of focused reports and comparisons (e.g. "How is RJ doing on deliveries?", "Compare SP and RJ"), scored against the facts it was given | Numbers grounded, Faithfulness, GEval report quality (answers the question, comparisons, recommendation) |

```bash
pip install -r requirements-dev.txt
python -m pytest tests                                      # quick code-only checks, no model (seconds)
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
| Answer generator | 7/16 | 11/16 | 15/16 | 16/16 |
| Report generator | 1/15 | 7/15 | 0/15 | 5/15 |
| **Total** | **76/105** | **83/105** | **88/105** | **95/105** |

The "prompt fix" runs added SQL templates for common rankings and counts, and changed how answers are written (rows listed instead of "see the table", "R$" only on money). The report prompt didn't change, so report scores differ only by run-to-run variation. Reports are factually accurate (93–100% faithful) but mostly fail "report quality": the judge wants bullets that interpret the facts, not repeat them. Some of Nemotron's "numbers grounded" failures are figures it calculated from the facts (e.g. the combined share of the top three cities) rather than invented ones. Many retriever test questions now use those templates rather than model-written SQL, so the retriever gain overstates how much better the models are at new questions. The judge is from the same model family as Nemotron 30B, which may slightly favour it on the judged metrics. Per-question details are in the comparison reports in `evals/runs/` (`compare_*.md`).

Evaluation runs take one to two hours, mostly because the judge is a reasoning model and NVIDIA's free tier is rate-limited. Run one evaluation at a time: parallel runs hit the rate limit. Judge calls retry on rate limits, and any judge errors are reported separately instead of being scored as failures. The run keeps Windows awake until it finishes, but closing a laptop lid can still pause it.

## Known limitations

- The query analyser uses keyword rules. Unusual wording can still be split or routed wrongly; unclear questions fall back to the model router.
- A city that is also a state name ("Rio de Janeiro", "São Paulo") is treated as the city unless the question says "state".
- Gemma 2B still gets harder questions wrong when they fall outside the templates (e.g. revenue totals that should exclude canceled orders, "per month" breakdowns). Nemotron handles these better.
- Olist products have no names, only IDs and categories.
- The data covers Sep 2016 – Aug 2018 in practice, with very few orders before 2017, so year-over-year comparisons start in 2018.
