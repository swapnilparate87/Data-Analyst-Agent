"""Streamlit chat UI for the Olist data analyst agent. Run with: streamlit run app.py"""

import httpx
import ollama
import openai
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # before importing the agent, which reads its settings from the environment

from analyst import db  # noqa: E402
from analyst.agent import MODEL, OLLAMA_HOST, PROVIDER, REQUEST_TIMEOUT, Analyst, Chart, Report  # noqa: E402

# Categorical hues in fixed order (validated palette from the dataviz reference).
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

EXAMPLES = [
    "Give me a report on the health_beauty category",
    "How is Rio de Janeiro performing as a market?",
    "Which 10 sellers made the most revenue in 2018?",
    "How many orders were delivered late?",
]

st.set_page_config(page_title="Olist Data Analyst", page_icon="📊", layout="wide")


@st.cache_resource
def get_connection():
    return db.connect()


def figure(chart: Chart):
    df, kw = chart.data, dict(title=chart.title, color_discrete_sequence=SERIES_COLORS)
    if chart.color:
        kw["color"] = chart.color
    if chart.chart_type == "bar":
        fig = px.bar(df, x=chart.x, y=chart.y, **kw)
    elif chart.chart_type == "horizontal_bar":
        fig = px.bar(df, x=chart.y, y=chart.x, orientation="h", **kw)
        fig.update_yaxes(categoryorder="total ascending")
    elif chart.chart_type == "line":
        fig = px.line(df, x=chart.x, y=chart.y, markers=True, **kw)
        fig.update_traces(line_width=2, marker_size=8)
    elif chart.chart_type == "pie":
        fig = px.pie(df, names=chart.x, values=chart.y, hole=0.5, **kw)
        fig.update_traces(marker_line_width=2, marker_line_color="rgba(0,0,0,0)")
    else:  # scatter
        fig = px.scatter(df, x=chart.x, y=chart.y, **kw)
        fig.update_traces(marker_size=9)
    if chart.chart_type in ("bar", "horizontal_bar"):
        fig.update_layout(bargap=0.3)
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), legend_title_text="")
    return fig


def render(report: Report, key: str):
    if report.kpis:
        cols = st.columns(min(len(report.kpis), 6))
        for i, kpi in enumerate(report.kpis):
            cols[i % len(cols)].metric(kpi["label"], kpi["value"], kpi.get("note"), delta_color="off")

    if report.charts:
        cols = st.columns(2) if len(report.charts) > 1 else [st.container()]
        for i, chart in enumerate(report.charts):
            with cols[i % len(cols)]:
                if chart.chart_type == "table":
                    st.markdown(f"**{chart.title}**")
                    st.dataframe(chart.data, hide_index=True, width="stretch")
                    continue
                tab_chart, tab_data = st.tabs(["Chart", "Data"])
                tab_chart.plotly_chart(figure(chart), width="stretch", key=f"{key}-chart-{i}")
                tab_data.dataframe(chart.data, hide_index=True, width="stretch")

    st.markdown(report.text)

    if report.queries:
        with st.expander(f"SQL used ({len(report.queries)} queries)"):
            for purpose, sql, result in report.queries:
                st.markdown(f"**{purpose}**")
                st.code(sql, language="sql")
                if isinstance(result, str):
                    st.error(result)
                else:
                    st.caption(f"{len(result):,} rows")


st.title("📊 Olist Data Analyst")
st.caption("Ask about products, categories, cities, states, sellers or time periods. "
           f"Simple questions get a direct answer; broader ones get a report. Model: `{MODEL}` "
           + ("(NVIDIA API)" if PROVIDER == "nvidia" else "(local, via Ollama)"))

if "history" not in st.session_state:
    st.session_state.history = []  # list of (question, Report)
# Streamlit reloads edited modules on rerun, but an Analyst kept in the session would still run
# the old code. Rebuild it when the class has changed, keeping the follow-up context.
if type(st.session_state.get("analyst")) is not Analyst:
    previous = st.session_state.get("analyst")
    with st.spinner("Loading data (first run builds data/olist.duckdb, ~30s)..."):
        st.session_state.analyst = Analyst(get_connection())
    st.session_state.analyst.history = getattr(previous, "history", [])

with st.sidebar:
    st.subheader("Try asking")
    for example in EXAMPLES:
        if st.button(example, width="stretch"):
            st.session_state.pending = example
    if st.button("New conversation", type="primary", width="stretch"):
        st.session_state.analyst = Analyst(get_connection())
        st.session_state.history = []
        st.rerun()

for i, (question, report) in enumerate(st.session_state.history):
    st.chat_message("user").markdown(question)
    with st.chat_message("assistant"):
        render(report, key=str(i))

question = st.chat_input("e.g. Give me a report on sales in Sao Paulo") or st.session_state.pop("pending", None)
if question:
    st.chat_message("user").markdown(question)
    with st.chat_message("assistant"):
        with st.status("Analyzing...", expanded=False) as status:
            try:
                report = st.session_state.analyst.ask(question, on_step=status.write)
            except ConnectionError:
                status.update(label="Ollama is not running", state="error")
                st.error(f"Could not reach Ollama at {OLLAMA_HOST}. Start the Ollama app (or `ollama serve`) and retry.")
                st.stop()
            except (httpx.TimeoutException, openai.APITimeoutError):
                status.update(label="The model took too long", state="error")
                st.error(f"`{MODEL}` didn't answer within {REQUEST_TIMEOUT:.0f}s. Check that no other program "
                         "is using Ollama (`ollama ps`), then try again or ask a narrower question.")
                st.stop()
            except openai.APIError as e:
                status.update(label="NVIDIA API error", state="error")
                st.error(f"NVIDIA API error: {e}. Check NVIDIA_API_KEY in .env and the model name.")
                st.stop()
            except ollama.ResponseError as e:
                status.update(label="Ollama error", state="error")
                hint = f" Pull it with `ollama pull {MODEL}`." if e.status_code == 404 else ""
                st.error(f"Ollama error: {e.error}.{hint}")
                st.stop()
            status.update(label="Done", state="complete")
        render(report, key=str(len(st.session_state.history)))
    st.session_state.history.append((question, report))
