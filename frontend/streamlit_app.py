from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests
import streamlit as st


DEFAULT_API_URL = os.getenv("GRAPHRAG_API_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 120


st.set_page_config(
    page_title="GraphRAG",
    page_icon="R",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_state() -> None:
    st.session_state.setdefault("api_url", DEFAULT_API_URL)
    st.session_state.setdefault("messages", [])


def api_url(path: str) -> str:
    base_url = st.session_state.api_url.rstrip("/")
    return f"{base_url}/{path.lstrip('/')}"


def request_json(
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    json: dict[str, Any] | None = None,
    files: dict[str, Any] | None = None,
    timeout: int = REQUEST_TIMEOUT,
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        response = requests.request(
            method,
            api_url(path),
            params=params,
            json=json,
            files=files,
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        return None, "Could not connect to the API. Start FastAPI on the configured URL."
    except requests.exceptions.Timeout:
        return None, "The API request timed out."
    except requests.exceptions.HTTPError as exc:
        detail = response.text
        try:
            detail = response.json().get("detail", detail)
        except ValueError:
            pass
        return None, f"API returned {response.status_code}: {detail}"
    except requests.exceptions.RequestException as exc:
        return None, str(exc)

    if not response.content:
        return {}, None

    try:
        return response.json(), None
    except ValueError:
        return {"raw": response.text}, None


def show_error(error: str | None) -> bool:
    if error:
        st.error(error)
        return True
    return False


def status_badge(value: Any, label: str) -> None:
    st.metric(label, "Yes" if value else "No")


def render_sidebar() -> None:
    with st.sidebar:
        st.title("GraphRAG")
        st.caption("GraphRAG Retriever.")

        st.text_input(
            "API URL",
            key="api_url",
            help="Set GRAPHRAG_API_URL to change the default.",
        )

        data, error = request_json("GET", "/health", timeout=10)
        if error:
            st.warning("API offline")
            st.caption(error)
        else:
            st.success(f"API {data.get('status', 'ok')}")

        st.divider()
        st.caption("Run backend:")
        st.code("uvicorn backend.main:app --reload", language="bash")
        st.caption("Run frontend:")
        st.code("streamlit run frontend/streamlit_app.py", language="bash")


def render_answer_metadata(result: dict[str, Any]) -> None:
    with st.expander("Routing and retrieval details", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Route", result.get("route") or "unknown")
        col2.metric("Cache", "hit" if result.get("cache_hit") else "miss")
        col3.metric("Retriever cache", "hit" if result.get("retriever_cache_hit") else "miss")
        col4.metric("Similarity", result.get("cache_similarity") or "-")

        details = {
            "cache_type": result.get("cache_type"),
            "cached_question": result.get("cached_question"),
            "sql_source": result.get("sql_source"),
            "sql_template_hit": result.get("sql_template_hit"),
            "sql_template_name": result.get("sql_template_name"),
            "sql_template_similarity": result.get("sql_template_similarity"),
            "cypher_source": result.get("cypher_source"),
            "kg_template_hit": result.get("kg_template_hit"),
            "kg_template_name": result.get("kg_template_name"),
            "kg_template_similarity": result.get("kg_template_similarity"),
            "retriever_cache_type": result.get("retriever_cache_type"),
            "retriever_cache_hits": result.get("retriever_cache_hits"),
        }
        st.json({key: value for key, value in details.items() if value not in (None, {}, [])})


def render_chat_tab() -> None:
    st.header("Ask Your Data")
    st.caption("Ask questions about policies, customers, orders, delays, top sells etc.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask about invoices, policies, orders, customers, products...")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking through the RAG pipeline..."):
            result, error = request_json("POST", "/ask", json={"question": question}, timeout=180)

        if show_error(error):
            return

        answer = result.get("answer") or "No answer returned."
        st.markdown(answer)
        render_answer_metadata(result)

    st.session_state.messages.append({"role": "assistant", "content": answer})


def render_ingestion_tab() -> None:
    st.header("Load Data")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Ingest All Data", type="primary", use_container_width=True):
            with st.spinner("Running full ingestion..."):
                result, error = request_json("POST", "/ingest", timeout=600)
            if not show_error(error):
                st.success("All data loaded.")
                st.json(result)

    with col2:
        if st.button("Ingest Structured CSVs", use_container_width=True):
            with st.spinner("Loading CSV data into Neon..."):
                result, error = request_json("POST", "/structured/ingest", timeout=600)
            if not show_error(error):
                st.success("Structured data loaded.")
                st.json(result)

    with col3:
        if st.button("Ingest Local PDFs", use_container_width=True):
            with st.spinner("Embedding local PDFs..."):
                result, error = request_json("POST", "/pdf/ingest", timeout=600)
            if not show_error(error):
                st.success("PDFs loaded.")
                st.json(result)

    st.subheader("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])
    if uploaded_file and st.button("Upload and Ingest PDF", use_container_width=True):
        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                "application/pdf",
            )
        }
        with st.spinner(f"Ingesting {uploaded_file.name}..."):
            result, error = request_json("POST", "/pdf_upload", files=files, timeout=600)
        if not show_error(error):
            st.success("PDF uploaded and indexed.")
            st.json(result)


def render_structured_tab() -> None:
    st.header("Structured Data")
    result, error = request_json("GET", "/structured")
    if show_error(error):
        return

    tables = result.get("tables", [])
    if not tables:
        st.info("No structured tables were returned. Run structured ingestion first.")
        return

    table_options = {
        table["table_name"]: table
        for table in tables
        if isinstance(table, dict) and table.get("table_name")
    }
    if not table_options:
        st.error("The API returned table data in an unexpected format.")
        st.json(tables)
        return

    def format_table_option(table_name: str) -> str:
        table = table_options[table_name]
        row_count = table.get("row_count")
        if row_count is None:
            return table_name
        return f"{table_name} ({row_count:,} rows)"

    table_name = st.selectbox(
        "Table",
        list(table_options),
        format_func=format_table_option,
    )
    if not table_name:
        return

    with st.spinner(f"Loading {table_name}..."):
        table_result, table_error = request_json("GET", f"/structured/{table_name}")
    if show_error(table_error):
        return

    rows = table_result.get("rows", [])
    st.caption(f"{len(rows)} rows returned from `{table_name}`")
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("This table did not return any rows.")


def render_kg_tab() -> None:
    st.header("Knowledge Graph")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Build Knowledge Graph", type="primary", use_container_width=True):
            with st.spinner("Building graph in Neo4j..."):
                result, error = request_json("GET", "/kg/build", timeout=600)
            if not show_error(error):
                st.success("Knowledge graph built.")
                st.json(result)

    with col2:
        if st.button("Delete Knowledge Graph", use_container_width=True):
            with st.spinner("Deleting graph data..."):
                result, error = request_json("DELETE", "/kg", timeout=600)
            if not show_error(error):
                st.success("Knowledge graph deleted.")
                st.json(result)

    st.subheader("Ask Graph Directly")
    question = st.text_input("Graph question")
    if question and st.button("Ask KG", use_container_width=True):
        with st.spinner("Querying Neo4j..."):
            result, error = request_json("GET", "/kg/ask", params={"question": question}, timeout=180)
        if not show_error(error):
            st.json(result)


def render_evaluation_tab() -> None:
    st.header("Evaluation")

    col1, col2, col3 = st.columns([1, 1, 2])
    limit = col1.number_input("Limit", min_value=1, value=5, step=1)
    run_deepeval = col2.checkbox("Run DeepEval", value=False)
    print_results = col3.checkbox("Print backend results", value=True)

    router_col, full_col = st.columns(2)
    with router_col:
        if st.button("Run Router Evaluation", use_container_width=True):
            with st.spinner("Evaluating router..."):
                result, error = request_json(
                    "POST",
                    "/evaluation/run",
                    params={"limit": limit},
                    timeout=600,
                )
            if not show_error(error):
                st.success("Router evaluation complete.")
                st.json(result.get("summary", {}))
                if result.get("results"):
                    st.dataframe(pd.DataFrame(result["results"]), use_container_width=True)

    with full_col:
        if st.button("Run Full Evaluation", use_container_width=True):
            with st.spinner("Running full evaluation..."):
                result, error = request_json(
                    "POST",
                    "/evaluation/full/run",
                    params={
                        "limit": limit,
                        "run_deepeval": run_deepeval,
                        "print_results": print_results,
                    },
                    timeout=900,
                )
            if not show_error(error):
                st.success("Full evaluation complete.")
                st.json(result)

    with st.expander("Preview evaluation questions"):
        preview_limit = st.number_input("Preview limit", min_value=1, value=10, step=1)
        preview_col1, preview_col2 = st.columns(2)
        with preview_col1:
            if st.button("Router Questions", use_container_width=True):
                result, error = request_json(
                    "GET",
                    "/evaluation/questions",
                    params={"limit": preview_limit},
                )
                if not show_error(error):
                    st.json(result)
        with preview_col2:
            if st.button("Full Questions", use_container_width=True):
                result, error = request_json(
                    "GET",
                    "/evaluation/full/questions",
                    params={"limit": preview_limit},
                )
                if not show_error(error):
                    st.json(result)


def main() -> None:
    init_state()
    render_sidebar()

    tab_chat, tab_ingest, tab_structured, tab_kg, tab_eval = st.tabs(
        ["Ask", "Ingest", "Structured Data", "Knowledge Graph", "Evaluation"]
    )

    with tab_chat:
        render_chat_tab()
    with tab_ingest:
        render_ingestion_tab()
    with tab_structured:
        render_structured_tab()
    with tab_kg:
        render_kg_tab()
    with tab_eval:
        render_evaluation_tab()


if __name__ == "__main__":
    main()
