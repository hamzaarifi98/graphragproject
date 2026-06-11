from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import APIRouter, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from backend.services.pdf_services import pdf_to_neon
from backend.services.csv_services import csv_to_neon
from backend.services.data_ingestion import main as ingest_all_data
from backend.services.pdf_services.vector_store import get_olist_rows, get_pdf_chunks, list_olist_tables

from backend.services.cache.querry_cache import clear_query_cache
from backend.services.cache.retriever_cache import clear_retriever_cache



app_router = APIRouter()


@app_router.get("/health")
async def health_check():
    return {"status": "ok"}


@app_router.post("/ingest")
async def ingest_all():
    results = await run_in_threadpool(ingest_all_data)
    clear_query_cache()
    clear_retriever_cache("sql_retriever_cache")
    clear_retriever_cache("neo4j_retriever_cache")
    return {"status": "loaded", "results": results}


@app_router.post("/structured/ingest")
async def ingest_structured_data():
    results = await run_in_threadpool(csv_to_neon.ingest_csvs)
    clear_query_cache()
    clear_retriever_cache("sql_retriever_cache")
    return {"status": "loaded", "tables": results}


@app_router.get("/structured")
async def list_structured_tables():
    tables = await run_in_threadpool(list_olist_tables)
    return {"tables": tables}


@app_router.get("/structured/{table_name}")
async def get_structured_data(table_name: str):
    allowed_tables = set(csv_to_neon.CSV_TABLES.values())

    if table_name not in allowed_tables:
        raise HTTPException(status_code=404, detail="Unknown structured table")

    rows = await run_in_threadpool(get_olist_rows, table_name)
    return {"schema": "olist", "table_name": table_name, "rows": rows}


@app_router.post("/pdf/ingest")
async def ingest_local_pdfs():
    results = await run_in_threadpool(pdf_to_neon.main)
    clear_query_cache()
    return {"status": "loaded", "pdfs": results}


@app_router.post("/pdf_upload")
async def pdf_upload(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Upload a PDF file")

    safe_name = Path(file.filename).name

    with TemporaryDirectory() as temp_dir:
        pdf_path = Path(temp_dir) / safe_name
        pdf_path.write_bytes(await file.read())
        result = await run_in_threadpool(pdf_to_neon.ingest_pdf, pdf_path)

    clear_query_cache()

    return {"status": "loaded", "pdf": result}

@app_router.post("/upload_invoice")
async def upload_invoice(file: UploadFile = File(...)):
    return await pdf_upload(file)


@app_router.get("/pdf/{pdf_name}")
async def get_pdf(pdf_name: str):
    chunks = await run_in_threadpool(get_pdf_chunks, pdf_name)

    if not chunks:
        raise HTTPException(status_code=404, detail="PDF not found")

    return {"source_name": pdf_name, "chunks": chunks}



from pydantic import BaseModel
from backend.pipeline.query_pipeline import query_graph


class QueryRequest(BaseModel):
    question: str




# backend/api/routes.py
# replace ask_question with this version

@app_router.post("/ask")
async def ask_question(request: QueryRequest):
    result = await run_in_threadpool(
        query_graph.invoke,
        {"question": request.question},
    )

    return {
        "question": result.get("question"),
        "route": result.get("route"),
        "answer": result.get("answer"),
        "cache_hit": result.get("cache_hit", False),
        "cache_type": result.get("cache_type"),
        "cache_similarity": result.get("cache_similarity"),
        "cached_question": result.get("cached_question"),
    }


@app_router.get("/kg/build")
async def build_kg():
    from backend.services.kg_services.kg_builder import build_knowledge_graph

    result = await run_in_threadpool(build_knowledge_graph)
    clear_query_cache()
    clear_retriever_cache("neo4j_retriever_cache")

    return {"status": "kg built", "details": result}


@app_router.delete("/kg")
async def delete_kg():
    from backend.services.kg_services.kg_query import delete_knowledge_graph

    result = await run_in_threadpool(delete_knowledge_graph)
    clear_query_cache()
    clear_retriever_cache("neo4j_retriever_cache")

    return {"status": "kg deleted", "details": result}


@app_router.get("/kg/ask")
async def ask_kg_question(question: str):
    from backend.services.kg_services.llm_query import ask_graph

    result = await run_in_threadpool(ask_graph, question)

    return result
