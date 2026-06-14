# GraphRAG Project

A production-style GraphRAG assistant built on the Olist e-commerce dataset.
The project combines document RAG, SQL retrieval, Neo4j knowledge graph retrieval, Redis caching, and evaluation into one FastAPI + Streamlit application.

You can see sreenshots at Screenshots folder.

## Overview

This system answers questions over both unstructured documents and structured e-commerce data. It can route user questions to the correct retrieval path:

* **PDF RAG** for policy/document questions
* **PostgreSQL SQL retrieval** for structured analytics and aggregations
* **Neo4j graph retrieval** for relationship-based questions
* **Hybrid retrieval** when a question needs multiple sources

The goal is to demonstrate a job-ready GraphRAG architecture with ingestion, retrieval, caching, UI, and evaluation.

## Features

* FastAPI backend with ingestion, retrieval, graph, and evaluation endpoints
* Streamlit frontend for asking questions, uploading PDFs, viewing tables, building the graph, and running evaluations
* PostgreSQL with pgvector for document chunks and embeddings
* Olist structured data ingestion into PostgreSQL
* Neo4j knowledge graph built from customers, orders, products, sellers, payments, reviews, and categories
* LangGraph pipeline for routing and orchestration
* Redis exact, semantic, SQL, and Neo4j retriever caching
* Template-first SQL/Cypher generation with LLM fallback
* Manual reranking using vector similarity and lexical overlap
* DeepEval, RAGAS-ready dependencies, and LangSmith tracing support

## Architecture

```text
User Question
    |
    v
Redis Cache Lookup
    |
    v
LangGraph Router
    |
    |-- PDF Retriever -> pgvector chunks
    |-- SQL Retriever -> PostgreSQL / Olist tables
    |-- KG Retriever  -> Neo4j / Cypher
    |-- Hybrid        -> multiple retrievers
    |
    v
Answer Generation
    |
    v
Cache Save + Response
```

## Tech Stack

* **Backend:** FastAPI, Pydantic, LangGraph, LangChain
* **Frontend:** Streamlit
* **Databases:** PostgreSQL, pgvector, Neo4j
* **Caching:** Redis
* **Evaluation:** DeepEval, RAGAS, LangSmith tracing
* **Deployment:** Docker Compose

## Data

The project uses the Olist e-commerce dataset, including:

* Customers
* Orders
* Order items
* Payments
* Reviews
* Products
* Sellers
* Geolocation
* Product category translations

PDF files can also be ingested and embedded for document-based question answering.

## Running with Docker

Create a `.env` file with the required database, Redis, Neo4j, API, and model settings.

Then start the full stack:

```bash
docker compose up --build
```

Services started by Docker Compose:

* PostgreSQL/pgvector
* Neo4j
* Redis
* FastAPI backend
* Streamlit frontend

Backend:

```text
http://localhost:8000
```

Frontend:

```text
http://localhost:8501
```

## Main API Endpoints

```http
GET  /health
POST /ingest
POST /structured/ingest
GET  /structured
GET  /structured/{table_name}
POST /pdf/ingest
POST /pdf_upload
POST /ask
GET  /kg/build
DELETE /kg
GET  /kg/ask
POST /evaluation/run
POST /evaluation/full/run
```

## Example Questions

```text
Which sellers are connected to the most delayed orders and bad reviews?

What is the total revenue by product category?

What are the top selling products?

What does the uploaded policy say about refunds?

Which payment types are most common?
```

## Evaluation

The project includes router-level and full end-to-end evaluation.

Evaluation checks include:

* Route accuracy
* Answer presence
* Context availability
* Expected answer containment
* Latency
* Cache hit rate
* SQL template hit rate
* KG template hit rate
* Answer relevancy
* Faithfulness
* Contextual relevancy

Run evaluation from the Streamlit UI or through the API:

```http
POST /evaluation/run
POST /evaluation/full/run
```


