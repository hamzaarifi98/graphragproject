# GraphRAG Project

Built GraphRAG assistant on the Olist e-commerce dataset and deployed in AWS ECS.
The project combines document RAG, SQL retrieval, Neo4j knowledge graph retrieval, Redis caching, and evaluation into one FastAPI + Streamlit application and it has authentication to be able to use different users and have access to their data, preventing access to others data. 

For optimization I used SQL and CYPHER templates, where i embedded most popular questions so when a user asks similar question to those embedded, template data get fetched, in this way we prevent using SQL or CYPHER generation and we have some of the costs reduced. Besides that there is REDIS caching where questions get cached and be used for neartime querries, this also reduces latency and cost.

For evaluation I used DeepEval because of Pytest, tracing and monitoring is done by LangSmith, AWS CloudWatch, and manually crafted logs to get more insights if things go south.Metadata are assigned to answers.

Neo4j database is deployed in AWS EC2 and the project is deployed in AWS ECS using FARGATE serverless service, pay-as-you-go compute engine for containers.

Screenshots at Screenshot folder.

## Architecture

```text
    |
Register or Login
    |
User Question
    |
    v
Redis Cache Lookup
    |
SQL or CYPHER template check
    v
LangGraph Router
    |
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
* **Evaluation:** DeepEval, LangSmith tracing
* **Deployment:** Docker, AWS EC2, AWS ECS

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

As policies there are PDF data where there are written different policies. It supports uppload to use specific data.


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
POST /auth/register
POST /auth/login
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

The project includes full end-to-end evaluation.

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

Hallucination and other metrics about LLMs are tracked by LangSmith.


## Improvements

There are future improvements for model and stack selection using Paretto optimisation, Constrains, Weighted score and ablation analysis for specific improvements.
