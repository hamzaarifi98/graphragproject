from fastapi import FastAPI

from backend.api.evaluation_routes import evaluation_router
from backend.api.routes import app_router
from backend.api.auth import router

app = FastAPI(title="GraphRAG API")
app.include_router(app_router)
app.include_router(evaluation_router)
app.include_router(router)

