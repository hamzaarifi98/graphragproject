from fastapi import FastAPI

from backend.api.routes import app_router

app = FastAPI(title="RagInvoice API")
app.include_router(app_router)
