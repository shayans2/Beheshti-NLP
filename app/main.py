from fastapi import FastAPI
from app.api.router import api_router

app = FastAPI(title="NLP Services with FastAPI", version="1.0.0")

# Include the API router
app.include_router(api_router)