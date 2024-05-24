from fastapi import APIRouter
from .endpoints import ner, paraphraser

api_router = APIRouter()

# Include routers from different endpoints
api_router.include_router(ner.router, tags=["NER"])
api_router.include_router(paraphraser.router, tags=["Paraphraser"])
