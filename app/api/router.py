from fastapi import APIRouter
from .endpoints import ner, paraphraser, intent, keyword_extraction

api_router = APIRouter()

# Include routers from different endpoints
api_router.include_router(ner.router, tags=["NER"])
api_router.include_router(paraphraser.router, tags=["Paraphraser"])
api_router.include_router(intent.router, tags=["Intent Classification"])
api_router.include_router(keyword_extraction.router, tags=["Keyword Extraction"])
