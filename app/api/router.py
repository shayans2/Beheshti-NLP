from fastapi import APIRouter
from .endpoints import ner, paraphraser, keyword_extractor

api_router = APIRouter()

# Include routers from different endpoints
api_router.include_router(ner.router, tags=["NER"])
api_router.include_router(paraphraser.router, tags=["Paraphraser"])
api_router.include_router(keyword_extractor.router, tags=["KeywordExtractor"])
