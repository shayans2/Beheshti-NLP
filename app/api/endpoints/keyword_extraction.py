from fastapi import APIRouter, Depends, HTTPException

from app.services.index import get_keyword_extraction_service
from app.schemas import KeywordExtractionSchema

from typing import Any
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/extract-keywords", response_model=Any)
def extract_entities(
    text: KeywordExtractionSchema,
    keyword_extraction_service=Depends(get_keyword_extraction_service),
):
    try:
        query = text.query
        top_n = text.top_n
        keywords = keyword_extraction_service.extract_keywords(query, top_n)
        return keywords
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
