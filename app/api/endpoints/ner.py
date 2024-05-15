from fastapi import APIRouter, Depends, HTTPException

from app.services.index import get_ner_service
from app.schemas import NERSchema

from typing import Any
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/extract-entities", response_model=Any)
def extract_entities(text: NERSchema, ner_service = Depends(get_ner_service)):
    try:
        query = text.query
        entities = ner_service.get_full_entity_names(query)
        return entities
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
