from fastapi import APIRouter, Depends, HTTPException

from app.services.index import get_paraphrase_service
from app.schemas import ParaphraserSchema

from typing import Any
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/paraphrase", response_model=Any)
def paraphrase(text: ParaphraserSchema, service = Depends(get_paraphrase_service)):
    try:
        query = text.query
        paraphrased = service.paraphrase(query)
        return {
            'result': paraphrased
        }
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
