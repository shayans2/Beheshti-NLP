from fastapi import APIRouter, Depends, HTTPException

from app.services.index import get_intent_service
from app.schemas import IntentSchema

from typing import Any
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/intent-classification", response_model=Any)
def intent_classification(text: IntentSchema, intent_service = Depends(get_intent_service)):
    """
    Endpoint for intent classification.

    This endpoint receives a query and training data, processes them using the intent classification service,
    and returns the classified intent.

    Parameters:
    ----------
    text : IntentSchema
        The input data containing the query and training data.
    intent_service : IntentService
        The intent classification service dependency.

    Returns:
    -------
    dict
        The classification results, including indices, values of the nearest neighbors, and the majority class.

    Raises:
    ------
    HTTPException:
        If an internal server error occurs during intent classification.

    Example:
    --------
    Request body:
    {
        "query": "برای من یک کشک بادمجون سفارش بده",
        "data": {
            "رزرو غذا": [
                "یک پیتزا با پپرونی و قارچ بساز .",
                "یک ساندویچ مرغی و سیب‌زمینی بگیر .",
                ...
            ],
            ...
        }
    }

    Response:
    {
        "Indices": [0, 1, 2],
        "Values": [0.2, 0.5, 0.7],
        "Majority Class": 0
    }
    """
    try:
        query = text.query
        data = text.data
        entities = intent_service.intent_classifier(data, query)
        return entities
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
