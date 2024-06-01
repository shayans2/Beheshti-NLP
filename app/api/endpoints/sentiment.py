from fastapi import APIRouter, Depends, HTTPException

from app.services.index import get_sentiment_service
from app.schemas import SentimentSchema

from typing import Any
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/sentiment-classifcation", response_model=Any)
def sentiment_classification(info: SentimentSchema, sentiment_service = Depends(get_sentiment_service)):
    """
    Endpoint for sentiment classification.

    This endpoint receives a query and training data, processes them using the sentiment classification service,
    and returns the classified sentiment.

    Parameters:
    ----------
    text : SentimentSchema
        The input data containing the query and training data.
    senitment_service : SentimentService
        The sentiment classification service dependency.

    Returns:
    -------
    dict
        The classification results, including indices, values of the nearest neighbors, and the majority class.

    Raises:
    ------
    HTTPException:
        If an internal server error occurs during sentiment classification.

    Example:
    --------
    Request body:
    {
        "query": "غذا بد مزه بود",
        "data": {
            "منفی": [
                "غذا دیر به دستم رسید",
                "غذا سرد بود"
                ...
            ],
            ...
        }
    }

    Response:
    
    {
        "Indices": [0, 0, 0, 0],
        "Values": [
            9.540067672729492, 9.540067672729492, 10.913495063781738, 12.315370559692383],
        "Majority Class": 0
    }
    """
    try:
        query = info.query
        data = info.data
        sentiment = sentiment_service.sentiment_classifier(data, query)
        return {
            'result': sentiment
        }
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
