from pydantic import BaseModel
from typing import Dict

class NERSchema(BaseModel):
    query: str

class ParaphraserSchema(BaseModel):
    query: str

class IntentSchema(BaseModel):
    """
    Schema for intent classification input using Pydantic.

    This schema defines the structure of the input data required for intent classification.
    
    Attributes:
    ----------
    query : str
        The input sentence to classify.
    data : Dict
        A dictionary where keys are intent labels and values are lists of example sentences.
    """
    query: str
    data: Dict

class SentimentSchema(BaseModel):
    """
    Schema for sentiment classification input using Pydantic.

    This schema defines the structure of the input data required for sentiment classification.
    
    Attributes:
    ----------
    query : str
        The input sentence to classify.
    data : Dict
        A dictionary where keys are sentiment labels and values are lists of example sentences.
    """
    query: str
    data: Dict