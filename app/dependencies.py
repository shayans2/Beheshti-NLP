from transformers import AutoModelForTokenClassification, AutoTokenizer
from fastapi import Depends

from app.services.transformers_service import TransformersService
from app.services.ner_service import NERService
from app.services.paraphraser_service import ParaphraseService
from app.services.keyword_extractor_service import KeywordExtractorService

from app.config.settings import (
    NER_MODEL_NAME,
    PARAPHRASER_MODEL_PATH,
    KEYWORD_EXTRACTOR_MODEL_PATH,
)


ner_bert_service = TransformersService(
    model=AutoModelForTokenClassification,
    tokenizer=AutoTokenizer,
    model_name_or_path=NER_MODEL_NAME,
)
paraphrase_service = ParaphraseService(model_checkpoint_path=PARAPHRASER_MODEL_PATH)
keyword_extractor_service = KeywordExtractorService(
    model_file_path=KEYWORD_EXTRACTOR_MODEL_PATH
)


def get_ner_bert_service():
    return ner_bert_service


def get_ner_service(
    ner_bert_service: TransformersService = Depends(get_ner_bert_service),
):
    return NERService(ner_bert_service)


def get_paraphrase_service():
    return paraphrase_service


def get_keyword_extractor_service():
    return keyword_extractor_service
