from app.utils.service_manager import ServiceManager

from app.services.ner_service import NERService
from app.services.paraphraser_service import ParaphraseService
from app.services.intent_service import IntentService
from app.services.keyword_extraction_service import KeywordExtractionService


def get_ner_service():
    return ServiceManager.get_service(NERService)


def get_paraphrase_service():
    return ServiceManager.get_service(ParaphraseService)


def get_intent_service():
    return ServiceManager.get_service(IntentService)


def get_keyword_extraction_service():
    return ServiceManager.get_service(KeywordExtractionService)
