from app.utils.service_manager import ServiceManager

from app.services.ner_service import NERService
from app.services.paraphraser_service import ParaphraseService

def get_ner_service():
    return ServiceManager.get_service(NERService)

def get_paraphrase_service():
    return ServiceManager.get_service(ParaphraseService)