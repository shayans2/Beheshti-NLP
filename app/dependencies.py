from transformers import AutoModelForTokenClassification, AutoTokenizer,CLIPModel,AutoProcessor,MT5ForConditionalGeneration,MT5Tokenizer
from fastapi import Depends

from app.services.transformers_service import TransformersService
from app.services.ner_service import NERService
from app.services.paraphraser_service import ParaphraseService
from app.services.image_search_service import ImageEmbeddingService

from app.config.settings import NER_MODEL_NAME, PARAPHRASER_MODEL_PATH,CLIP_MODEL_PATH,MT5_MODLE_PATH


ner_bert_service = TransformersService(model=AutoModelForTokenClassification, tokenizer=AutoTokenizer, model_name_or_path=NER_MODEL_NAME)
paraphrase_service = ParaphraseService(model_checkpoint_path=PARAPHRASER_MODEL_PATH)
image_search_service = ImageEmbeddingService(clip_model_name=CLIP_MODEL_PATH,mt5_model_name=MT5_MODLE_PATH,clip_model=CLIPModel,clip_processor=AutoProcessor,clip_tokenizer=AutoTokenizer,mt5_model=MT5ForConditionalGeneration,mt5_tokenizer=MT5Tokenizer)
def get_ner_bert_service():
    return ner_bert_service
def get_image_search_service():
    return image_search_service
def get_ner_service(ner_bert_service: TransformersService = Depends(get_ner_bert_service)):
    return NERService(ner_bert_service)

def get_paraphrase_service():
    return paraphrase_service
