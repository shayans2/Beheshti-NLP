from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
from hazm import Normalizer
from app.services.transformers_service import TransformersService
from app.config.settings import NER_MODEL_NAME

class NERService:
    def __init__(self):
        self.bert_service = TransformersService(model=AutoModelForTokenClassification, tokenizer=AutoTokenizer, model_name_or_path=NER_MODEL_NAME)
        self.normalizer = Normalizer()
        self.ner_pipeline = None

    def load_model(self):
        self.bert_service.load_model()

    def get_full_entity_names(self, text: str):
        if not self.bert_service.loaded:
            raise ValueError("Model not loaded. Call load_model() first. NER_SERVICE")

        if self.ner_pipeline is None:
            self.ner_pipeline = pipeline("ner", model=self.bert_service.get_model(), tokenizer=self.bert_service.get_tokenizer())

        entity_groups = {
            'organization': [],
            'money': [],
            'location': [],
            'person': [],
            'time': [],
            'date': [],
            'percent': []
        }

        current_entity_words = None
        current_entity_type = None

        normalized_text = self.normalizer.normalize(text)

        for entity in self.ner_pipeline(normalized_text):
            entity_label = entity['entity']
            entity_type = entity_label.split('-')[1] if '-' in entity_label else None

            if entity_type in entity_groups:
                if 'B-' in entity_label:
                    if current_entity_words is not None:
                        entity_groups[current_entity_type].append(current_entity_words.replace(' ##', '').strip())
                    current_entity_words = entity['word']
                    current_entity_type = entity_type
                elif 'I-' in entity_label and current_entity_type == entity_type:
                    current_entity_words += ' ' + entity['word'].replace(' ##', '')

        if current_entity_words is not None:
            entity_groups[current_entity_type].append(current_entity_words.strip())

        return entity_groups
