from transformers import pipeline
from hazm import Normalizer

class NERService:
    def __init__(self, bert_service):
        self.bert_service = bert_service
        self.normalizer = Normalizer()

    def get_full_entity_names(self, text: str):
        """
        Extracts and constructs full entity names from the given text.

        Args:
            text (str): The text from which to extract entities.

        Returns:
            dict: A dictionary containing lists of entities for each entity type.
        """
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

        tokenizer = self.bert_service.get_tokenizer()
        model = self.bert_service.get_model()
        ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

        for entity in ner_pipeline(normalized_text):
            entity_label = entity['entity']
            entity_type = entity_label.split('-')[1] if '-' in entity_label else None

            if entity_type in entity_groups:
                if 'B-' in entity_label:
                    # Start of a new entity
                    if current_entity_words is not None:
                        # Save the previous entity
                        entity_groups[current_entity_type].append(current_entity_words.replace(' ##', '').strip())
                    # Start a new entity
                    current_entity_words = entity['word']
                    current_entity_type = entity_type
                elif 'I-' in entity_label and current_entity_type == entity_type:
                    # Continuation of the current entity
                    current_entity_words += ' ' + entity['word'].replace(' ##', '')

        # Add the last entity if there is one being constructed
        if current_entity_words is not None:
            entity_groups[current_entity_type].append(current_entity_words.strip())

        return entity_groups