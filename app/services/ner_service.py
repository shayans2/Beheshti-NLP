from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
from hazm import Normalizer
from app.services.transformers_service import TransformersService
from app.config.settings import NER_MODEL_NAME

class NERService:
    """
    Service for Named Entity Recognition (NER) in Persian language using pre-trained Transformer models.

    This service utilizes a pre-trained Transformer model to perform NER on Persian text, identifying entities such as 
    organizations, money, locations, persons, time, date, and percent. The service normalizes the input text, tokenizes it, 
    and processes it through the model to extract entities.

    Attributes:
    ----------
    bert_service : TransformersService
        A service for handling the loading and management of the Transformer model and tokenizer.
    normalizer : hazm.Normalizer
        Normalizer for preprocessing Persian text.
    ner_pipeline : pipeline
        The Hugging Face pipeline for performing NER.

    Methods:
    -------
    __init__():
        Initializes the NERService instance with the required model, tokenizer, and normalizer.
    
    load_model():
        Loads the pre-trained NER model and tokenizer.

    get_full_entity_names(text: str):
        Processes the input text to extract named entities and returns them grouped by entity types.
    """
    def __init__(self):
        """
        Initializes the NERService instance.

        This method initializes the `TransformersService` with the model and tokenizer specified for NER and sets up the normalizer.
        """
        self.bert_service = TransformersService(model=AutoModelForTokenClassification, tokenizer=AutoTokenizer, model_name_or_path=NER_MODEL_NAME)
        self.normalizer = Normalizer()
        self.ner_pipeline = None

    def load_model(self):
        """
        Loads the pre-trained NER model and tokenizer.

        This method calls the `load_model` method of `TransformersService` to load the pre-trained model and tokenizer for NER.
        """
        self.bert_service.load_model()

    def get_full_entity_names(self, text: str):
        """
        Processes the input text to extract named entities and returns them grouped by entity types.

        This method normalizes the input text, processes it through the NER model, and groups the recognized entities by type. 
        It supports entities like organization, money, location, person, time, date, and percent.

        Parameters:
        ----------
        text : str
            The input text to be processed for named entity recognition.

        Returns:
        -------
        dict
            A dictionary where the keys are entity types (e.g., 'organization', 'money') and the values are lists of recognized entities.
        
        Raises:
        ------
        ValueError:
            If the model is not loaded before calling this method.
        """
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
