'''
    Instructions for Using the Template:
    1. Import Necessary Modules: Make sure to import the required libraries and modules.
    2. Class Name and Docstring: Replace ServiceTemplate and [Your Task Description] with the specific name and description of the new service you are implementing.
    3. Attributes: Update the attributes and their descriptions to match the requirements of the new service.
    4. Methods: Implement necessary methods (load_model, get_model, get_tokenizer, task_method) and update their docstrings and logic according to the new service.
    5. Settings: Ensure that MODEL_NAME and TOKENIZER_NAME are defined in app/config/settings.py and correspond to the appropriate pre-trained model and tokenizer for your service.
'''

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModel
from hazm import Normalizer
from app.config.settings import MODEL_NAME, TOKENIZER_NAME

class ServiceTemplate:
    """
    Service for [Your Task Description] using pre-trained Transformer models.

    This service is responsible for loading a pre-trained Transformer model,
    tokenizing input sentences, obtaining their representations, and performing
    [Your Task Description].

    Attributes:
    ----------
    _config : transformers.PretrainedConfig
        Configuration of the pre-trained Transformer model.
    _model : transformers.PreTrainedModel
        The pre-trained Transformer model for [Your Task Description].
    _tokenizer : transformers.PreTrainedTokenizer
        Tokenizer associated with the pre-trained Transformer model.
    _normalizer : hazm.Normalizer
        Normalizer for preprocessing Persian text.
    loaded : bool
        Flag indicating whether the model and tokenizer are loaded.
    """
    def __init__(self):
        """
        Initializes the ServiceTemplate instance.
        """
        self._config = None
        self._model = None
        self._tokenizer = None
        self._normalizer = None
        self.loaded = False
    
    def load_model(self):
        """
        Loads the pre-trained Transformer model, tokenizer, and normalizer.
        """
        self._model = AutoModel.from_pretrained(MODEL_NAME)
        self._tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        self._normalizer = Normalizer()
        self.loaded = True
    
    def get_model(self):
        """
        Returns the loaded pre-trained Transformer model.

        Raises:
        ------
        ValueError:
            If the model is not loaded.
        """
        if not self.loaded:
            raise ValueError("[Your Task Description] Model is not loaded. Call load_model() first.")
        return self._model

    def get_tokenizer(self):
        """
        Returns the loaded tokenizer.

        Raises:
        ------
        ValueError:
            If the tokenizer is not loaded.
        """
        if not self.loaded:
            raise ValueError("[Your Task Description] Tokenizer is not loaded. Call load_model() first.")
        return self._tokenizer


    def task_method(self, input):
        """
        [Your Task Method Description]

        Parameters:
        ----------
        input

        Returns:
        -------
        An appropriate output.
        """
        pass
