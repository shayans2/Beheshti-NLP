class TransformersService:
    """
    A service interface for loading and managing pre-trained Transformer models and tokenizers.

    This class provides a standard interface for loading pre-trained Transformer models and their corresponding tokenizers.
    It ensures that the model and tokenizer are only loaded once and provides methods to access them.

    Attributes:
    ----------
    model_name_or_path : str
        The name or path of the pre-trained model to be loaded.
    model_class : type
        The class of the model to be loaded, typically from the `transformers` library.
    tokenizer_class : type
        The class of the tokenizer to be loaded, typically from the `transformers` library.
    _model : transformers.PreTrainedModel
        The loaded pre-trained model instance.
    _tokenizer : transformers.PreTrainedTokenizer
        The loaded tokenizer instance.
    loaded : bool
        A flag indicating whether the model and tokenizer have been loaded.

    Methods:
    -------
    __init__(model, tokenizer, model_name_or_path: str):
        Initializes the TransformersService instance with the specified model and tokenizer classes and model name/path.

    load_model():
        Loads the pre-trained model and tokenizer from the specified model name or path.

    get_model():
        Returns the loaded model, raises an error if the model is not loaded.

    get_tokenizer():
        Returns the loaded tokenizer, raises an error if the tokenizer is not loaded.
    """
    def __init__(self, model, tokenizer, model_name_or_path: str):
        """
        Initializes the TransformersService instance.

        Parameters:
        ----------
        model : type
            The class of the model to be loaded, typically from the `transformers` library.
        tokenizer : type
            The class of the tokenizer to be loaded, typically from the `transformers` library.
        model_name_or_path : str
            The name or path of the pre-trained model to be loaded.
        """
        self.model_name_or_path = model_name_or_path
        self.model_class = model
        self.tokenizer_class = tokenizer
        self._model = None
        self._tokenizer = None
        self.loaded = False

    def load_model(self):
        """
        Loads the pre-trained model and tokenizer.

        This method loads the model and tokenizer from the specified model name or path.
        After loading, it sets the `loaded` flag to True.
        """
        self._tokenizer = self.tokenizer_class.from_pretrained(self.model_name_or_path)
        self._model = self.model_class.from_pretrained(self.model_name_or_path)
        self.loaded = True

    def get_model(self):
        """
        Returns the loaded model.

        Raises:
        ------
        ValueError:
            If the model is not loaded before calling this method.
        """
        if not self.loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        return self._model

    def get_tokenizer(self):
        """
        Returns the loaded tokenizer.

        Raises:
        ------
        ValueError:
            If the tokenizer is not loaded before calling this method.
        """
        if not self.loaded:
            raise ValueError("Tokenizer not loaded. Call load_model() first.")
        return self._tokenizer
