from app.models.paraphraser import ParaphraseModel
from transformers import T5Tokenizer
from app.config.settings import PARAPHRASER_MODEL_NAME, PARAPHRASER_MODEL_PATH

class ParaphraseService:
    """
    Service for Paraphrasing Persian text using a pre-trained T5 Transformer model.

    This service leverages a fine-tuned T5 model to generate paraphrases of given Persian text. 
    It handles the loading of the model and tokenizer, and provides methods to generate paraphrased text.

    Attributes:
    ----------
    _model : ParaphraseModel
        The pre-trained and fine-tuned T5 model for paraphrasing.
    _tokenizer : T5Tokenizer
        The tokenizer associated with the T5 model.
    loaded : bool
        Flag indicating whether the model and tokenizer are loaded.

    Methods:
    -------
    __init__():
        Initializes the ParaphraseService instance with default attributes.

    load_model():
        Loads the pre-trained paraphrasing model and tokenizer.

    get_model():
        Returns the loaded model, raises an error if the model is not loaded.

    get_tokenizer():
        Returns the loaded tokenizer, raises an error if the tokenizer is not loaded.

    paraphrase(text: str):
        Generates a paraphrase of the given text using the pre-trained model.
    """
    def __init__(self):
        """
        Initializes the ParaphraseService instance.

        This method initializes the `_model` and `_tokenizer` attributes to None and sets the `loaded` flag to False.
        """
        self._model = None
        self._tokenizer = None
        self.loaded = False

    def load_model(self):
        """
        Loads the pre-trained paraphrasing model and tokenizer.

        This method loads the model from the specified checkpoint and freezes its parameters to prevent further training.
        It also loads the associated tokenizer from the pre-trained model name. After loading, it sets the `loaded` flag to True.
        """
        self._model = ParaphraseModel.load_from_checkpoint(PARAPHRASER_MODEL_PATH)
        self._model.freeze()
        self._tokenizer = T5Tokenizer.from_pretrained(PARAPHRASER_MODEL_NAME)
        self.loaded = True

    def get_model(self):
        """
        Returns the loaded paraphrasing model.

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

    def paraphrase(self, text: str):
        """
        Generates a paraphrase of the given text using the pre-trained model.

        This method tokenizes the input text, generates paraphrased text using the model, and decodes the generated tokens to return the final paraphrase.

        Parameters:
        ----------
        text : str
            The input text to be paraphrased.

        Returns:
        -------
        str
            The paraphrased text.

        Raises:
        ------
        ValueError:
            If the model or tokenizer is not loaded before calling this method.
        """
        model = self.get_model()
        tokenizer = self.get_tokenizer()

        text_encoding = tokenizer(
            text,
            max_length=90,
            padding='max_length',
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )

        generated_ids = model.model.generate(
            input_ids=text_encoding["input_ids"],
            attention_mask=text_encoding["attention_mask"],
            max_length=512,
            num_beams=2,
            early_stopping=True
        )

        preds = [
            tokenizer.decode(gen_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for gen_id in generated_ids
        ]

        return "".join(preds)
