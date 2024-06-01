import torch
import re
from hazm import Normalizer
from transformers import AutoTokenizer, AutoModel
from app.config.settings import BERT_BASE_MODEL, BERT_BASE_TOKENIZER

class SentimentService:
    """
    Service for Intent Classification using pre-trained Transformer models.

    This service is responsible for loading a pre-trained Transformer model,
    tokenizing input sentences, obtaining their representations, and performing
    intent classification.

    Attributes:
    ----------
    _config : transformers.PretrainedConfig
        Configuration of the pre-trained Transformer model.
    _model : transformers.PreTrainedModel
        The pre-trained Transformer model for sentiment classification.
    _tokenizer : transformers.PreTrainedTokenizer
        Tokenizer associated with the pre-trained Transformer model.
    _normalizer : hazm.Normalizer
        Normalizer for preprocessing Persian text.
    loaded : bool
        Flag indicating whether the model and tokenizer are loaded.
    """
    def __init__(self):
        """
        Initializes the SentimentService instance.
        """
        self._model = None
        self._tokenizer = None
        self._normalizer = None
        self.loaded = False

    def load_model(self):
        """
        Loads the pre-trained Transformer model and tokenizer.
        """
        self._model = AutoModel.from_pretrained(BERT_BASE_MODEL)
        self._tokenizer = AutoTokenizer.from_pretrained(BERT_BASE_TOKENIZER)
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
            raise ValueError("Sentiment Classification Model not loaded. Call load_model() first.")
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
            raise ValueError("Sentiment Classification Tokenizer not loaded. Call load_model() first.")
        return self._tokenizer

    def get_representation(self, input_text: str) -> torch.tensor:
        """
        Obtains the representation of a sentence using the pre-trained Transformer model.

        Parameters:
        ----------
        sentence : str
            The input sentence to be tokenized and processed.

        Returns:
        -------
        torch.tensor
            The last hidden state of the model's output for the input sentence.
        """

        model = self.get_model()
        tokenizer = self.get_tokenizer()
        
        # Tokenize the example string
        tokens = tokenizer(input_text, return_tensors='pt')
        
        # Forward pass to obtain the model's representation
        with torch.no_grad():
            outputs = model(**tokens)
        last_hidden_states = outputs.last_hidden_state
        return last_hidden_states

    def most_repeated_element(self, tensor: torch.tensor):
        """
        Finds the most repeated element in a tensor.

        Parameters:
        ----------
        tensor : torch.tensor
            Input tensor from which to find the most repeated element.

        Returns:
        -------
        int or float
            The most repeated element in the tensor.

        Raises:
        ------
        ValueError:
            If the input is not a tensor.
        """

        # Check if the input is a torch.Tensor
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("Input must be a torch.Tensor")
        
        # Flatten the tensor to a 1D array
        flattened_tensor = tensor.view(-1)

        # Count the occurences of each unique element
        unique_elements, counts = torch.unique(flattened_tensor, return_counts=True)

        # Find the index of the element with the maximum count
        max_count_index = torch.argmax(counts)

        # Get the most repeated element
        most_repeated = unique_elements[max_count_index].item()

        return most_repeated

    def sentiment_classifier(self, data:dict, target_sentence:str) -> dict:
        """
        Classifies the sentiment of the given sentence based on the provided training data.

        Parameters:
        ----------
        data : dict
            A dictionary where keys are sentiment labels and values are lists of example sentences.
            Example: {"مثبت": ["غذا گرم و خوشمزه بود.", ...], ...}
        sentence : str
            The input sentence to classify.
            Example: "غذا زود و سریع به دستم رسید"

        Returns:
        -------
        dict
            A dictionary with indices, values of the nearest neighbors, and the majority class.
        """

        j = 0
        for class_name in data.keys():
            class_point = self.get_representation(class_name)
            class_points = []
            for sentence in data[class_name]:
                class_points.append(torch.mean(self.get_representation(self._normalizer.normalize(sentence)), dim=1))
            if j == 0:
                points = class_points
            
            points += class_points

            j += 1
        sentence_rep = torch.mean(self.get_representation(self._normalizer.normalize(target_sentence)), dim=1)
        data = torch.concat(points, dim=0)
        dist = torch.norm(data - sentence_rep, dim=1, p=None)
        knn = dist.topk(4, largest=False)
        return {'Indices': torch.floor(knn.indices / 5).tolist(), 'Values': knn.values.tolist(), 'Majority Class': self.most_repeated_element(torch.floor(knn.indices / 5))}
