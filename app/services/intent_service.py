import torch
from transformers import AutoConfig, AutoTokenizer, AutoModel
from hazm import Normalizer
from transformers import AutoTokenizer
from app.config.settings import BERT_BASE_MODEL, BERT_BASE_TOKENIZER

class IntentService:
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
        The pre-trained Transformer model for intent classification.
    _tokenizer : transformers.PreTrainedTokenizer
        Tokenizer associated with the pre-trained Transformer model.
    _normalizer : hazm.Normalizer
        Normalizer for preprocessing Persian text.
    loaded : bool
        Flag indicating whether the model and tokenizer are loaded.
    """
    def __init__(self):
        """
        Initializes the IntentService instance.
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
        self._model = AutoModel.from_pretrained(BERT_BASE_MODEL)
        self._tokenizer = AutoTokenizer.from_pretrained(BERT_BASE_TOKENIZER)
        self._normalizer = Normalizer()
    
    def get_model(self):
        """
        Returns the loaded pre-trained Transformer model.

        Raises:
        ------
        ValueError:
            If the model is not loaded.
        """
        if not self.loaded:
            raise ValueError("Intent Classification Model is not loaded. Call load_model() first.")
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
            raise ValueError("Intent Classification Tokenizer is not loaded. Call load_model() first.")
        return self._tokenizer

    def _get_representation(self, sentence: str) -> torch.tensor:
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

        # Tokenize the example string
        tokens = self._tokenizer(sentence, return_tensors='pt')

        # Forward pass to obtain the model's representation
        with torch.no_grad():
            outputs = self._model(**tokens)
        last_hidden_states = outputs.last_hidden_state
        return last_hidden_states

    def _most_repeated_element(self, tensor: torch.tensor):
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
            raise ValueError("Input to _most_repeated_element function should be a tensor.")
        
        # Flatten the tensor to a 1D array
        flattened_tensor = tensor.view(-1)

        # Count the occurences of each unique element
        unique_elements, counts = torch.unique(flattened_tensor, return_counts=True)

        # Find the index of the element with the maximum count
        max_count_index = torch.argmax(counts)

        # Get the most repeated element
        most_repeated = unique_elements[max_count_index].item()

        return most_repeated

    def intent_classifier(self, data: dict, sentence: str) -> dict:
        """
        Classifies the intent of the given sentence based on the provided training data.

        Parameters:
        ----------
        data : dict
            A dictionary where keys are intent labels and values are lists of example sentences.
            Example: {"رزرو غذا": ["یک پیتزا با پپرونی و قارچ بساز .", ...], ...}
        sentence : str
            The input sentence to classify.
            Example: "برای من یک کشک بادمجون سفارش بده"

        Returns:
        -------
        dict
            A dictionary with indices, values of the nearest neighbors, and the majority class.
        """
        data_points = []
        distinct_data_points = {}
        sentiment_points = []
        for key in data.keys():
            sentiment_points.append(self._get_representation(key))
            tmp = []
            sentences = data[key]
            for sent in sentences:
                tmp.append(torch.mean(self._get_representation(self._normalizer.normalize(sent)), dim=1))
            data_points += tmp
            distinct_data_points[key] = tmp
        
        target_rep = torch.mean(self._get_representation(self._normalizer.normalize(sentence)), dim=1)
        points = torch.concat(data_points, dim=0)
        dist = torch.norm(points - target_rep, dim=1, p=None)
        knn = dist.topk(3, largest=False)
        print({'Indices': torch.floor(knn.indices / len(sentiment_points)).tolist(), 'Values': knn.values.tolist(), 'Majority Class': self._most_repeated_element(torch.floor(knn.indices / len(sentiment_points)))})
        return {'Indices': torch.floor(knn.indices / len(sentiment_points)).tolist(), 'Values': knn.values.tolist(), 'Majority Class': self._most_repeated_element(torch.floor(knn.indices / len(sentiment_points)))}
        



