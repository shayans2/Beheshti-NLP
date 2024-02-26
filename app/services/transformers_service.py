class TransformersService:
    def __init__(self, model, tokenizer, model_name_or_path: str):
        self.model_name_or_path = model_name_or_path
        self.model = model
        self.tokenizer = tokenizer
        self.load_model()

    def load_model(self):
        """
        Loads the Transformer model and tokenizer.
        """
        self.tokenizer_loader = self.tokenizer.from_pretrained(self.model_name_or_path)
        self.model_loader = self.model.from_pretrained(self.model_name_or_path)
    
    def get_model(self):
        """
        Returns the loaded Transformer model.
        """
        return self.model_loader
    
    def get_tokenizer(self):
        """
        Returns the tokenizer associated with the Transformer model.
        """
        return self.tokenizer_loader
    