class TransformersService:
    def __init__(self, model, tokenizer, model_name_or_path: str):
        self.model_name_or_path = model_name_or_path
        self.model_class = model
        self.tokenizer_class = tokenizer
        self._model = None
        self._tokenizer = None
        self.loaded = False

    def load_model(self):
        print("Loading model...")
        self._tokenizer = self.tokenizer_class.from_pretrained(self.model_name_or_path)
        self._model = self.model_class.from_pretrained(self.model_name_or_path)
        self.loaded = True
        print("Model loaded, setting loaded to True")

    def get_model(self):
        if not self.loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        return self._model

    def get_tokenizer(self):
        if not self.loaded:
            raise ValueError("Tokenizer not loaded. Call load_model() first.")
        return self._tokenizer
