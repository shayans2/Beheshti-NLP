from app.models.paraphraser import ParaphraseModel
from transformers import T5Tokenizer
from app.config.settings import PARAPHRASER_MODEL_NAME, PARAPHRASER_MODEL_PATH

class ParaphraseService:
    def __init__(self):
        self._model = None
        self._tokenizer = None
        self.loaded = False

    def load_model(self):
        self._model = ParaphraseModel.load_from_checkpoint(PARAPHRASER_MODEL_PATH)
        self._model.freeze()
        self._tokenizer = T5Tokenizer.from_pretrained(PARAPHRASER_MODEL_NAME)
        self.loaded = True

    def get_model(self):
        if not self.loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        return self._model

    def get_tokenizer(self):
        if not self.loaded:
            raise ValueError("Tokenizer not loaded. Call load_model() first.")
        return self._tokenizer

    def paraphrase(self, text: str):
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
