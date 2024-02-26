from app.models.paraphraser import ParaphraseModel
from transformers import T5Tokenizer

from app.config.settings import PARAPHRASER_MODEL_NAME

class ParaphraseService:
    def __init__(self, model_checkpoint_path: str):
        self.model_checkpoint_path = model_checkpoint_path
        self.model = ParaphraseModel.load_from_checkpoint(model_checkpoint_path)
        self.tokenizer = T5Tokenizer.from_pretrained(PARAPHRASER_MODEL_NAME)
        self.model.freeze()

    def paraphrase(self, text: str):
        text_encoding = self.tokenizer(
            text,
            max_length= 90,
            padding='max_length',
            return_attention_mask= True,
            add_special_tokens= True,
            return_tensors= "pt"
        )

        generated_ids = self.model.model.generate(
            input_ids= text_encoding["input_ids"],
            attention_mask= text_encoding["attention_mask"],
            max_length= 512,
            num_beams= 2,
            early_stopping= True
        )

        preds =  [
            self.tokenizer.decode(gen_id, skip_special_tokens= True, clean_up_tokenization_spaces= True)
            for gen_id in generated_ids
        ]

        return "".join(preds)