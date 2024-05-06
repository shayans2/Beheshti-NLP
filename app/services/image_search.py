from transformers import pipeline
from hazm import Normalizer
import numpy as np 
import pandas as pd 
from tqdm import tqdm
from PIL import Image
import faiss
import pickle
import matplotlib.pyplot as plt

class ImageEmbeddingService:
    def __init__(self, clip_model, clip_processor, mt5_model, mt5_tokenizer):
        self.clip_model = clip_model
        self.clip_processor = clip_processor
        self.mt5_model = mt5_model
        self.mt5_tokenizer = mt5_tokenizer

    def process_image_dataset(self, image_infos):
        images_embeds = []
        for image_info in tqdm(image_infos, desc="Processing", unit="item"):
            current_image = Image.open(image_info)
            image_inputs = self.clip_processor(images=current_image, return_tensors="pt", padding=True)
            image_feature = self.clip_model.get_image_features(**image_inputs).detach().cpu().numpy()
            images_embeds.extend(image_feature)
            current_image.close()
        return np.stack(images_embeds)

    def get_query_embedding(self, query):
        tokenized_caps = self.clip_tokenizer(query, padding=True, return_tensors="pt")
        text_feature = self.clip_model.get_text_features(**tokenized_caps).detach().cpu().numpy()
        return np.stack(text_feature)

    def run_translation_model(self, input_string, **generator_args):
        input_ids = self.mt5_tokenizer.encode(input_string, return_tensors="pt")
        res = self.mt5_model.generate(input_ids, **generator_args)
        output = self.mt5_tokenizer.batch_decode(res, skip_special_tokens=True)
        return str(output[0])

    def search_similar_images(self, query_embedding, index, images_path, k=5):
        query_embedding = query_embedding / np.linalg.norm(query_embedding, ord=2, axis=-1, keepdims=True)
        distances, indices = index.search(query_embedding.reshape(1, -1), k)
        distances = distances[0]
        indices = indices[0]
        indices_distances = list(zip(indices, distances))
        indices_distances.sort(key=lambda x: x[1])
        similar_images = []
        for idx, distance in indices_distances:
            path = images_path[idx]
            similar_images.append(path)
        return similar_images

if __name__ == "__main__":
    image_root_path = '/kaggle/input/flickr30k/Images/'
    captions = pd.read_csv('/kaggle/input/flickr30k/captions.txt')

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    
    model_size = "base"
    model_name = f"persiannlp/mt5-{model_size}-parsinlu-opus-translation_fa_en"
    mt5_tokenizer = MT5Tokenizer.from_pretrained(model_name)
    mt5_model = MT5ForConditionalGeneration.from_pretrained(model_name)

    images_path = captions['image'].apply(lambda x: image_root_path + x).unique()

    embedding_service = ImageEmbeddingService(clip_model, clip_processor, mt5_model, mt5_tokenizer)

    embeds = embedding_service.process_image_dataset(images_path[:8000])
    with open('flicker30k_image_embeddings.pkl', 'wb') as f:
        pickle.dump(embeds, f)
    with open('/kaggle/working/flicker30k_image_embeddings.pkl', 'rb') as fp:
        embeds = pickle.load(fp)

    index = faiss.IndexFlatIP(embeds.shape[1])
    index.add(embeds)

    query = 'سگی در حال دویدن'
    query_embedding = embedding_service.get_query_embedding(embedding_service.run_translation_model(query))

    similar_images = embedding_service.search_similar_images(query_embedding, index, images_path)

    for img_path in similar_images:
        im = Image.open(img_path)
        plt.imshow(im)
        plt.show()
