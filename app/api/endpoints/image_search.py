from fastapi import APIRouter, Depends, File, UploadFile, HTTPException
from app.dependencies import get_transformers_service
from app.schemas import NERSchema
from typing import Any
import logging
import zipfile
import os
import pickle

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/process-images", response_model=Any)
def process_images(zip_file: UploadFile = File(...), transformers_service = Depends(get_transformers_service)):
    try:
        # Create a temporary directory to extract the images
        with zipfile.ZipFile(zip_file.file, 'r') as zip_ref:
            extract_dir = "/tmp/extracted_images"
            zip_ref.extractall(extract_dir)

        image_infos = []
        for filename in os.listdir(extract_dir):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                # Assuming image file format is jpg or png, adjust as needed
                image_path = os.path.join(extract_dir, filename)
                image_infos.append(image_path)

        # Process the images (Replace this with your own image processing logic)
        processed_data = process_images_function(extract_dir)
        images_embeds = []
        for image_info in image_infos:
            current_image = Image.open(image_info)
            image_inputs = processor(images=current_image, return_tensors="pt", padding=True)
            image_feature = model.get_image_features(**image_inputs).detach().cpu().numpy()
            images_embeds.extend(image_feature)
            #self.save_images(image_feature,image_info)
            current_image.close()
        images_embeds = np.stack(images_embeds)

        # Serialize the processed data and save it as a .pkl file
        pkl_file_path = "/tmp/processed_data.pkl"
        with open(pkl_file_path, 'wb') as f:
            pickle.dump(processed_data, f)

        # Return the path to the .pkl file
        return {"pkl_file_path": pkl_file_path}
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

def process_images_function(extract_dir):
    # Placeholder function for image processing logic
    # Replace this function with your actual image processing logic
    processed_data = {}  # Replace this with the processed data
    return processed_data