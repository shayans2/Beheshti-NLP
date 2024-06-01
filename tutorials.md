# Giumeh Project Documentation

## Overview
Giumeh is an NLP project focused on developing tools for Persian language tasks using FastAPI and PyTorch. This guide is intended to help new developers deploy a new model within the Giumeh project. Follow the steps outlined in this document to integrate your model into the existing framework.

## Project Structure
Below is the directory structure of the Giumeh project:
```
.
├── README.md
├── app
│   ├── api
│   │   ├── endpoints
│   │   │   ├── intent.py
│   │   │   ├── ner.py
│   │   │   └── paraphraser.py
│   │   └── router.py
│   ├── config
│   │   └── settings.py
│   ├── main.py
│   ├── model_files
│   │   ├── paraphraser.ckpt
│   │   ├── pos_tagger.model
│   │   └── tf-idf-model.pkl
│   ├── models
│   │   └── paraphraser.py
│   ├── schemas.py
│   ├── services
│   │   ├── index.py
│   │   ├── intent_service.py
│   │   ├── ner_service.py
│   │   ├── paraphraser_service.py
│   │   └── transformers_service.py
│   └── utils
│       └── service_manager.py
├── requirements.txt
└── tree.txt
```

## Step-by-Step Guide to Deploying a New Model

To deploy a new model in the Giumeh project, follow these steps:

1. __Configure Model Paths__
   In `app/config/settings.py`, define any static final variables needed for your model paths. This helps avoid typos and redundancy. Here's an example:
   ```python
    YOUR_SERVICE_MODEL_PATH = "/path/to/your/model"
    YOUR_SERVICE_TOKENIZER_PATH = "/path/to/your/tokenizer"
   ```

2. __Implement the Service__
    Navigate to `app/services` and create a new file named `{your_service}_service.py`. Implement your model in this file using the following template:
    ```python
    import torch
    from transformers import AutoConfig, AutoTokenizer, AutoModel
    from transformers import AutoTokenizer
    from app.config.settings import YOUR_SERVICE_MODEL_PATH, YOUR_SERVICE_TOKENIZER_PATH

    class {YourService}Service:
        """
        Service for {YourService} using pre-trained Transformer models.

        Attributes:
        ----------
        _config : transformers.PretrainedConfig
            Configuration of the pre-trained Transformer model.
        _model : transformers.PreTrainedModel
            The pre-trained Transformer model.
        _tokenizer : transformers.PreTrainedTokenizer
            Tokenizer associated with the pre-trained Transformer model.
        loaded : bool
            Flag indicating whether the model and tokenizer are loaded.
        """
        def __init__(self):
            self._config = None
            self._model = None
            self._tokenizer = None
            self._normalizer = None
            self.loaded = False
        
        def load_model(self):
            self._model = AutoModel.from_pretrained(YOUR_SERVICE_MODEL_PATH)
            self._tokenizer = AutoTokenizer.from_pretrained(YOUR_SERVICE_TOKENIZER_PATH)
            self._normalizer = Normalizer()
            self.loaded = True
        
        def get_model(self):
            if not self.loaded:
                raise ValueError("Model is not loaded. Call load_model() first.")
            return self._model

        def get_tokenizer(self):
            if not self.loaded:
                raise ValueError("Tokenizer is not loaded. Call load_model() first.")
            return self._tokenizer

        def {your_service}(self, data: dict, sentence: str):
            '''
                In this function, your model should get the appropriate input and return desired output.
            '''
            pass

    ```
3. __Define Request Schema (Optional)__
   If your service requires a custom input format, define it in `app/schemas.py`. Here's an example:
   ```python
    from pydantic import BaseModel
    from typing import Dict

    class {YourService}Schema(BaseModel):
        """
        Schema for {YourService} classification input using Pydantic.

        This schema defines the structure of the input data required for your service.
        
        Attributes:
        ----------
        query : str
            The input sentence to classify.
        data : Dict
            A dictionary where keys are intent labels and values are lists of example sentences.
        """
        query: str
        data: Dict
   ```

4. __Index Your Service__
   Navigate to `app/services/index.py`, import your service model and add a function like this at the end of it:
   ```python
    def get_{your service}_service():
        return ServiceManager.get_service({your service}Service)
   ```

5. __Create New Endpoint__
   Navigate to `app/api/endpoints` and create a new file named {your_service}.py. Use the following template for your endpoint:
   ```python
    from fastapi import APIRouter, Depends, HTTPException

    from app.services.index import get_{your_service}_service
    from app.schemas import {YourServiceSchema}

    from typing import Any
    import logging

    logger = logging.getLogger(__name__)
    router = APIRouter()

    @router.post("/{your_service}", response_model=Any)
    def {your_service}(text: {YourServiceSchema}, {your_service}_service = Depends(get_{your_service}_service)):
        """
        Endpoint for {your_service}.

        This endpoint receives a query and training data, processes them using the {your_service} service,
        and returns the output.

        Parameters:
        ----------
        text : {YourServiceSchema}
            The input data containing the query and training data.
        {your_service}_service : {YourService}Service
            The {your_service} service dependency.

        Returns:
        -------
        dict
            The results.

        Raises:
        ------
        HTTPException:
            If an internal server error occurs during service.
        """
        try:
            query = text.query
            data = text.data
            entities = {your_service}_service.{your_service}(data, query)
            return entities
        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error")

   ```

6. __Update the Router__
   Open `app/api/router.py` and add a line to include your new endpoint:
   ```python
    api_router.include_router({your_service}.router, tags=['{your_service}'])
   ```

7. __Update the Tree View of Project__
   Navigate to `Beheshti-NLP` directory and run following command in your terminal:
   ```terminal
   tree > tree.txt
   ```

8. __Create a Pull Request__
   After following all steps and write clear and explainable comments, create a pull request to main repository.

## Running the Project
To run the Giumeh project locally, follow these steps:

1. __Install Conda__
   If you haven't already, install Conda by following the instructions on [the official Conda website](https://www.anaconda.com/).
2. __Setup Environment__
   Navigate to the project directory and run the following command in your terminal:
   ```bash
    python setup_env.py
   ```
   This command will create a new Conda environment named `giumeh` and install all required modules and dependencies specified in `requirements.txt`.

3. __Activate Environment__
   Activate the `giumeh` environment using the following command:
   ```bash
    conda activate giumeh
   ```

4. __Run the Project__
   Once the environment is activate, run the following command to start the project services:
   ```bash
    uvicorn app.main:app --reload
   ```