# Giumeh NLP Service

Temp repo

## Prepreation

### Environment

Create a new environment and install dependencies with `requirement.txt`:

```shell
conda create -n giumeh

conda activate giumeh

conda install --file requirements.txt
```

## Launch

Run the app using the following command:

```shell
uvicorn app.main:app
```

Then load localhost:8000/docs to use the Swagger app.
