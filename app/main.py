import asyncio
from fastapi import FastAPI
from app.api.router import api_router
from contextlib import asynccontextmanager
from app.services.index import get_ner_service, get_paraphrase_service, get_intent_service, get_keyword_extraction_service

async def load_model_async(service):
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, service.load_model)

@asynccontextmanager
async def lifespan(app: FastAPI):
    ner_service = get_ner_service()
    paraphrase_service = get_paraphrase_service()
    intent_service = get_intent_service()
    keyword_extraction_service = get_keyword_extraction_service()

    # Create background tasks to load services
    asyncio.create_task(load_model_async(ner_service))
    asyncio.create_task(load_model_async(paraphrase_service))
    asyncio.create_task(load_model_async(intent_service))
    asyncio.create_task(load_model_async(keyword_extraction_service))


    yield
    # Any shutdown procedures goes here

app = FastAPI(title="NLP Services with FastAPI", version="1.0.0", lifespan=lifespan)
app.include_router(api_router)