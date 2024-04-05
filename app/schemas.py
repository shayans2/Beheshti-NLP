from pydantic import BaseModel


class NERSchema(BaseModel):
    query: str


class ParaphraserSchema(BaseModel):
    query: str


class KeywordExtractorSchema(BaseModel):
    query: str
    top_n: int = 10
