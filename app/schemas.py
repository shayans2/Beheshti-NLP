from pydantic import BaseModel

class NERSchema(BaseModel):
    query: str

class ParaphraserSchema(BaseModel):
    query: str


class ImageSearchQuerySchema(BaseModel):
    query: str
    k:int
    