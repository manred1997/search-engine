from typing import Union

from fastapi import Body, FastAPI
from fastapi.params import Body
from pydantic import BaseModel

app = FastAPI()


class Post(BaseModel):
    input_sentence: str


@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.post("/spell_check")
async def spell_check(payLoad: dict = Body(...)):
    corrected_text = "Hello word"
    return {"corrected_text": corrected_text}


@app.get("/openapi.json")
async def get_openapi_spec():
    return app.openapi()


@app.get("/docs")
async def get_docs():
    return app.docs()
