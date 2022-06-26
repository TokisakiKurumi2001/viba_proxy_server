from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import requests
import time
from starlette.background import BackgroundTask


class TranslationItem(BaseModel):
    text: str
    model: str


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_domains = {
    'Transformer': 'http://localhost:10002',
    'PhoBERT-fused NMT': 'http://localhost:10002',
    'Loanformer': 'http://localhost:10002',
    'BartPho': 'http://localhost:10002',
    'Combined': 'http://localhost:10002',
    'BARTphoEncoderPGN': 'http://localhost:10002',
    'PE-PD-PGN': 'http://localhost:10002',
    'M2M': 'http://localhost:10012',
}


@app.post("/translate/text")
async def translate(translation: TranslationItem):
    model = translation.model
    domain = model_domains[model]
    text = translation.text
    response = requests.post(
        domain + '/translate/text', json={'text': text, 'model': model})
    return response.json()


@app.post("/translate/file")
async def translateFile(file: UploadFile = File(...), model: str = Form(...)):
    content = await file.read()
    domain = model_domains[model]
    response = requests.post(domain + '/translate/file',
                             json={'file': content, 'model': model})
    return response


@app.get("/models")
async def getModels():
    return {
        'models': list(model_domains.keys())
    }
