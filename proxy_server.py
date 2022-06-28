from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import requests
import time
from starlette.background import BackgroundTask
import json


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

#TODO store in database
model_domains = {
    'Transformer': 'https://ura.hcmut.edu.vn/NMT/api1',
    'PhoBERT-fused NMT': 'https://ura.hcmut.edu.vn/NMT/api1',
    'Loanformer': 'https://ura.hcmut.edu.vn/NMT/api1',
    'BartPho': 'https://ura.hcmut.edu.vn/NMT/api1',
    'Combined': 'https://ura.hcmut.edu.vn/NMT/api1',
    'BARTphoEncoderPGN': 'https://ura.hcmut.edu.vn/NMT/api1',
    'PE-PD-PGN': 'https://ura.hcmut.edu.vn/NMT/api1',
    'M2M': 'https://ura.hcmut.edu.vn/NMT/api2',
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
    domain = model_domains[model] #TODO query domain from database

    content = await file.read()
    text = content.decode('utf-8')
    response = requests.post(
        f'{domain}/translate/text', 
        json={
            'text': text, 
            'model': model
        }
    )
    responseData = json.loads(response.content.decode('utf-8'))
    translated_text = responseData['ResultObj']['tgt'] #TODO handle failure cases

    tmp_dir = 'tmp_files'
    isExist = os.path.exists(tmp_dir)
    if not isExist: 
        os.makedirs(tmp_dir)
        print(f'Directory "{tmp_dir}" is created!')

    filename = str(int(time.time())) + file.filename
    with open(f'{tmp_dir}/{filename}', encoding='utf-8', mode='w') as f:
        f.write('\n'.join(translated_text))
    
    def cleanup(filename):
        os.remove(f'{tmp_dir}/{filename}')

    return FileResponse(
        f'{tmp_dir}/{filename}',
        background=BackgroundTask(cleanup, filename),
    )


@app.get("/models")
async def getModels():
    #TODO query models from database
    return {
        'models': list(model_domains.keys())
    }
