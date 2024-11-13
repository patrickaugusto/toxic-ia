from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from detoxify import Detoxify
from fastapi.responses import JSONResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite acesso de qualquer origem; para segurança, pode-se restringir a URLs específicas.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextRequest(BaseModel):
    text: str

@app.post("/check_toxicity")
async def check_toxicity(request: TextRequest):
    text = request.text
    model = Detoxify('multilingual') 
    result = model.predict([text])
    
    return JSONResponse(content={"text": text, "result": result})

