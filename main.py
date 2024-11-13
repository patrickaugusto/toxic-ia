from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from detoxify import Detoxify
from pydantic import BaseModel
from fastapi.responses import JSONResponse

app = FastAPI()

# Habilitar CORS para permitir acesso a partir do front-end
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
