from fastapi import FastAPI
from pydantic import BaseModel
from detoxify import Detoxify
from fastapi.responses import JSONResponse

app = FastAPI()

class TextRequest(BaseModel):
    text: str

@app.post("/check_toxicity")
async def check_toxicity(request: TextRequest):
    text = request.text
    model = Detoxify('multilingual') 
    result = model.predict([text])
    
    return JSONResponse(content={"text": text, "result": result})

