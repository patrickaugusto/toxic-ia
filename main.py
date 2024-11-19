from fastapi import FastAPI
from pydantic import BaseModel
from detoxify import Detoxify
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = Detoxify('multilingual')

class TextRequest(BaseModel):
    text: str

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/check_toxicity")
def check_toxicity(request: TextRequest):
    text = request.text
    result = model.predict([text])

    return {"text": text, "result": result}
