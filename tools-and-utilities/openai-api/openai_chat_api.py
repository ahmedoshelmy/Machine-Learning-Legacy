import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai

# Set your OpenAI API key
openai.api_key = "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"

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # Reverting to the correct `openai.ChatCompletion.create` method
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": request.message}]
        )
        return {"response": response["choices"][0]["message"]["content"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
