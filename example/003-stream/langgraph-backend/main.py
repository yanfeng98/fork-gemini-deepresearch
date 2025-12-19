from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from graph import llm

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat-stream")
def chat_stream(req: ChatRequest):
    async def event_generator():
        async for chunk in llm.astream([
            HumanMessage(content=req.message)
        ]):
            yield chunk.content

    return StreamingResponse(
        event_generator(),
        media_type="text/plain"
    )